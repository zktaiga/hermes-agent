"""Live cron controls use the scheduler's exact agent, not a route cache or durable lineage."""
import asyncio
import threading
import weakref
from types import SimpleNamespace

import pytest
import pytest_asyncio


class Agent:
    def __init__(self, sid, db):
        self.session_id, self._session_db = sid, db
        self.steered, self.stopped = [], []
        self._delegate_parent_ref: object | None = None

    def steer(self, text):
        self.steered.append(text)
        return True

    def redirect(self, text):
        return self.steer(text)

    def hard_interrupt(self, message=None):
        self.stopped.append(message)
        return True


@pytest_asyncio.fixture
async def fire(tmp_path, monkeypatch):
    from cron import jobs, scheduler
    from gateway import run, session_cron
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'prompt': 'work', 'model': 'fake'})
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    db = SessionDB(tmp_path / 'state.db')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_draining=False, session_store=store, adapters={}, _cached_agent_for=lambda key: None)
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    runner.session_authority = authority
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    receipt = await session_cron.operation(authority, 'submit', {'job_id': 'job', 'request_id': 'fire', 'extra_prompt': None})
    sid = receipt['session_id']
    ref = SessionRef(authority.profile_id, sid)
    row = claim_session_input(db, epoch=authority.epoch, session_id=sid)
    authority.executing[sid] = row['generation']
    authority.sessions[sid].event_stream.execution = {'execution_generation': row['generation']}
    policy = next(iter(runner.adapters.values())).policies[sid]
    actor = SessionActor(authority.sessions[sid].source.user_id, authority.profile_id,
                         frozenset({'session:control', 'session:read'}), 'fixture')
    connection = SimpleNamespace(authority=authority, actor=actor)
    agent = Agent(sid, db)
    monkeypatch.setattr(scheduler, '_prepare_job_prompt', lambda *a: (None, 'work'))
    monkeypatch.setattr(scheduler, '_reload_dotenv_and_publish_delivery_target', lambda *a: None)
    monkeypatch.setattr(scheduler, '_load_cron_job_config', lambda *a: SimpleNamespace(cfg={}, model='fake'))
    setup = SimpleNamespace(blocked=None, model='fake', fallback_notice=None)
    monkeypatch.setattr(scheduler, '_resolve_cron_agent_setup', lambda *a: setup)
    monkeypatch.setattr(scheduler, '_open_cron_session_db', lambda *a: db)
    monkeypatch.setattr(scheduler, '_construct_cron_agent', lambda *a, **kw: agent)
    monkeypatch.setattr(scheduler, '_final_response_from_result', lambda *a: 'done')
    monkeypatch.setattr(scheduler._FireAudit, 'write', lambda *a: None)
    finalizers = []

    def finalized(*args, **kwargs):
        assert authority.adopted.get(sid, (None, None))[1] is None
        with authority.sessions[sid].event_stream.lock:
            assert authority.steer_agent(ref, row['generation'], 'during teardown') == (None, False)
        finalizers.append('session')

    def teardown(*args, **kwargs):
        assert authority.adopted.get(sid, (None, None))[1] is None
        with authority.sessions[sid].event_stream.lock:
            assert authority.steer_agent(ref, row['generation'], 'during teardown') == (None, False)
        finalizers.append('agent')

    monkeypatch.setattr(scheduler, '_finalize_cron_session', finalized)
    monkeypatch.setattr(scheduler, '_teardown_cron_agent', teardown)
    yield SimpleNamespace(authority=authority, ref=ref, row=row, policy=policy, connection=connection,
                          agent=agent, finalizers=finalizers)
    db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fails', [False, True])
async def test_scheduler_adopts_live_cron_agent_and_releases_before_teardown(fire, monkeypatch, fails):
    from cron import scheduler
    from gateway import session_cron
    from gateway.session_busy_controls import correct
    from gateway.session_ancillary import read, subagent_control
    from gateway.session_contract import SessionActor, SessionRef
    from hermes_state_runtime import RuntimeStoreError
    from tools import delegate_tool_registry as registry

    a, ref, row = fire.authority, fire.ref, fire.row
    params = {'session_id': ref.session_id, 'execution_generation': row['generation'], 'text': 'before construction'}
    assert (await correct(fire.connection, ref, params, verb='steer'))['status'] == 'queued'
    await a.interrupt(fire.connection.actor, ref, row['generation'])
    entered, finish = threading.Event(), threading.Event()

    def conversation(agent, *args, **kwargs):
        assert agent is fire.agent and a.agent(ref) is agent
        assert agent.steered == ['before construction'] and len(agent.stopped) == 1
        entered.set()
        assert finish.wait(10)
        if fails:
            raise ValueError('fixture conversation failed')
        return {}

    monkeypatch.setattr(scheduler, '_run_agent_with_watchdog', conversation)
    task = asyncio.create_task(session_cron.execute(a, ref, row, fire.policy))
    children = {}
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert (await correct(fire.connection, ref, dict(params, text='live'), verb='steer'))['status'] == 'queued'
        assert (await correct(fire.connection, ref, dict(params, text='redirect'), verb='redirect'))['status'] == 'redirected'
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            await correct(fire.connection, ref, dict(params, execution_generation=row['generation'] - 1), verb='steer')
        with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
            await correct(fire.connection, SessionRef('foreign-profile', ref.session_id), params, verb='steer')
        foreign = SimpleNamespace(authority=a, actor=SessionActor('other-person', a.profile_id,
                                  fire.connection.actor.capabilities, 'fixture'))
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await correct(foreign, ref, params, verb='steer')
        # Durable lineage may display a child whose parent was evicted; it does NOT authorize controls.
        other = Agent('other-session', a.db)
        evicted = Agent(ref.session_id, a.db)
        other_profile = Agent(ref.session_id, SimpleNamespace(db_path='other-profile/state.db'))
        for name, parent in [('owned', fire.agent), ('sibling', other), ('evicted', evicted), ('foreign', other_profile)]:
            child = children[name] = Agent(name, parent._session_db)
            child._delegate_parent_ref = weakref.ref(parent)
            with registry._active_subagents_lock:
                registry._active_subagents[name] = {'subagent_id': name, 'agent': child, 'accepting_steer': True,
                                                   'owner_agent_session_id': parent.session_id}
        roster = (await read(fire.connection, ref, {'session_id': ref.session_id}, kind='subagents'))['subagents']
        assert {r['subagent_id']: r['accepting_steer'] for r in roster} == {'owned': True, 'evicted': False}
        for name in children:
            p = {'session_id': ref.session_id, 'subagent_id': name}
            steered = await subagent_control(fire.connection, ref, dict(p, text='child text'), steer=True)
            stopped = await subagent_control(fire.connection, ref, p, steer=False)
            assert steered['status'] == ('queued' if name == 'owned' else 'rejected')
            assert stopped['found'] is (name == 'owned')
        registry._close_subagent_steering('owned', children['owned'])
        assert (await subagent_control(fire.connection, ref, {'session_id': ref.session_id,
                'subagent_id': 'owned', 'text': 'late'}, steer=True))['status'] == 'rejected'
        assert fire.agent.steered == ['before construction', 'live', 'redirect']
    finally:
        finish.set()
        if fails:
            with pytest.raises(RuntimeError, match='fixture conversation failed'):
                await asyncio.wait_for(task, 15)
        else:
            assert await asyncio.wait_for(task, 15) == 'done'
        with registry._active_subagents_lock:
            for name in children:
                registry._active_subagents.pop(name, None)
    assert a.adopted[ref.session_id] == (row['generation'], None)
    assert fire.finalizers == ['session', 'agent']
    assert session_cron.current_execution() is None


@pytest.mark.asyncio
async def test_cron_roster_durable_lineage_is_visible_but_not_steerable(fire):
    from gateway.session_ancillary import read, subagent_control
    from tools import delegate_tool_registry as registry

    child = Agent('evicted-child', fire.authority.db)
    name = 'evicted-child'
    with registry._active_subagents_lock:
        registry._active_subagents[name] = {'subagent_id': name, 'agent': child, 'accepting_steer': True,
                                           'owner_agent_session_id': fire.ref.session_id}
    try:
        params = {'session_id': fire.ref.session_id}
        roster = (await read(fire.connection, fire.ref, params, kind='subagents'))['subagents']
        assert [(r['subagent_id'], r['accepting_steer']) for r in roster] == [(name, False)]
        assert (await subagent_control(fire.connection, fire.ref, dict(params, subagent_id=name, text='no'),
                                       steer=True))['status'] == 'rejected'
        assert not child.steered
    finally:
        with registry._active_subagents_lock:
            registry._active_subagents.pop(name, None)


@pytest.mark.asyncio
async def test_scheduler_adoption_is_released_before_deferred_delivery_teardown(fire, monkeypatch):
    from cron import scheduler
    from gateway import session_cron

    deferred = []
    run_job = scheduler.run_job

    def with_deferred_delivery(job, **kwargs):
        return run_job(job, defer_agent_teardown=deferred, **kwargs)

    def conversation(agent, *args, **kwargs):
        assert fire.authority.agent(fire.ref) is agent is fire.agent
        return {}

    monkeypatch.setattr(scheduler, 'run_job', with_deferred_delivery)
    monkeypatch.setattr(scheduler, '_run_agent_with_watchdog', conversation)
    assert await session_cron.execute(fire.authority, fire.ref, fire.row, fire.policy) == 'done'
    assert deferred == [fire.agent]
    assert fire.finalizers == ['session']
    assert fire.authority.agent(fire.ref) is None


@pytest.mark.asyncio
async def test_late_cron_construction_and_release_cannot_replace_or_detach_successor(fire, monkeypatch):
    from cron import scheduler
    from gateway import session_cron

    a, sid, generation = fire.authority, fire.ref.session_id, fire.row['generation']
    replacement = Agent(sid, a.db)

    def late_constructor(*args, **kwargs):
        a.executing[sid] = generation + 1
        a.adopt_agent(sid, generation + 1, replacement)
        return fire.agent

    monkeypatch.setattr(scheduler, '_construct_cron_agent', late_constructor)
    def must_not_run(*args, **kwargs):
        raise AssertionError('stale construction must never reach inference')
    monkeypatch.setattr(scheduler, '_run_agent_with_watchdog', must_not_run)
    monkeypatch.setattr(scheduler, '_finalize_cron_session', lambda *a, **kw: None)
    monkeypatch.setattr(scheduler, '_teardown_cron_agent', lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match='stale_generation'):
        await session_cron.execute(a, fire.ref, fire.row, fire.policy)
    assert a.adopted[sid] == (generation + 1, replacement)

    # A late cleanup receipt is identity- AND generation-fenced, even for same-generation replacement.
    binding = (a, sid, generation + 1, a.sessions[sid])
    session_cron.release_execution_agent(binding, fire.agent)
    assert a.adopted[sid] == (generation + 1, replacement)
    session_cron.release_execution_agent((a, sid, generation, a.sessions[sid]), replacement)
    assert a.adopted[sid] == (generation + 1, replacement)
    session_cron.release_execution_agent(binding, replacement)
    assert a.adopted[sid] == (generation + 1, None)


@pytest.mark.asyncio
async def test_child_stop_does_not_hold_event_lock_while_waiting_for_finalization(fire):
    from gateway.session_ancillary import subagent_control
    from tools import delegate_tool_registry as registry

    authority, ref = fire.authority, fire.ref
    authority.adopt_agent(ref.session_id, fire.row['generation'], fire.agent)
    live = authority.sessions[ref.session_id]
    callback_done = threading.Event()
    threads, observations = [], []

    class FinalizingChild(Agent):
        def hard_interrupt(self, message=None):
            def finalizer():
                with live.event_stream.lock:
                    callback_done.set()
            thread = threading.Thread(target=finalizer, daemon=True)
            threads.append(thread)
            thread.start()
            observations.append(callback_done.wait(3))
            return True

    child = FinalizingChild('finishing-child', authority.db)
    child._delegate_parent_ref = weakref.ref(fire.agent)
    name = 'finishing-child'
    with registry._active_subagents_lock:
        registry._active_subagents[name] = {'subagent_id': name, 'agent': child,
            'accepting_steer': True, 'owner_agent_session_id': ref.session_id}
    try:
        result = await subagent_control(fire.connection, ref,
            {'session_id': ref.session_id, 'subagent_id': name}, steer=False)
        assert result['found'] is True
        assert observations == [True]
    finally:
        for thread in threads:
            thread.join(3)
        with registry._active_subagents_lock:
            registry._active_subagents.pop(name, None)
