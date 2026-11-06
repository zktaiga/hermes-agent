"""The scheduler polls an opted-in runtime budget independently of its inactivity watchdog."""
import concurrent.futures
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('completed', [False, True])
def test_runtime_budget_polls_without_inactivity_but_never_after_completion(monkeypatch, completed):
    from cron import scheduler
    from cron.scheduler_runtime_budget import CronRuntimeBudget

    future = concurrent.futures.Future()
    result = {'final_response': 'already complete'}
    if completed:
        future.set_result(result)
    shutdowns, polls, ticks = [], [], []

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, *args, **kwargs):
            return future

        def shutdown(self, **kwargs):
            shutdowns.append(kwargs)

    def wait(futures, *, timeout):
        assert futures == {future} and timeout > 0
        polls.append(True)
        return ({future}, set()) if completed else (set(), {future})

    agent = SimpleNamespace(run_conversation=lambda *a, **kw: None)

    def tick(actual, name):
        assert actual is agent and name == 'budget job'
        ticks.append(True)
        raise TimeoutError('fixture runtime limit')

    monkeypatch.setattr(scheduler, '_cron_inactivity_seconds', lambda: 0)
    monkeypatch.setattr(scheduler.concurrent.futures, 'ThreadPoolExecutor', Pool)
    monkeypatch.setattr(scheduler.concurrent.futures, 'wait', wait)
    monkeypatch.setattr(CronRuntimeBudget, 'from_job', lambda job: SimpleNamespace(enabled=True, tick=tick))
    args = (agent, 'work', {'max_runtime_seconds': 1}, 'job', 'budget job', 'task', None)
    if completed:
        assert scheduler._run_agent_with_watchdog(*args) is result
        assert ticks == []
    else:
        with pytest.raises(TimeoutError, match='fixture runtime limit'):
            scheduler._run_agent_with_watchdog(*args)
        assert ticks == [True]
    assert polls == [True]
    assert shutdowns and all(not call['wait'] for call in shutdowns)
