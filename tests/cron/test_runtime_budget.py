"""Elapsed cron budgets are opt-in, include descendants and never report expiry as success."""
import argparse
import threading

import pytest

from cron.scheduler_runtime_budget import CronRuntimeBudget


class Agent:
    def __init__(self):
        self.notes = []
        self.stops = []
        self._active_children = []
        self._active_children_lock = threading.Lock()

    def steer(self, text):
        self.notes.append(text)
        return True

    def hard_interrupt(self, message=None, *, tool_reason=None):
        self.stops.append((message, tool_reason))


def test_opt_in_budget_winds_down_only_owned_tree_then_stops_once():
    now = [100.0]
    budget = CronRuntimeBudget.from_job(
        {"max_runtime_seconds": 60, "runtime_grace_seconds": 15}, clock=lambda: now[0])
    parent, child, unrelated = Agent(), Agent(), Agent()
    parent._active_children.append(child)
    child._active_children.append(parent)  # malformed ancestry cannot loop the watchdog
    now[0] = 159
    budget.tick(parent, "maintenance")
    assert not parent.notes and not child.notes
    now[0] = 160
    budget.tick(parent, "maintenance")
    budget.tick(parent, "maintenance")
    assert len(parent.notes) == len(child.notes) == 1
    assert "INCOMPLETE" in parent.notes[0]
    assert not parent.stops and not unrelated.notes
    now[0] = 175
    for _ in range(2):
        with pytest.raises(TimeoutError, match="run is incomplete"):
            budget.tick(parent, "maintenance")
    assert parent.stops == [("Cron runtime budget exhausted", "cron runtime budget")]
    assert not unrelated.stops


@pytest.mark.parametrize("value", [None, 0])
def test_unconfigured_jobs_do_not_gain_a_runtime_limit(value):
    now = [0.0]
    budget = CronRuntimeBudget.from_job({"max_runtime_seconds": value}, clock=lambda: now[0])
    agent = Agent()
    now[0] = 10**9
    budget.tick(agent, "unlimited")
    assert not budget.enabled and not agent.notes and not agent.stops


def test_zero_grace_is_not_replaced_by_the_default():
    now = [0.0]
    budget = CronRuntimeBudget.from_job(
        {"max_runtime_seconds": 1, "runtime_grace_seconds": 0}, clock=lambda: now[0])
    now[0] = 1
    agent = Agent()
    with pytest.raises(TimeoutError):
        budget.tick(agent, "immediate")
    assert len(agent.stops) == 1


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "60", float("inf"), 10**1000])
@pytest.mark.parametrize("field", ["max_runtime_seconds", "runtime_grace_seconds"])
def test_invalid_budget_rejected_before_agent_execution(field, value):
    with pytest.raises(ValueError):
        CronRuntimeBudget.from_job({field: value})


def test_job_store_roundtrips_owner_budget_and_preserves_zero():
    from cron.jobs import create_job, get_job, update_job
    job = create_job("bounded maintenance", "every 1h", paused=True,
                     max_runtime_seconds=60, runtime_grace_seconds=0)
    assert get_job(job["id"])["max_runtime_seconds"] == 60
    assert get_job(job["id"])["runtime_grace_seconds"] == 0
    updated = update_job(job["id"], {"max_runtime_seconds": 0})
    assert updated["max_runtime_seconds"] == 0
    with pytest.raises(ValueError):
        update_job(job["id"], {"max_runtime_seconds": True})
    assert get_job(job["id"])["max_runtime_seconds"] == 0


def test_owner_cli_passes_budget_but_model_dispatch_cannot_raise_or_disable_it(monkeypatch):
    from hermes_cli.subcommands.cron import build_cron_parser
    from hermes_cli.cron import _job_api_kwargs
    from tools import cronjob_tools
    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=lambda args: None)
    args = parser.parse_args(["cron", "edit", "job", "--max-runtime-seconds", "60",
                              "--runtime-grace-seconds", "0"])
    kwargs = _job_api_kwargs(args)
    assert kwargs["max_runtime_seconds"] == 60 and kwargs["runtime_grace_seconds"] == 0
    called = []
    monkeypatch.setattr(cronjob_tools, "cronjob", lambda **kw: called.append(kw) or "{}")
    cronjob_tools._cronjob_handler({"action": "update", "job_id": "job",
                                   "max_runtime_seconds": 0, "runtime_grace_seconds": 99999})
    assert "max_runtime_seconds" not in called[0]
    assert "runtime_grace_seconds" not in called[0]


def test_cli_tool_job_format_preserves_an_explicit_disabled_budget_and_zero_grace():
    from tools.cronjob_job_args import _format_job
    result = _format_job({"id": "job", "max_runtime_seconds": 0, "runtime_grace_seconds": 0})
    assert result["max_runtime_seconds"] == 0
    assert result["runtime_grace_seconds"] == 0
