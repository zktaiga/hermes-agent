"""Opt-in elapsed-runtime limits for agent jobs, separate from the idle watchdog.

The grace period is for checkpoint/cleanup, not another work batch. Like an explicit
Stop, expiry can interrupt tools: jobs must reserve rollback time before mutations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable

from agent.interrupt_compat import request_hard_interrupt


def normalize_runtime_seconds(value):
    """Owner setting: None/0 disables; booleans and non-integral values are errors."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("max_runtime_seconds must be a non-negative integer or null")
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("max_runtime_seconds must be finite")
    return value


def normalize_runtime_grace_seconds(value):
    """None inherits the 15-minute grace; explicit zero is an immediate stop."""
    normalized = normalize_runtime_seconds(value)
    return normalized


def _wind_down_tree(agent, text):
    """Notify only this job's current agent tree; never a process-global roster."""
    pending, seen = [agent], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        steer = getattr(current, "steer", None)
        if callable(steer):
            steer(text)
        lock = getattr(current, "_active_children_lock", None)
        if lock is not None:
            with lock:
                pending.extend(list(getattr(current, "_active_children", ())))


@dataclass
class CronRuntimeBudget:
    max_runtime_seconds: int | None
    runtime_grace_seconds: int = 900
    clock: Callable[[], float] = time.monotonic
    started_at: float = field(init=False)
    winding_down: bool = False
    expired: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self.started_at = self.clock()

    @classmethod
    def from_job(cls, job, *, clock=time.monotonic):
        limit = normalize_runtime_seconds(job.get("max_runtime_seconds"))
        grace = normalize_runtime_grace_seconds(job.get("runtime_grace_seconds"))
        return cls(limit, 900 if grace is None else grace, clock)

    @property
    def enabled(self):
        return bool(self.max_runtime_seconds)

    def tick(self, agent, job_name):
        """One cooperative wind-down, then an explicit Stop and a failed run on expiry."""
        limit = self.max_runtime_seconds
        if not limit:
            return
        with self._lock:
            elapsed = max(0.0, self.clock() - self.started_at)
            if elapsed < limit:
                return
            if not self.winding_down:
                self.winding_down = True
                _wind_down_tree(agent, (
                    "This cron job's configured work budget has elapsed. Start no new work, "
                    "delegations, deployments or retries. Finish only necessary safety/recovery, "
                    "checkpoint unfinished work as INCOMPLETE, clean up owned temporary resources, "
                    f"and return your final handoff within {self.runtime_grace_seconds} seconds. "
                    "Do not create a continuation job or claim unfinished work completed."
                ))
            if elapsed < limit + self.runtime_grace_seconds:
                return
            if not self.expired:
                self.expired = True
                request_hard_interrupt(agent, "Cron runtime budget exhausted", tool_reason="cron runtime budget")
            raise TimeoutError(
                f"Cron job '{job_name}' exceeded its {self.max_runtime_seconds}s work budget "
                f"and {self.runtime_grace_seconds}s checkpoint grace; run is incomplete")
