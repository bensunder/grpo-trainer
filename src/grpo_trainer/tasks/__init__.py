"""Task registry. Adding a task = one module implementing `Task` + one entry here."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..errors import ConfigError
from .base import (
    DEFAULT_SYSTEM_PROMPT,
    Example,
    Prediction,
    Task,
    assert_disjoint,
    content_hash,
    dataset_fingerprint,
    drop_contaminated,
)
from .gsm8k import GSM8KTask

if TYPE_CHECKING:
    from ..config import Config

# Tasks fully defined by code. Spec-driven tasks (below) need a file and are built by get_task.
TASKS: dict[str, Callable[[], Task]] = {
    "gsm8k": GSM8KTask,
}
SPEC_TASKS = ("extraction",)


def get_task(name: str, *, spec: str | None = None) -> Task:
    if name in SPEC_TASKS:
        if not spec:
            raise ConfigError(f"task {name!r} requires task.spec (path to its spec YAML)")
        from .extraction import ExtractionTask

        return ExtractionTask(spec)
    if spec:
        raise ConfigError(f"task.spec is only valid for {list(SPEC_TASKS)}, not {name!r}")
    factory = TASKS.get(name)
    if factory is None:
        raise ConfigError(f"unknown task {name!r}; registered: {sorted([*TASKS, *SPEC_TASKS])}")
    return factory()


def task_for(cfg: Config) -> Task:
    return get_task(cfg.task.name, spec=cfg.task.spec)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "SPEC_TASKS",
    "TASKS",
    "Example",
    "GSM8KTask",
    "Prediction",
    "Task",
    "assert_disjoint",
    "content_hash",
    "dataset_fingerprint",
    "drop_contaminated",
    "get_task",
    "task_for",
]
