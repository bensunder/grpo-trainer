"""Task registry. Adding a task = one module implementing `Task` + one entry here."""

from __future__ import annotations

from collections.abc import Callable

from ..errors import ConfigError
from .base import (
    DEFAULT_SYSTEM_PROMPT,
    Example,
    Task,
    assert_disjoint,
    content_hash,
    dataset_fingerprint,
    drop_contaminated,
)
from .gsm8k import GSM8KTask

TASKS: dict[str, Callable[[], Task]] = {
    "gsm8k": GSM8KTask,
}


def get_task(name: str) -> Task:
    factory = TASKS.get(name)
    if factory is None:
        raise ConfigError(f"unknown task {name!r}; registered: {sorted(TASKS)}")
    return factory()


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "TASKS",
    "Example",
    "GSM8KTask",
    "Task",
    "assert_disjoint",
    "content_hash",
    "dataset_fingerprint",
    "drop_contaminated",
    "get_task",
]
