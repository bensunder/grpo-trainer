"""Typed failures. Every user-facing failure maps to one of these and a stable exit code."""

from __future__ import annotations


class GRPOTrainerError(Exception):
    """Base class. `exit_code` is what the CLI returns."""

    exit_code: int = 1


class ConfigError(GRPOTrainerError):
    """Config file missing, malformed, or semantically invalid."""

    exit_code = 2


class PreflightError(GRPOTrainerError):
    """Environment cannot run this config (versions, GPU, memory, disk)."""

    exit_code = 3


class DataError(GRPOTrainerError):
    """Dataset could not be loaded, is empty after filtering, or train/test overlap."""

    exit_code = 4


class ManifestMismatchError(GRPOTrainerError):
    """Attempt to resume a run directory with a different config or code version."""

    exit_code = 5


class GateFailedError(GRPOTrainerError):
    """Trained model did not clear the eval gate."""

    exit_code = 6
