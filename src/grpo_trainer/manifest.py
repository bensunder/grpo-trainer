"""Run manifest: what exactly produced this run directory. Guards resume against silent config drift."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import tempfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config
from .errors import ManifestMismatchError
from .pins import PINNED

MANIFEST_NAME = "manifest.json"


def installed_versions(packages: list[str] | None = None) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in packages or list(PINNED):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_revision(cwd: Path | None = None) -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.SubprocessError):
        return None


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write via temp file + rename so a crash never leaves a half-written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def build_manifest(
    cfg: Config, *, dataset_fingerprint: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "harness_version": __version__,
        "config_fingerprint": cfg.fingerprint(),
        "dataset_fingerprint": dataset_fingerprint,
        "git_revision": git_revision(Path(__file__).resolve().parent),
        "packages": installed_versions(),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "config": cfg.model_dump(mode="json"),
        **(extra or {}),
    }


def ensure_manifest(
    run_dir: Path,
    cfg: Config,
    *,
    dataset_fingerprint: str,
    force: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the manifest for a new run, or verify an existing one before resuming.

    Resuming with a different config fingerprint or training data is refused unless `force`.
    """
    path = run_dir / MANIFEST_NAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = []
        if existing.get("config_fingerprint") != cfg.fingerprint():
            mismatches.append(f"config {existing.get('config_fingerprint')} -> {cfg.fingerprint()}")
        if existing.get("dataset_fingerprint") != dataset_fingerprint:
            mismatches.append(f"dataset {existing.get('dataset_fingerprint')} -> {dataset_fingerprint}")
        if mismatches and not force:
            raise ManifestMismatchError(
                f"run dir {run_dir} was created with different inputs ({'; '.join(mismatches)}). "
                "Use a new run.name, or pass --force to resume anyway."
            )
        history = existing.setdefault("resumes", [])
        history.append(
            {"at": datetime.now(UTC).isoformat(), "forced": bool(mismatches), "mismatches": mismatches}
        )
        atomic_write_json(path, existing)
        return existing
    manifest = build_manifest(cfg, dataset_fingerprint=dataset_fingerprint, extra=extra)
    atomic_write_json(path, manifest)
    return manifest
