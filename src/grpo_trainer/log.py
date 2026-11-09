"""Process-wide logging. Rank-aware so multi-GPU runs don't print N copies of every line."""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(level: str | int = "INFO") -> None:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0)
    root = logging.getLogger()
    if getattr(root, "_grpo_configured", False):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s %(levelname)s [rank{rank}] %(name)s: %(message)s", "%Y-%m-%dT%H:%M:%S"
        )
    )
    root.handlers[:] = [handler]
    # Non-zero ranks only surface warnings and above.
    root.setLevel(level if rank == 0 else logging.WARNING)
    for noisy in ("httpx", "urllib3", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._grpo_configured = True  # type: ignore[attr-defined]
