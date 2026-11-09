from pathlib import Path

import pytest

from grpo_trainer.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture
def base_config() -> Config:
    return load_config(CONFIGS / "base.yaml", "gpu24")


@pytest.fixture
def config_factory():
    def make(*overrides: str, profile: str | None = "gpu24") -> Config:
        return load_config(CONFIGS / "base.yaml", profile, list(overrides))

    return make
