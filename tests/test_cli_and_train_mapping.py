import json

import pytest

from grpo_trainer.cli import main
from grpo_trainer.train import grpo_config_kwargs

from .conftest import CONFIGS

BASE = ["--config", str(CONFIGS / "base.yaml"), "--profile", "gpu24"]


def test_cli_validate_ok(capsys):
    assert main(["validate", *BASE, "--print"]) == 0
    assert json.loads(capsys.readouterr().out)["run"]["name"] == "gsm8k-qwen2.5-1.5b-grpo"


def test_cli_validate_bad_config_exit_code():
    assert (
        main(
            [
                "validate",
                *BASE,
                "--set",
                "grpo.per_device_train_batch_size=1",
                "--set",
                "grpo.gradient_accumulation_steps=4",
            ]
        )
        == 2
    )


def _write_reports(run_dir, base_acc, trained_acc):
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    common = {
        "split": "test",
        "num_examples": 10,
        "dataset_fingerprint": "f",
        "max_completion_length": 768,
        "system_prompt_sha": "s",
    }
    for label, acc in (("base", base_acc), ("trained", trained_acc)):
        (eval_dir / f"{label}.json").write_text(
            json.dumps({**common, "greedy": {"accuracy": acc, "format_rate": 1.0}})
        )


def test_cli_gate_pass_and_fail(tmp_path):
    args = [*BASE, "--set", f"run.output_root={tmp_path}", "--set", "run.name=r"]
    _write_reports(tmp_path / "r", 0.5, 0.6)
    assert main(["gate", *args]) == 0
    assert json.loads((tmp_path / "r" / "eval" / "gate.json").read_text())["passed"] is True

    args2 = [*BASE, "--set", f"run.output_root={tmp_path}", "--set", "run.name=r2"]
    _write_reports(tmp_path / "r2", 0.5, 0.51)
    assert main(["gate", *args2]) == 6


def test_cli_gate_missing_reports(tmp_path):
    assert main(["gate", *BASE, "--set", f"run.output_root={tmp_path}"]) == 6


# ------------------------------------------------------------------ TRL mapping


def test_mapping_core_values(base_config, tmp_path):
    kw = grpo_config_kwargs(base_config, tmp_path)
    assert kw["num_generations"] == 8
    assert kw["reward_weights"] == [1.0, 0.2, 0.1]
    assert kw["model_init_kwargs"]["dtype"] == "bfloat16"  # TRL defaults to float32 otherwise
    assert kw["remove_unused_columns"] is False
    assert kw["vllm_mode"] == "colocate" and kw["use_vllm"] is True
    assert "max_prompt_length" not in kw and "warmup_ratio" not in kw
    assert kw["bf16"] is True and kw["fp16"] is False


def test_mapping_server_mode(config_factory, tmp_path):
    kw = grpo_config_kwargs(config_factory(profile="multi"), tmp_path)
    assert kw["vllm_mode"] == "server"
    assert "vllm_gpu_memory_utilization" not in kw
    assert kw["vllm_server_port"] == 8000


def test_mapping_vllm_disabled(config_factory, tmp_path):
    kw = grpo_config_kwargs(config_factory("vllm.enabled=false"), tmp_path)
    assert kw["use_vllm"] is False and "vllm_mode" not in kw


def _trl_field_names():
    trl = pytest.importorskip("trl", reason="full training stack not installed")
    import dataclasses

    return {f.name for f in dataclasses.fields(trl.GRPOConfig)}


@pytest.mark.parametrize("profile", ["gpu24", "gpu80", "multi"])
def test_mapping_keys_exist_in_installed_trl(config_factory, tmp_path, profile):
    fields = _trl_field_names()
    unknown = set(grpo_config_kwargs(config_factory(profile=profile), tmp_path)) - fields
    assert not unknown, f"GRPOConfig has no fields {sorted(unknown)}"


@pytest.mark.parametrize("profile", ["gpu24", "gpu80"])
def test_grpo_config_constructs(config_factory, tmp_path, profile):
    trl = pytest.importorskip("trl", reason="full training stack not installed")
    kw = grpo_config_kwargs(config_factory(profile=profile), tmp_path)
    # bf16 needs a GPU at construction time in transformers; validate the rest on CPU.
    kw.update(bf16=False, use_vllm=False, use_cpu=True)
    kw.pop("vllm_mode", None)
    args = trl.GRPOConfig(**kw)
    assert args.generation_batch_size % args.num_generations == 0
    assert args.loss_type == "dapo"


def test_flashinfer_sampler_disabled_by_default(monkeypatch):
    import os

    from grpo_trainer.cli import apply_runtime_env_defaults

    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    apply_runtime_env_defaults()
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_flashinfer_sampler_explicit_override_wins(monkeypatch):
    import os

    from grpo_trainer.cli import apply_runtime_env_defaults

    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    apply_runtime_env_defaults()
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_main_applies_runtime_defaults(monkeypatch):
    import os

    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    assert main(["validate", *BASE]) == 0
    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
