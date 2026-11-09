import json
from math import comb

import pytest

from grpo_trainer.config import GateConfig
from grpo_trainer.errors import GateFailedError, ManifestMismatchError
from grpo_trainer.manifest import MANIFEST_NAME, atomic_write_json, ensure_manifest
from grpo_trainer.metrics import evaluate_gate, pass_at_k
from grpo_trainer.pins import PINNED
from grpo_trainer.preflight import Status, check_memory, check_versions, estimate_memory

# ------------------------------------------------------------------ pass@k


@pytest.mark.parametrize(("n", "c", "k"), [(8, 0, 1), (8, 3, 1), (8, 3, 4), (8, 8, 4), (10, 5, 3), (4, 1, 4)])
def test_pass_at_k_matches_closed_form(n, c, k):
    expected = 1.0 - comb(n - c, k) / comb(n, k)
    assert pass_at_k(n, c, k) == pytest.approx(expected)


@pytest.mark.parametrize(("n", "c", "k"), [(0, 0, 1), (4, 0, 5), (4, 5, 1), (4, -1, 1), (4, 1, 0)])
def test_pass_at_k_validates(n, c, k):
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


# ------------------------------------------------------------------ gate


def report(acc, fmt, **over):
    r = {
        "split": "test",
        "num_examples": 100,
        "dataset_fingerprint": "abc",
        "max_completion_length": 768,
        "system_prompt_sha": "s1",
        "greedy": {"accuracy": acc, "format_rate": fmt},
    }
    r.update(over)
    return r


def test_gate_passes_on_gain():
    res = evaluate_gate(report(0.60, 0.9), report(0.66, 0.95), GateConfig(min_accuracy_delta=0.03))
    assert res.passed and res.accuracy_delta == pytest.approx(0.06)


def test_gate_fails_on_small_gain_and_format_regression():
    res = evaluate_gate(report(0.60, 0.9), report(0.61, 0.8), GateConfig(min_accuracy_delta=0.03))
    assert not res.passed and len(res.reasons) == 2


@pytest.mark.parametrize(
    "key", ["split", "num_examples", "dataset_fingerprint", "max_completion_length", "system_prompt_sha"]
)
def test_gate_refuses_incomparable_reports(key):
    with pytest.raises(GateFailedError, match=key):
        evaluate_gate(report(0.5, 0.5), report(0.9, 0.9, **{key: "different"}), GateConfig())


# ------------------------------------------------------------------ versions


def test_versions_ok_with_local_build_tag():
    found = {**PINNED, "torch": PINNED["torch"] + "+cu130"}
    assert check_versions(lambda: found).status is Status.OK


def test_versions_fail_on_drift_and_missing():
    found = {**PINNED, "trl": "0.29.1", "vllm": None}
    r = check_versions(lambda: found)
    assert r.status is Status.FAIL and "trl 0.29.1" in r.detail and "vllm missing" in r.detail


# ------------------------------------------------------------------ memory estimate


def test_default_profile_fits_24gb(base_config):
    r = check_memory(base_config, num_params=1_543_714_304, vram_gb=24.0, vocab_size=151_936)
    assert r.status is Status.OK, r.detail


def test_original_template_7b_full_ft_on_24gb_fails(config_factory):
    cfg = config_factory("adapter.mode=none", "model.name_or_path=Qwen/Qwen2.5-7B-Instruct")
    r = check_memory(cfg, num_params=7_615_616_512, vram_gb=24.0, vocab_size=152_064)
    assert r.status is Status.FAIL, r.detail


def test_qlora_7b_on_24gb_is_not_a_hard_fail(config_factory):
    cfg = config_factory(
        "adapter.mode=qlora",
        "model.name_or_path=Qwen/Qwen2.5-7B-Instruct",
        "grpo.per_device_train_batch_size=2",
        "grpo.gradient_accumulation_steps=16",
        "vllm.gpu_memory_utilization=0.3",
    )
    r = check_memory(cfg, num_params=7_615_616_512, vram_gb=24.0, vocab_size=152_064)
    assert r.status in (Status.OK, Status.WARN), r.detail


def test_memory_skips_when_unknown(base_config):
    assert check_memory(base_config, None, 24.0, None).status is Status.SKIP


def test_zero3_shards_full_ft_state(config_factory):
    cfg = config_factory(profile="multi")
    one = estimate_memory(
        cfg.model_copy(update={"hardware": cfg.hardware.model_copy(update={"num_processes": 1})}),
        7_615_616_512,
        80.0,
    )
    three = estimate_memory(cfg, 7_615_616_512, 80.0)
    assert three.optimizer_gb == pytest.approx(one.optimizer_gb / 3)
    assert three.vllm_reserved_gb == 0  # server mode: vLLM lives on its own GPU


# ------------------------------------------------------------------ manifest


def test_manifest_create_resume_and_mismatch(tmp_path, config_factory):
    run_dir = tmp_path / "run"
    cfg = config_factory()
    m = ensure_manifest(run_dir, cfg, dataset_fingerprint="d1")
    assert m["config_fingerprint"] == cfg.fingerprint()
    assert (run_dir / MANIFEST_NAME).is_file()

    ensure_manifest(run_dir, config_factory("grpo.logging_steps=1"), dataset_fingerprint="d1")  # logging-only
    saved = json.loads((run_dir / MANIFEST_NAME).read_text())
    assert len(saved["resumes"]) == 1 and saved["resumes"][0]["forced"] is False

    with pytest.raises(ManifestMismatchError, match="config"):
        ensure_manifest(run_dir, config_factory("grpo.learning_rate=5.0e-5"), dataset_fingerprint="d1")
    with pytest.raises(ManifestMismatchError, match="dataset"):
        ensure_manifest(run_dir, cfg, dataset_fingerprint="d2")

    ensure_manifest(run_dir, cfg, dataset_fingerprint="d2", force=True)
    saved = json.loads((run_dir / MANIFEST_NAME).read_text())
    assert saved["resumes"][-1]["forced"] is True


def test_atomic_write_leaves_no_temp_files(tmp_path):
    atomic_write_json(tmp_path / "x.json", {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


# ------------------------------------------------------------------ trackers


def test_tracker_check_fails_when_backend_missing(base_config):
    from grpo_trainer.preflight import check_trackers

    assert base_config.grpo.report_to == ["tensorboard"]  # the shipped default
    assert check_trackers(base_config, find_spec=lambda m: None).status is Status.FAIL
    assert check_trackers(base_config, find_spec=lambda m: object()).status is Status.OK


def test_tracker_check_none_always_ok(config_factory):
    from grpo_trainer.preflight import check_trackers

    cfg = config_factory("grpo.report_to=[none]")
    assert check_trackers(cfg, find_spec=lambda m: None).status is Status.OK


def test_tensorboard_is_pinned():
    # The default config reports to tensorboard; it must be part of the pinned, locked stack.
    from pathlib import Path

    assert "tensorboard" in PINNED
    lock = (Path(__file__).resolve().parents[1] / "requirements.lock").read_text().splitlines()
    assert f"tensorboard=={PINNED['tensorboard']}" in lock
