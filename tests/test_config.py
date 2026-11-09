import pytest

from grpo_trainer.config import load_config, parse_override
from grpo_trainer.errors import ConfigError

from .conftest import CONFIGS


@pytest.mark.parametrize("profile", ["gpu24", "gpu16", "gpu80", "multi", None])
def test_shipped_configs_validate(profile):
    cfg = load_config(CONFIGS / "base.yaml", profile)
    assert cfg.generation_batch_size % cfg.grpo.num_generations == 0


def test_profile_overrides_base():
    cfg = load_config(CONFIGS / "base.yaml", "gpu80")
    assert cfg.model.name_or_path == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.adapter.r == 64
    assert cfg.rewards[0].name == "correctness"  # untouched base sections survive the merge


def test_multi_profile_matches_accelerate_config():
    import yaml

    cfg = load_config(CONFIGS / "base.yaml", "multi")
    accel = yaml.safe_load((CONFIGS / "accelerate" / "multi_zero3.yaml").read_text())
    assert accel["num_processes"] == cfg.hardware.num_processes


def test_prompts_per_step(base_config):
    assert base_config.generation_batch_size == 32
    assert base_config.prompts_per_step == 4


def test_the_original_template_batch_is_rejected(config_factory):
    # per_device=1, grad_accum=4, num_generations=8 on one GPU: 4 % 8 != 0
    with pytest.raises(ConfigError, match="not divisible by num_generations"):
        config_factory("grpo.per_device_train_batch_size=1", "grpo.gradient_accumulation_steps=4")


def test_removed_trl_field_is_rejected(config_factory):
    with pytest.raises(ConfigError, match="max_prompt_length"):
        config_factory("grpo.max_prompt_length=512")


def test_unknown_top_level_key_rejected(config_factory):
    with pytest.raises(ConfigError, match="typo_section"):
        config_factory("typo_section.x=1")


def test_correctness_reward_required(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "run: {name: x}\nmodel: {name_or_path: m}\nrewards:\n  - {name: format, weight: 1.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="correctness"):
        load_config(p)


def test_shaping_weights_cannot_outvote_correctness(config_factory):
    with pytest.raises(ConfigError, match="shaping reward weights"):
        config_factory("rewards=[{name: correctness, weight: 1.0}, {name: format, weight: 1.0}]")


def test_duplicate_reward_names_rejected(config_factory):
    with pytest.raises(ConfigError, match="duplicate reward"):
        config_factory("rewards=[{name: correctness, weight: 1.0}, {name: correctness, weight: 0.1}]")


def test_profiler_band_must_be_ordered(config_factory):
    with pytest.raises(ConfigError, match="keep_min_pass_rate"):
        config_factory("profiler.keep_min_pass_rate=0.8", "profiler.keep_max_pass_rate=0.2")


def test_vllm_max_model_length_must_fit_completion(config_factory):
    with pytest.raises(ConfigError, match="max_model_length"):
        config_factory("vllm.max_model_length=512", "grpo.max_completion_length=768")


def test_bad_run_name(config_factory):
    with pytest.raises(ConfigError, match="run.name"):
        config_factory("run.name=has spaces")


def test_unknown_tracker(config_factory):
    with pytest.raises(ConfigError, match="report_to"):
        config_factory("grpo.report_to=[splunk]")


@pytest.mark.parametrize(
    ("expr", "path", "value"),
    [
        ("grpo.max_steps=20", ["grpo", "max_steps"], 20),
        ("grpo.learning_rate=1e-6", ["grpo", "learning_rate"], "1e-6"),  # YAML 1.1: needs a dot
        ("grpo.learning_rate=1.0e-6", ["grpo", "learning_rate"], 1.0e-6),
        ("vllm.enabled=false", ["vllm", "enabled"], False),
        ("model.revision=null", ["model", "revision"], None),
        ("grpo.report_to=[none]", ["grpo", "report_to"], ["none"]),
        ("run.name=a=b", ["run", "name"], "a=b"),
    ],
)
def test_parse_override(expr, path, value):
    assert parse_override(expr) == (path, value)


@pytest.mark.parametrize("expr", ["no_equals", "=5"])
def test_parse_override_errors(expr):
    with pytest.raises(ConfigError):
        parse_override(expr)


def test_override_through_scalar_rejected(config_factory):
    with pytest.raises(ConfigError, match="crosses non-mapping"):
        config_factory("run.name.sub=1")


def test_missing_file():
    with pytest.raises(ConfigError, match="not found"):
        load_config("does/not/exist.yaml")


def test_fingerprint_ignores_logging_but_not_semantics(config_factory):
    a = config_factory()
    assert (
        a.fingerprint()
        == config_factory("grpo.logging_steps=1", "grpo.save_steps=7", "eval.limit=10").fingerprint()
    )
    assert a.fingerprint() != config_factory("grpo.learning_rate=2.0e-5").fingerprint()


def test_scientific_notation_override_coerces_to_float(config_factory):
    assert config_factory("grpo.learning_rate=1e-6").grpo.learning_rate == pytest.approx(1e-6)


def test_fingerprint_changes_with_reward_weights(tmp_path):
    base = (CONFIGS / "base.yaml").read_text()
    p = tmp_path / "b.yaml"
    p.write_text(base.replace("weight: 0.2", "weight: 0.3"), encoding="utf-8")
    assert (
        load_config(p, CONFIGS / "profiles" / "gpu24.yaml").fingerprint()
        != load_config(CONFIGS / "base.yaml", "gpu24").fingerprint()
    )


def test_config_is_immutable(base_config):
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen model raises ValidationError
        base_config.run.seed = 1  # type: ignore[misc]


def test_gpu16_profile_fits_16gb_and_keeps_step_size():
    from grpo_trainer.preflight import Status, check_memory

    cfg = load_config(CONFIGS / "base.yaml", "gpu16")
    assert cfg.prompts_per_step == 4 and cfg.generation_batch_size == 32
    r = check_memory(cfg, num_params=1_543_714_304, vram_gb=16376 / 1024, vocab_size=151_936)
    assert r.status is Status.OK, r.detail
