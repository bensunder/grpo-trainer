"""Typed run configuration.

Layering: base YAML <- hardware profile YAML <- CLI `--set key.path=value` overrides.
Every section forbids unknown keys, so a typo or a field removed upstream (e.g. TRL's old
`max_prompt_length`) fails here, before any GPU is touched.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(_Strict):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    output_root: str = "runs"
    seed: int = 42


class ModelConfig(_Strict):
    name_or_path: str = Field(min_length=1)
    revision: str | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] | None = "sdpa"
    trust_remote_code: bool = False


class AdapterConfig(_Strict):
    """`none` = full fine-tune, `lora` = bf16 base + LoRA, `qlora` = 4-bit NF4 base + LoRA."""

    mode: Literal["none", "lora", "qlora"] = "lora"
    r: int = Field(default=32, ge=1, le=512)
    alpha: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    target_modules: str | list[str] = "all-linear"


class TaskConfig(_Strict):
    name: str = "gsm8k"
    train_split: str = "train"
    eval_split: str = "test"
    train_limit: int | None = Field(default=None, ge=1)
    eval_limit: int | None = Field(default=None, ge=1)
    system_prompt: str | None = None  # None -> task default (states the output contract)


class RewardSpec(_Strict):
    name: str
    weight: float = Field(gt=0.0)
    params: dict[str, Any] = Field(default_factory=dict)


class GRPOParams(_Strict):
    num_generations: int = Field(default=8, ge=2)
    max_completion_length: int = Field(default=768, ge=16)
    temperature: float = Field(default=1.0, gt=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    learning_rate: float = Field(default=1e-5, gt=0.0)
    lr_scheduler_type: str = "constant_with_warmup"
    # transformers 5.x removed `warmup_ratio`; a float in [0, 1) here is a ratio of total steps,
    # a value >= 1 is an exact step count.
    warmup_steps: float = Field(default=0.03, ge=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    per_device_train_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    max_steps: int = Field(default=300, ge=1)
    beta: float = Field(default=0.0, ge=0.0)
    epsilon: float = Field(default=0.2, gt=0.0)
    epsilon_high: float | None = Field(default=None, gt=0.0)
    num_iterations: int = Field(default=1, ge=1)
    loss_type: Literal["grpo", "dr_grpo", "dapo", "bnpo"] = "dapo"
    scale_rewards: Literal["group", "batch", "none"] = "group"
    mask_truncated_completions: bool = True
    gradient_checkpointing: bool = True
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=50, ge=1)
    save_total_limit: int = Field(default=3, ge=1)
    log_completions: bool = True
    num_completions_to_print: int = Field(default=2, ge=0)
    report_to: list[str] = Field(default_factory=lambda: ["none"])

    @field_validator("report_to")
    @classmethod
    def _known_trackers(cls, v: list[str]) -> list[str]:
        allowed = {"none", "tensorboard", "wandb", "mlflow", "trackio"}
        unknown = set(v) - allowed
        if unknown:
            raise ValueError(f"unknown report_to entries {sorted(unknown)}; allowed {sorted(allowed)}")
        return v


class VLLMConfig(_Strict):
    enabled: bool = True
    mode: Literal["colocate", "server"] = "colocate"
    gpu_memory_utilization: float = Field(default=0.35, gt=0.0, lt=1.0)
    enable_sleep_mode: bool = True
    max_model_length: int | None = Field(default=None, ge=128)
    tensor_parallel_size: int = Field(default=1, ge=1)
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=8000, ge=1, le=65535)
    server_timeout: float = Field(default=240.0, gt=0.0)


class ProfilerConfig(_Strict):
    """Offline difficulty profiling. Prompts the base model always/never solves give zero gradient."""

    enabled: bool = True
    samples_per_prompt: int = Field(default=8, ge=2)
    max_prompts: int | None = Field(default=2000, ge=1)
    keep_min_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)  # exclusive
    keep_max_pass_rate: float = Field(default=1.0, ge=0.0, le=1.0)  # exclusive
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _band(self) -> ProfilerConfig:
        if self.keep_min_pass_rate >= self.keep_max_pass_rate:
            raise ValueError("keep_min_pass_rate must be < keep_max_pass_rate")
        return self


class GateConfig(_Strict):
    # Strict accuracy: answer must be inside <answer> tags (format + knowledge).
    min_accuracy_delta: float = 0.03  # absolute, e.g. 0.03 = +3 points
    max_format_regression: float = 0.0  # candidate format rate may not drop more than this
    # Lenient accuracy: final number anywhere in the output (knowledge only, format ignored).
    # Default 0.0 = format gains may never hide a reasoning regression.
    min_lenient_accuracy_delta: float = 0.0
    # Paired exact (McNemar) test on per-question lenient correctness.
    require_significant_gain: bool = False
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)


class EvalConfig(_Strict):
    limit: int | None = Field(default=None, ge=1)
    pass_k: int = Field(default=4, ge=1)
    pass_k_temperature: float = Field(default=0.7, gt=0.0)
    max_completion_length: int | None = Field(default=None, ge=16)  # None -> grpo.max_completion_length
    gpu_memory_utilization: float = Field(default=0.85, gt=0.0, lt=1.0)
    gate: GateConfig = Field(default_factory=GateConfig)


class HardwareConfig(_Strict):
    profile: str = "unspecified"
    num_processes: int = Field(default=1, ge=1)  # training processes (excl. a dedicated vLLM server GPU)
    min_vram_gb: float = Field(default=20.0, gt=0.0)
    min_free_disk_gb: float = Field(default=40.0, ge=0.0)
    require_bf16: bool = True


class Config(_Strict):
    run: RunConfig
    model: ModelConfig
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    rewards: list[RewardSpec] = Field(min_length=1)
    grpo: GRPOParams = Field(default_factory=GRPOParams)
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @model_validator(mode="after")
    def _cross_field(self) -> Config:
        g = self.grpo
        gen_batch = self.generation_batch_size
        if gen_batch % g.num_generations != 0:
            raise ValueError(
                f"generation batch = per_device_train_batch_size({g.per_device_train_batch_size}) x "
                f"gradient_accumulation_steps({g.gradient_accumulation_steps}) x "
                f"num_processes({self.hardware.num_processes}) = {gen_batch}, which is not divisible by "
                f"num_generations({g.num_generations}). TRL needs whole prompt groups per generation batch."
            )
        names = [r.name for r in self.rewards]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate reward names: {names}")
        if "correctness" not in names:
            raise ValueError("a 'correctness' reward is required; format/length alone are trivially hackable")
        correctness_w = next(r.weight for r in self.rewards if r.name == "correctness")
        shaping_w = sum(r.weight for r in self.rewards if r.name != "correctness")
        if shaping_w >= correctness_w:
            raise ValueError(
                f"shaping reward weights sum ({shaping_w}) must be < correctness weight ({correctness_w}); "
                "otherwise a well-formatted wrong answer can outscore a correct one"
            )
        if self.adapter.mode == "qlora" and self.model.dtype == "float32":
            raise ValueError("qlora requires bfloat16 or float16 compute dtype")
        v = self.vllm
        if v.enabled and v.mode == "colocate" and v.tensor_parallel_size > self.hardware.num_processes:
            raise ValueError("colocate tensor_parallel_size cannot exceed num_processes")
        if self.vllm.max_model_length is not None and self.vllm.max_model_length <= g.max_completion_length:
            raise ValueError(
                "vllm.max_model_length must exceed grpo.max_completion_length (prompt needs room)"
            )
        return self

    @property
    def generation_batch_size(self) -> int:
        g = self.grpo
        return g.per_device_train_batch_size * g.gradient_accumulation_steps * self.hardware.num_processes

    @property
    def prompts_per_step(self) -> int:
        return self.generation_batch_size // self.grpo.num_generations

    @property
    def run_dir(self) -> Path:
        return Path(self.run.output_root) / self.run.name

    @property
    def eval_max_completion_length(self) -> int:
        return self.eval.max_completion_length or self.grpo.max_completion_length

    def fingerprint(self) -> str:
        """Stable hash of everything that changes training semantics."""
        payload = self.model_dump(mode="json")
        # Logging-only knobs don't change semantics; exclude so tweaking them doesn't block resume.
        for key in (
            "logging_steps",
            "log_completions",
            "num_completions_to_print",
            "report_to",
            "save_steps",
            "save_total_limit",
        ):
            payload["grpo"].pop(key, None)
        payload.pop("eval", None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------- loading


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {path} must be a mapping")
    return data


def parse_override(expr: str) -> tuple[list[str], Any]:
    """`grpo.max_steps=20` -> (["grpo", "max_steps"], 20). Values are parsed as YAML scalars/lists."""
    if "=" not in expr:
        raise ConfigError(f"override must be key.path=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    path = [p for p in key.strip().split(".") if p]
    if not path:
        raise ConfigError(f"empty key in override {expr!r}")
    try:
        value = yaml.safe_load(raw) if raw.strip() != "" else ""
    except yaml.YAMLError as e:
        raise ConfigError(f"cannot parse value in override {expr!r}: {e}") from e
    return path, value


def _apply_override(data: dict[str, Any], path: list[str], value: Any) -> None:
    node = data
    for part in path[:-1]:
        nxt = node.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise ConfigError(f"override path {'.'.join(path)} crosses non-mapping key {part!r}")
        node = nxt
    node[path[-1]] = value


def load_config(
    config_path: str | Path,
    profile: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    config_path = Path(config_path)
    data = _read_yaml(config_path)
    if profile:
        ppath = Path(profile)
        if not ppath.suffix:  # bare profile name -> configs/profiles/<name>.yaml next to base config
            ppath = config_path.parent / "profiles" / f"{profile}.yaml"
        data = _deep_merge(data, _read_yaml(ppath))
    for expr in overrides or []:
        path, value = parse_override(expr)
        _apply_override(data, path, value)
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "<root>"
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError("invalid configuration:\n" + "\n".join(lines)) from None
