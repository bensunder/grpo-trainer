"""Fail-fast environment checks. Runs before any weights load.

Each check returns a CheckResult; the run is blocked if any check FAILs. Memory numbers are
estimates, deliberately conservative, and labelled as such -- they catch "7B full fine-tune on 24 GB",
not 10% tuning questions.
"""

from __future__ import annotations

import json
import logging
import shutil
import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .config import Config
from .errors import PreflightError
from .manifest import installed_versions
from .pins import PINNED, REQUIRED_CUDA_MAJOR

log = logging.getLogger(__name__)

GiB = 1024**3
DEFAULT_VOCAB = 152_064  # Qwen2.5 family; used only when the model config can't be read
CUDA_CONTEXT_GB = 1.5
TYPICAL_PROMPT_TOKENS = 384


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str


# ------------------------------------------------------------------ versions


def check_versions(get_versions: Callable[[], dict[str, str | None]] = installed_versions) -> CheckResult:
    found = get_versions()
    problems = []
    for pkg, want in PINNED.items():
        have = found.get(pkg)
        if have is None:
            problems.append(f"{pkg} missing (want {want})")
        elif have != want and not have.startswith(want + "+"):  # allow local tags like 2.13.0+cu130
            problems.append(f"{pkg} {have} != {want}")
    if problems:
        return CheckResult(
            "versions", Status.FAIL, "; ".join(problems) + " -- install from requirements.lock"
        )
    return CheckResult("versions", Status.OK, "all pinned packages match")


# ------------------------------------------------------------------ memory estimate


@dataclass(frozen=True)
class MemoryEstimate:
    weights_gb: float
    optimizer_gb: float
    activations_gb: float
    vllm_reserved_gb: float
    overhead_gb: float

    @property
    def total_gb(self) -> float:
        return (
            self.weights_gb
            + self.optimizer_gb
            + self.activations_gb
            + self.vllm_reserved_gb
            + self.overhead_gb
        )


def estimate_memory(
    cfg: Config, num_params: int, vram_gb: float, vocab_size: int = DEFAULT_VOCAB
) -> MemoryEstimate:
    """Per-training-GPU estimate in GiB.

    weights:     bf16/fp16 2 B/param, fp32 4 B/param, NF4 ~0.55 B/param (incl. quant constants)
    trainable:   full FT = all params; LoRA ~ r/16 * 1% of params (all-linear targets)
    optimizer:   full FT 10 B/param (grad 2 + AdamW fp32 m,v 8); LoRA params kept fp32 -> 16 B/param
    activations: dominated by fp32 logits over the vocab for the micro-batch, x2 for backward
    vLLM:        colocate reserves gpu_memory_utilization x VRAM on the same device
    """
    a = cfg.adapter
    bytes_per_weight = {"bfloat16": 2.0, "float16": 2.0, "float32": 4.0}[cfg.model.dtype]
    if a.mode == "qlora":
        bytes_per_weight = 0.55
    weights = num_params * bytes_per_weight
    if a.mode == "none":
        trainable, opt_bytes = num_params, 10.0
    else:
        trainable, opt_bytes = num_params * 0.01 * (a.r / 16), 16.0
    shard = cfg.hardware.num_processes if a.mode == "none" and cfg.hardware.num_processes > 1 else 1  # ZeRO-3
    seq = TYPICAL_PROMPT_TOKENS + cfg.grpo.max_completion_length
    acts = cfg.grpo.per_device_train_batch_size * seq * vocab_size * 4 * 2
    vllm = (
        cfg.vllm.gpu_memory_utilization * vram_gb * GiB
        if (cfg.vllm.enabled and cfg.vllm.mode == "colocate")
        else 0
    )
    return MemoryEstimate(
        weights_gb=weights / shard / GiB,
        optimizer_gb=trainable * opt_bytes / shard / GiB,
        activations_gb=acts / GiB,
        vllm_reserved_gb=vllm / GiB,
        overhead_gb=CUDA_CONTEXT_GB,
    )


def check_memory(
    cfg: Config, num_params: int | None, vram_gb: float | None, vocab_size: int | None
) -> CheckResult:
    if num_params is None or vram_gb is None:
        return CheckResult("memory", Status.SKIP, "parameter count or VRAM unknown; estimate skipped")
    est = estimate_memory(cfg, num_params, vram_gb, vocab_size or DEFAULT_VOCAB)
    detail = (
        f"~{est.total_gb:.1f} GiB of {vram_gb:.1f} GiB "
        f"(weights {est.weights_gb:.1f}, optim {est.optimizer_gb:.1f}, "
        f"acts {est.activations_gb:.1f}, vLLM {est.vllm_reserved_gb:.1f}, ctx {est.overhead_gb:.1f}); "
        f"{num_params / 1e9:.2f}B params"
    )
    ratio = est.total_gb / vram_gb
    if ratio > 1.25:
        return CheckResult(
            "memory", Status.FAIL, detail + " -- will not fit; use qlora, smaller model, or bigger GPU"
        )
    if ratio > 0.95:
        return CheckResult(
            "memory", Status.WARN, detail + " -- tight; lower vllm.gpu_memory_utilization or batch"
        )
    return CheckResult("memory", Status.OK, detail)


# ------------------------------------------------------------------ model metadata


def _local_safetensors_params(path: Path) -> int | None:
    files = sorted(path.glob("*.safetensors"))
    if not files:
        return None
    total = 0
    for f in files:
        with f.open("rb") as fh:
            (header_len,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(header_len))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            n = 1
            for dim in meta.get("shape", []):
                n *= int(dim)
            total += n
    return total


def model_metadata(cfg: Config) -> tuple[int | None, int | None, str]:
    """(num_params, vocab_size, source). Never raises; returns Nones when unavailable."""
    ref = cfg.model.name_or_path
    num_params: int | None = None
    vocab: int | None = None
    source = "unavailable"
    local = Path(ref)
    try:
        if local.is_dir():
            num_params = _local_safetensors_params(local)
            cfg_file = local / "config.json"
            if cfg_file.is_file():
                vocab = json.loads(cfg_file.read_text()).get("vocab_size")
            source = "local"
        else:
            from huggingface_hub import get_safetensors_metadata, hf_hub_download

            meta = get_safetensors_metadata(ref, revision=cfg.model.revision)
            num_params = int(sum(meta.parameter_count.values()))
            cfg_path = hf_hub_download(ref, "config.json", revision=cfg.model.revision)
            vocab = json.loads(Path(cfg_path).read_text()).get("vocab_size")
            source = "hub"
    except Exception as e:  # noqa: BLE001 - metadata is advisory
        log.warning("could not read model metadata for %s: %s", ref, e)
    return num_params, vocab, source


# ------------------------------------------------------------------ GPU / disk


def check_gpu(cfg: Config) -> tuple[list[CheckResult], float | None]:
    try:
        import torch
    except ImportError:
        return [CheckResult("gpu", Status.FAIL, "torch not installed")], None
    if not torch.cuda.is_available():
        return [
            CheckResult("gpu", Status.FAIL, "CUDA not available (driver missing or not exposed to container)")
        ], None
    results: list[CheckResult] = []
    cuda = torch.version.cuda or "?"
    major = int(cuda.split(".")[0]) if cuda[0].isdigit() else -1
    if major != REQUIRED_CUDA_MAJOR:
        results.append(
            CheckResult("cuda", Status.FAIL, f"torch built for CUDA {cuda}, expected {REQUIRED_CUDA_MAJOR}.x")
        )
    else:
        results.append(CheckResult("cuda", Status.OK, f"torch CUDA {cuda}"))
    count = torch.cuda.device_count()
    # In server mode the vLLM GPU is excluded via CUDA_VISIBLE_DEVICES, so trainers only see their own.
    need = cfg.hardware.num_processes
    if count < need:
        results.append(CheckResult("gpu_count", Status.FAIL, f"{count} visible GPUs, config needs {need}"))
    else:
        results.append(CheckResult("gpu_count", Status.OK, f"{count} visible GPUs (need {need})"))
    props = [torch.cuda.get_device_properties(i) for i in range(count)]
    vram_gb = min(p.total_memory for p in props) / GiB
    names = sorted({p.name for p in props})
    if vram_gb < cfg.hardware.min_vram_gb:
        results.append(
            CheckResult(
                "vram", Status.FAIL, f"{vram_gb:.1f} GiB < profile minimum {cfg.hardware.min_vram_gb}"
            )
        )
    else:
        results.append(CheckResult("vram", Status.OK, f"{vram_gb:.1f} GiB per GPU ({', '.join(names)})"))
    if cfg.hardware.require_bf16 and cfg.model.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        results.append(CheckResult("bf16", Status.FAIL, "GPU lacks bf16 (needs compute capability >= 8.0)"))
    return results, vram_gb


def check_disk(cfg: Config) -> CheckResult:
    root = Path(cfg.run.output_root).resolve()
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / GiB
    if free_gb < cfg.hardware.min_free_disk_gb:
        return CheckResult(
            "disk", Status.FAIL, f"{free_gb:.0f} GiB free at {probe} < {cfg.hardware.min_free_disk_gb}"
        )
    return CheckResult("disk", Status.OK, f"{free_gb:.0f} GiB free at {probe}")


_TRACKER_MODULES = {
    "tensorboard": ("tensorboard", "tensorboardX"),
    "wandb": ("wandb",),
    "mlflow": ("mlflow",),
    "trackio": ("trackio",),
}


def check_trackers(cfg: Config, find_spec=None) -> CheckResult:
    """transformers raises at Trainer init if a report_to backend isn't importable; catch it here."""
    import importlib.util

    find_spec = find_spec or importlib.util.find_spec
    missing = []
    for tracker in cfg.grpo.report_to:
        modules = _TRACKER_MODULES.get(tracker)
        if modules and not any(find_spec(m) is not None for m in modules):
            missing.append(tracker)
    if missing:
        return CheckResult(
            "trackers",
            Status.FAIL,
            f"report_to {missing} not installed; install it or set grpo.report_to=[none]",
        )
    return CheckResult("trackers", Status.OK, f"report_to {cfg.grpo.report_to}")


# ------------------------------------------------------------------ orchestration


def run_preflight(cfg: Config, *, require_gpu: bool = True) -> list[CheckResult]:
    results = [
        CheckResult(
            "config",
            Status.OK,
            f"valid; fingerprint {cfg.fingerprint()}; {cfg.prompts_per_step} prompts x "
            f"{cfg.grpo.num_generations} generations per step",
        )
    ]
    results.append(check_versions())
    results.append(check_trackers(cfg))
    results.append(check_disk(cfg))
    if cfg.hardware.num_processes > 1 and cfg.adapter.mode == "none":
        has_ds = installed_versions(["deepspeed"]).get("deepspeed")
        results.append(
            CheckResult(
                "deepspeed",
                Status.OK if has_ds else Status.FAIL,
                f"deepspeed {has_ds}"
                if has_ds
                else "multi-GPU full fine-tune needs deepspeed: pip install '.[multi]'",
            )
        )
    vram_gb: float | None = None
    if require_gpu:
        gpu_results, vram_gb = check_gpu(cfg)
        results.extend(gpu_results)
    num_params, vocab, _ = model_metadata(cfg)
    results.append(check_memory(cfg, num_params, vram_gb, vocab))
    return results


def format_results(results: list[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    return "\n".join(f"  [{r.status.value.upper():4}] {r.name:<{width}}  {r.detail}" for r in results)


def enforce(results: list[CheckResult]) -> None:
    failed = [r for r in results if r.status is Status.FAIL]
    if failed:
        raise PreflightError("preflight failed:\n" + format_results(failed))
