"""Training entrypoint. Launch with `accelerate launch -m grpo_trainer.cli train ...` (see scripts/train.sh).

Stage contract:
  in : validated Config, profile (if profiler.enabled)
  out: <run_dir>/manifest.json, checkpoints/, final/adapter|model, final/merged (LoRA), final/COMPLETE
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any

from .config import Config
from .data import fingerprint, load_splits, profiling_sample, training_examples
from .errors import ConfigError
from .manifest import atomic_write_json, ensure_manifest
from .preflight import enforce, format_results, run_preflight
from .profiler import profile_path_for
from .rewards import build_rewards
from .tasks import get_task

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ pure config mapping (unit-tested)


def grpo_config_kwargs(cfg: Config, output_dir: Path) -> dict[str, Any]:
    """Map our validated config onto TRL 1.17 GRPOConfig / transformers 5.19 TrainingArguments."""
    g, v = cfg.grpo, cfg.vllm
    model_init: dict[str, Any] = {"dtype": cfg.model.dtype}
    if cfg.model.attn_implementation:
        model_init["attn_implementation"] = cfg.model.attn_implementation
    if cfg.model.revision:
        model_init["revision"] = cfg.model.revision
    kw: dict[str, Any] = {
        "output_dir": str(output_dir),
        "seed": cfg.run.seed,
        "data_seed": cfg.run.seed,
        # optimisation
        "learning_rate": g.learning_rate,
        "lr_scheduler_type": g.lr_scheduler_type,
        "warmup_steps": g.warmup_steps,
        "weight_decay": g.weight_decay,
        "max_grad_norm": g.max_grad_norm,
        "per_device_train_batch_size": g.per_device_train_batch_size,
        "gradient_accumulation_steps": g.gradient_accumulation_steps,
        "max_steps": g.max_steps,
        "bf16": cfg.model.dtype == "bfloat16",
        "fp16": cfg.model.dtype == "float16",
        "gradient_checkpointing": g.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if g.gradient_checkpointing else None,
        # checkpointing / logging
        "save_strategy": "steps",
        "save_steps": g.save_steps,
        "save_total_limit": g.save_total_limit,
        "logging_steps": g.logging_steps,
        "report_to": g.report_to,
        "log_completions": g.log_completions,
        "num_completions_to_print": g.num_completions_to_print,
        # GRPO
        "num_generations": g.num_generations,
        "max_completion_length": g.max_completion_length,
        "temperature": g.temperature,
        "top_p": g.top_p,
        "beta": g.beta,
        "epsilon": g.epsilon,
        "num_iterations": g.num_iterations,
        "loss_type": g.loss_type,
        "scale_rewards": g.scale_rewards,
        "mask_truncated_completions": g.mask_truncated_completions,
        "reward_weights": [r.weight for r in cfg.rewards],
        "remove_unused_columns": False,  # `ground_truth` must reach the reward functions
        "model_init_kwargs": model_init,
        "trust_remote_code": cfg.model.trust_remote_code,
        # generation backend
        "use_vllm": v.enabled,
    }
    if g.epsilon_high is not None:
        kw["epsilon_high"] = g.epsilon_high
    if v.enabled:
        kw["vllm_mode"] = v.mode
        if v.mode == "colocate":
            kw["vllm_gpu_memory_utilization"] = v.gpu_memory_utilization
            kw["vllm_enable_sleep_mode"] = v.enable_sleep_mode
            kw["vllm_tensor_parallel_size"] = v.tensor_parallel_size
            if v.max_model_length:
                kw["vllm_max_model_length"] = v.max_model_length
        else:
            kw["vllm_server_host"] = v.server_host
            kw["vllm_server_port"] = v.server_port
            kw["vllm_server_timeout"] = v.server_timeout
    return {k: val for k, val in kw.items() if val is not None}


def build_peft(cfg: Config) -> tuple[Any, Any]:
    """(peft_config, quantization_config) for the adapter mode."""
    a = cfg.adapter
    if a.mode == "none":
        return None, None
    from peft import LoraConfig

    peft_cfg = LoraConfig(
        r=a.r,
        lora_alpha=a.alpha,
        lora_dropout=a.dropout,
        target_modules=a.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    quant = None
    if a.mode == "qlora":
        import torch
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, cfg.model.dtype),
        )
    return peft_cfg, quant


# ------------------------------------------------------------------ preemption handling


class _StopFlag:
    requested = False
    signum: int | None = None


def _install_signal_handlers() -> None:
    def handler(signum: int, _frame: Any) -> None:
        if _StopFlag.requested:  # second signal: hard exit
            raise KeyboardInterrupt
        _StopFlag.requested, _StopFlag.signum = True, signum
        log.warning("signal %d received: will checkpoint at end of current step and stop", signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


def _graceful_stop_callback() -> Any:
    from transformers import TrainerCallback

    class GracefulStop(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if _StopFlag.requested:
                control.should_save = True
                control.should_training_stop = True
            return control

    return GracefulStop()


# ------------------------------------------------------------------ main


def run_train(cfg: Config, *, resume: bool = True, force: bool = False, skip_preflight: bool = False) -> Path:
    import datasets
    from accelerate import PartialState
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    state = PartialState()
    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "checkpoints"
    final_dir = run_dir / "final"

    if not skip_preflight:
        results = run_preflight(cfg, require_gpu=True)
        if state.is_main_process:
            log.info("preflight:\n%s", format_results(results))
        enforce(results)

    splits = load_splits(cfg)
    profile_path = (
        profile_path_for(cfg, profiling_sample(cfg, splits.train)) if cfg.profiler.enabled else None
    )
    examples = training_examples(cfg, splits, profile_path)
    task = get_task(cfg.task.name, spec=cfg.task.spec)
    grading_fp = task.grading_fingerprint()
    # Grading rules that live in a spec file are part of the training data's meaning: changing a
    # matcher and resuming would mix two reward definitions in one run. Fold them into the fingerprint.
    data_fp = fingerprint(examples) if grading_fp is None else f"{fingerprint(examples)}+{grading_fp}"

    with state.main_process_first():
        if state.is_main_process:
            ensure_manifest(
                run_dir,
                cfg,
                dataset_fingerprint=data_fp,
                force=force,
                extra={
                    "profile": str(profile_path) if profile_path else None,
                    "train_examples": len(examples),
                    "contaminated_dropped": splits.contaminated_dropped,
                    "grading_fingerprint": grading_fp,
                    "task_spec": cfg.task.spec,
                },
            )

    system_prompt = task.system_prompt(cfg.task.system_prompt)
    train_ds = datasets.Dataset.from_list([e.to_row(system_prompt) for e in examples])

    reward_funcs, _ = build_rewards(cfg, task)
    try:
        args = GRPOConfig(**grpo_config_kwargs(cfg, ckpt_dir))
    except TypeError as e:  # unknown/removed field -> pin drift
        raise ConfigError(f"GRPOConfig rejected arguments (TRL version drift?): {e}") from e
    peft_cfg, quant = build_peft(cfg)

    _install_signal_handlers()
    trainer = GRPOTrainer(
        model=cfg.model.name_or_path,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=train_ds,
        peft_config=peft_cfg,
        quantization_config=quant,
        callbacks=[_graceful_stop_callback()],
    )

    target = final_dir / ("adapter" if peft_cfg is not None else "model")
    last = get_last_checkpoint(str(ckpt_dir)) if (resume and ckpt_dir.is_dir()) else None
    last_step = _checkpoint_step(last) if last else 0

    # Idempotency: re-running a finished command must not train extra steps. Observed with
    # transformers 5.19 + TRL 1.17.1: resuming a checkpoint already at max_steps ran one more step
    # (root cause not isolated), so this is guarded explicitly and covered by a test.
    if resume and (final_dir / "COMPLETE").is_file():
        log.info("run already complete (%s); nothing to do", final_dir / "COMPLETE")
        return final_dir
    if last and last_step >= cfg.grpo.max_steps:
        log.info(
            "checkpoint %s is at max_steps=%d; finalising without further training", last, cfg.grpo.max_steps
        )
        if state.is_main_process:
            _export_checkpoint(Path(last), target)
            _finalise(
                cfg,
                final_dir,
                target,
                peft=peft_cfg is not None,
                global_step=last_step,
                elapsed=0.0,
                metrics={"finalised_from_checkpoint": str(last)},
            )
        state.wait_for_everyone()
        return final_dir

    if last:
        log.info("resuming from %s (step %d/%d)", last, last_step, cfg.grpo.max_steps)
    t0 = time.monotonic()
    result = trainer.train(resume_from_checkpoint=last)
    elapsed = time.monotonic() - t0

    if _StopFlag.requested:
        log.warning(
            "stopped early by signal %s at step %d; checkpoint saved, re-run the same command to resume",
            _StopFlag.signum,
            trainer.state.global_step,
        )
        return ckpt_dir

    trainer.save_model(str(target))
    state.wait_for_everyone()
    if state.is_main_process:
        trainer.processing_class.save_pretrained(str(target))
        _finalise(
            cfg,
            final_dir,
            target,
            peft=peft_cfg is not None,
            global_step=trainer.state.global_step,
            elapsed=elapsed,
            metrics=result.metrics,
        )
    state.wait_for_everyone()
    return final_dir


_CHECKPOINT_STATE_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "scaler.pt",
}


def _checkpoint_step(path: str) -> int:
    import json

    state_file = Path(path) / "trainer_state.json"
    try:
        return int(json.loads(state_file.read_text())["global_step"])
    except (OSError, ValueError, KeyError) as e:
        log.warning("cannot read %s (%s); falling back to directory name", state_file, e)
        return int(Path(path).name.rsplit("-", 1)[-1])


def _export_checkpoint(checkpoint: Path, target: Path) -> None:
    """Copy model/adapter + tokenizer files from a checkpoint, leaving optimizer/RNG state behind."""
    import shutil

    target.mkdir(parents=True, exist_ok=True)
    for f in checkpoint.iterdir():
        if f.is_file() and f.name not in _CHECKPOINT_STATE_FILES and not f.name.startswith("rng_state"):
            shutil.copy2(f, target / f.name)


def _finalise(
    cfg: Config,
    final_dir: Path,
    target: Path,
    *,
    peft: bool,
    global_step: int,
    elapsed: float,
    metrics: dict[str, Any],
) -> None:
    merged = _merge_adapter(cfg, target, final_dir / "merged") if peft else target
    atomic_write_json(
        final_dir / "COMPLETE",
        {
            "global_step": global_step,
            "train_runtime_s": elapsed,
            "train_metrics": metrics,
            "artifact": str(target),
            "eval_model_path": str(merged),
        },
    )
    log.info("training complete: %s (eval with: grpo eval --model %s --label trained)", final_dir, merged)


def _merge_adapter(cfg: Config, adapter_dir: Path, out_dir: Path) -> Path:
    """Merge LoRA into a bf16 copy of the base (QLoRA too: merging into 4-bit weights is lossy)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("merging adapter into base -> %s", out_dir)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        dtype=getattr(torch, cfg.model.dtype if cfg.model.dtype != "float32" else "bfloat16"),
        device_map="cpu",
        trust_remote_code=cfg.model.trust_remote_code,
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(out_dir))
    del base, merged
    return out_dir
