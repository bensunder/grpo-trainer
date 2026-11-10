"""`grpo` command line.

  grpo preflight --config configs/base.yaml --profile gpu24
  grpo profile   --config ... --profile gpu24
  scripts/train.sh gpu24      # = accelerate launch ... -m grpo_trainer.cli train --config ... --profile gpu24
  grpo eval      --config ... --profile gpu24 --model Qwen/Qwen2.5-1.5B-Instruct --label base
  grpo eval      --config ... --profile gpu24 --model runs/<name>/final/merged --label trained
  grpo gate      --config ... --profile gpu24
  grpo baseline  --config ... --profile gpu24 --provider NAME [--dry-run]   # PAID API calls
  grpo report    --config ... --profile gpu24
  grpo bundle    --config ... --profile gpu24

No config needed:
  grpo spec init  --data train.jsonl --out specs/x.yaml --name x [--eval eval.jsonl]
  grpo spec check --spec specs/x.yaml [--json]
  grpo synth invoices --out data/invoices [--train 3000 --eval 300 --seed 7]

Exit codes: 0 ok, 2 config, 3 preflight, 4 data, 5 manifest mismatch, 6 gate failed,
            7 budget, 8 provider, 1 other.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import Config, load_config
from .errors import GateFailedError, GRPOTrainerError
from .log import setup_logging

log = logging.getLogger("grpo_trainer")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="base YAML config")
    p.add_argument("--profile", help="hardware profile name (configs/profiles/<name>.yaml) or path")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set grpo.max_steps=20 (repeatable)",
    )
    p.add_argument("--log-level", default="INFO")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grpo", description="GRPO training harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate config only (no GPU, no network)")
    _common(p)
    p.add_argument("--print", action="store_true", help="print the resolved config as JSON")

    p = sub.add_parser("preflight", help="check environment against config")
    _common(p)
    p.add_argument("--no-gpu", action="store_true", help="skip GPU checks (CPU-only sanity run)")

    p = sub.add_parser("profile", help="measure base-model pass rates; select learnable prompts")
    _common(p)
    p.add_argument("--force", action="store_true", help="recompute even if a cached profile exists")

    p = sub.add_parser("train", help="run GRPO (launch via accelerate)")
    _common(p)
    p.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints")
    p.add_argument("--force", action="store_true", help="resume even if config/data fingerprint changed")
    p.add_argument("--skip-preflight", action="store_true")

    p = sub.add_parser("eval", help="evaluate a model on the held-out split")
    _common(p)
    p.add_argument("--model", required=True, help="HF id or local path (use final/merged for trained LoRA)")
    p.add_argument("--label", required=True, choices=["base", "trained"], help="report name")

    p = sub.add_parser("gate", help="compare base vs trained eval reports against the gate")
    _common(p)

    p = sub.add_parser("baseline", help="run a frontier API model on the held-out split (PAID)")
    _common(p)
    p.add_argument("--provider", required=True, help="baseline.providers[].name")
    p.add_argument("--dry-run", action="store_true", help="print the cost estimate; send nothing")

    p = sub.add_parser("report", help="write the proof report (HTML + JSON) from saved eval outputs")
    _common(p)

    p = sub.add_parser("bundle", help="package the gated model with a serving gateway")
    _common(p)
    p.add_argument("--out", type=Path, help="output dir (default <run>/bundle)")
    p.add_argument(
        "--allow-failed-gate", action="store_true", help="bundle even if the gate failed or never ran"
    )
    p.add_argument("--smoke-examples", type=int, default=5, help="held-out examples for smoke_test.py")
    p.add_argument("--vllm-image", default=None, help="vLLM serving image (default matches the pinned vllm)")
    p.add_argument("--force", action="store_true", help="replace an existing bundle")

    spec = sub.add_parser("spec", help="extraction spec tools (no config needed)")
    spec_sub = spec.add_subparsers(dest="spec_command", required=True)
    p = spec_sub.add_parser("init", help="draft a spec from labelled JSONL")
    p.add_argument("--data", required=True, type=Path, help="training JSONL")
    p.add_argument("--eval", dest="eval_data", type=Path, help="held-out JSONL")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--input-field", default="text")
    p.add_argument("--label-field", default="label")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO")
    p = spec_sub.add_parser("check", help="validate labels and self-test the compiled reward")
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--eval-split", default=None, help="held-out split name (default: eval or test)")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.add_argument("--log-level", default="INFO")

    synth = sub.add_parser("synth", help="generate synthetic datasets (no config needed)")
    synth_sub = synth.add_subparsers(dest="synth_command", required=True)
    p = synth_sub.add_parser("invoices", help="synthetic supplier invoices for the extraction proof point")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--train", type=int, default=3000)
    p.add_argument("--eval", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-level", default="INFO")
    return parser


def _no_config_command(args: argparse.Namespace) -> int:
    if args.command == "spec" and args.spec_command == "init":
        from .extraction.infer import init_spec

        notes = init_spec(
            args.data, args.out, name=args.name, input_field=args.input_field, label_field=args.label_field,
            eval_path=args.eval_data, force=args.force,
        )
        log.info(
            "draft spec -> %s (%d items to review); next: grpo spec check --spec %s",
            args.out,
            len(notes),
            args.out,
        )
        for n in notes:
            log.info("REVIEW %s", n)
        return 0
    if args.command == "spec" and args.spec_command == "check":
        from .errors import DataError
        from .extraction.selfcheck import check_spec, format_report

        rep = check_spec(args.spec, eval_split=args.eval_split)
        print(json.dumps(rep.to_dict(), indent=2, default=str) if args.json else format_report(rep))
        if not rep.ok:
            raise DataError("spec check failed")
        return 0
    if args.command == "synth":
        from .synth.invoices import generate

        paths = generate(args.out, n_train=args.train, n_eval=args.eval, seed=args.seed)
        print(json.dumps(paths, indent=2))
        return 0
    raise AssertionError(f"unhandled command {args.command}")


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.profile, args.overrides)


# Runtime defaults applied before vLLM is imported (child engine processes inherit os.environ).
# FlashInfer's sampler JIT-compiles a CUDA kernel on first use with whatever nvcc it finds; on hosts
# whose system CUDA differs from torch's (seen: RTX A4000 box, torch cu130, older /usr/local/cuda)
# that build fails and vLLM's engine dies at startup. vLLM's native PyTorch sampler needs no compile.
# setdefault: an explicit VLLM_USE_FLASHINFER_SAMPLER=1 in the environment still wins.
RUNTIME_ENV_DEFAULTS = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def apply_runtime_env_defaults() -> None:
    for key, value in RUNTIME_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)


def main(argv: list[str] | None = None) -> int:
    apply_runtime_env_defaults()
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        if args.command in ("spec", "synth"):
            return _no_config_command(args)
        cfg = _load(args)
        if args.command == "validate":
            log.info(
                "config valid: run=%s fingerprint=%s generation_batch=%d prompts/step=%d",
                cfg.run.name,
                cfg.fingerprint(),
                cfg.generation_batch_size,
                cfg.prompts_per_step,
            )
            if args.print:
                print(json.dumps(cfg.model_dump(mode="json"), indent=2))
            return 0
        if args.command == "preflight":
            from .preflight import enforce, format_results, run_preflight

            results = run_preflight(cfg, require_gpu=not args.no_gpu)
            print(format_results(results))
            enforce(results)
            return 0
        if args.command == "profile":
            from .profiler import run_profile

            path = run_profile(cfg, force=args.force)
            summary = path.with_suffix(".summary.json")
            if summary.is_file():
                print(summary.read_text())
            return 0
        if args.command == "train":
            from .train import run_train

            run_train(cfg, resume=not args.no_resume, force=args.force, skip_preflight=args.skip_preflight)
            return 0
        if args.command == "eval":
            from .evaluate import run_eval

            print(run_eval(cfg, args.model, args.label))
            return 0
        if args.command == "gate":
            return _gate(cfg)
        if args.command == "baseline":
            from .baseline import run_baseline

            result = run_baseline(cfg, args.provider, dry_run=args.dry_run)
            print(json.dumps(result, indent=2) if isinstance(result, dict) else result)
            return 0
        if args.command == "report":
            from .report import run_report

            print(run_report(cfg))
            return 0
        if args.command == "bundle":
            from .bundle import DEFAULT_VLLM_IMAGE, run_bundle

            print(
                run_bundle(
                    cfg, out_dir=args.out, allow_failed_gate=args.allow_failed_gate,
                    smoke_examples=args.smoke_examples, vllm_image=args.vllm_image or DEFAULT_VLLM_IMAGE,
                    force=args.force,
                )
            )
            return 0
        raise AssertionError(f"unhandled command {args.command}")
    except GRPOTrainerError as e:
        log.error("%s", e)
        return e.exit_code
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise GateFailedError(f"corrupt {path} line {n}: {e}") from None
    return rows


def _gate(cfg: Config) -> int:
    from .manifest import atomic_write_json
    from .metrics import evaluate_gate

    eval_dir = cfg.run_dir / "eval"
    labels = ("base", "trained")
    needed = [eval_dir / f"{lb}.json" for lb in labels] + [eval_dir / f"{lb}.examples.jsonl" for lb in labels]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise GateFailedError(f"missing eval outputs: {missing}; run `grpo eval` for base and trained")
    base, cand = (json.loads((eval_dir / f"{lb}.json").read_text()) for lb in labels)
    base_rows, cand_rows = (_read_jsonl(eval_dir / f"{lb}.examples.jsonl") for lb in labels)
    from .tasks import task_for

    task = task_for(cfg)
    result = evaluate_gate(base, cand, cfg.eval.gate, base_rows, cand_rows, task.lenient_correct)
    atomic_write_json(eval_dir / "gate.json", result.to_dict())
    print(json.dumps(result.to_dict(), indent=2))
    log.info(
        "format %+.1f pts | strict accuracy %+.1f pts | lenient (reasoning) %.1f%% -> %.1f%% (%+.1f pts); "
        "only-trained-right %d vs only-base-right %d, p=%.4f%s",
        100 * result.format_delta,
        100 * result.accuracy_delta,
        100 * result.lenient_accuracy_base,
        100 * result.lenient_accuracy_candidate,
        100 * result.lenient_delta,
        result.only_candidate_correct,
        result.only_base_correct,
        result.p_value,
        " (significant)" if result.significant else " (not significant)",
    )
    if not result.passed:
        raise GateFailedError("gate failed: " + "; ".join(result.reasons))
    log.info("gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
