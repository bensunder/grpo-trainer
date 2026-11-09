# grpo-trainer

Production GRPO harness on Hugging Face TRL. You own **tasks, rewards and eval**; TRL owns the optimizer.
Everything else — config validation, version pinning, difficulty profiling, preemption-safe resume,
adapter merge, held-out gate — is handled here so a run either fails in seconds or produces a
reproducible, comparable result.

```
preflight ──► profile ──► train ──► eval(base) + eval(trained) ──► gate
 versions,     drop        TRL GRPO    held-out test split           accuracy delta,
 GPU, VRAM,    prompts     + vLLM      greedy + pass@k               format regression
 memory est.   with zero   colocate,                                 
               gradient    LoRA/QLoRA
```

## Pinned stack

| Package | Version | Why pinned |
|---|---|---|
| trl | 1.17.1 | `GRPOConfig` fields change between releases (`max_prompt_length` is gone) |
| vllm | 0.32.0 | TRL 1.17.1 requires `vllm>=0.22,<0.33` |
| torch | 2.13.0 | hard-pinned by vLLM 0.32.0; PyPI wheels are **CUDA 13** |
| transformers | 5.19.0 | vLLM requires `<5.20`; 5.x removed `warmup_ratio` |
| peft / accelerate / datasets / bitsandbytes | 0.21.2 / 1.15.0 / 5.1.0 / 0.50.2 | |
| tensorboard | 2.21.0 | default `report_to`; transformers raises at Trainer init without it |

`requirements.lock` is the full 214-package closure, resolved for Python 3.12 / Linux x86_64.
`grpo preflight` refuses to run on any drift. Regenerate with `scripts/relock.sh`.

**Host requirement:** an NVIDIA driver that supports CUDA 13 (`nvidia-smi` header shows `CUDA Version: 13.x`).

## Quick start (GPU box)

```bash
# 1. Build (or: python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.lock && pip install --no-deps -e .)
docker build -t grpo-trainer .

# 2. Shell into the container with GPUs, persistent runs/ and HF cache
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD/runs:/app/runs" -v "$HOME/.cache/huggingface:/cache/huggingface" \
  --entrypoint bash grpo-trainer

# 3. Inside the container: smoke test (~20-40 min on 24 GB), then the real run
scripts/smoke.sh gpu24
scripts/pipeline.sh gpu24
```

## Commands

| Command | What it does |
|---|---|
| `grpo validate --config configs/base.yaml --profile gpu24 [--print]` | Config only. No GPU, no network. |
| `grpo preflight ...` | Versions, CUDA, GPU count, VRAM, bf16, disk, memory estimate. Non-zero exit on FAIL. |
| `grpo profile ...` | Samples the base model K times per prompt; caches pass rates under `runs/_cache/profiles/`. |
| `scripts/train.sh gpu24 [--set k=v]` | `accelerate launch` training. Resumes automatically; refuses to resume with a changed config. |
| `grpo eval ... --model <id or path> --label base\|trained` | Held-out greedy accuracy, format rate, truncation, pass@k, per-example JSONL. |
| `grpo gate ...` | Compares the two reports; exit 6 if the trained model doesn't clear the gate. |

Any config value can be overridden: `--set grpo.max_steps=50 --set eval.limit=200`.
Exit codes: `2` config, `3` preflight, `4` data, `5` manifest mismatch, `6` gate failed.

## Hardware profiles

| Profile | Default model | Mode | Notes |
|---|---|---|---|
| `gpu24` | Qwen2.5-1.5B-Instruct | LoRA + vLLM colocate, sleep mode | 7B possible with `adapter.mode=qlora`, batch 2 — tight |
| `gpu80` | Qwen2.5-7B-Instruct | LoRA r=64 + vLLM colocate | |
| `multi` | Qwen2.5-7B-Instruct | full fine-tune, ZeRO-3 on 3 GPUs, vLLM server on a 4th | needs `pip install '.[multi]'` (deepspeed) |

`multi` launch:
```bash
CUDA_VISIBLE_DEVICES=0 scripts/vllm_server.sh Qwen/Qwen2.5-7B-Instruct &
CUDA_VISIBLE_DEVICES=1,2,3 scripts/train.sh multi
```

## Design decisions

- **Correctness dominates the reward.** Config validation rejects any reward set where shaping weights
  (format + length) sum to ≥ the correctness weight, so a well-formatted wrong answer can never beat a
  correct one. The format reward scores 0 for an empty or trivial `<think>` block.
- **Answer spraying scores 0.** `<answer>72 or 73</answer>` has two distinct numbers → no credit.
- **Rewards never crash training.** Each item is scored inside an isolation boundary; exceptions and
  NaN/inf become 0.0 and are reported as `reward_errors/<name>`.
- **Difficulty profiling.** A prompt solved 0/K or K/K times gives zero advantage and zero gradient but
  still costs a full rollout. Profiling once, with training's own sampling settings, keeps only prompts
  in the learnable band. Watch `frac_reward_zero_std` during training — it should stay well below 1.
- **No reference model by default** (`beta: 0.0`): no KL term, one fewer model in memory.
- **`loss_type: dapo`, `mask_truncated_completions: true`**: no per-sequence length bias; rambling to the
  token cap earns nothing.
- **Idempotent stages.** Profile is cached by a key over model, sample, sampling params, system prompt
  and reward params. Re-running a finished `train` is a no-op; a crash after the last checkpoint
  finalises from that checkpoint instead of training an extra step.
- **Preemption.** SIGTERM/SIGINT → checkpoint at the end of the current step → exit. Re-run the same
  command to resume. A second signal hard-exits.
- **Run manifest** (`runs/<name>/manifest.json`): config fingerprint, dataset fingerprint, package
  versions, git revision, host, resume history.

## Metrics to watch (TensorBoard: `runs/<name>/checkpoints`)

| Metric | Healthy | Problem |
|---|---|---|
| `rewards/correctness/mean` | rising | flat while `rewards/format/mean` rises → format hacking |
| `frac_reward_zero_std` | well below 1 | near 1 → no learning signal; re-profile or change band |
| `completions/clipped_ratio` | low | near 1 → truncated completions are masked out of the loss, so training is a no-op; raise `max_completion_length` |
| `reward_errors/*` | 0 | > 0 → a reward function is throwing; check logs |

## Extending

- **New reward:** add a scorer class in `rewards/scorers.py` (pure function of text, token ids and
  row) and one entry in `rewards/registry.py`. Reference it by name in `rewards:` in the config.
- **New task:** implement `Task.load(split, limit) -> list[Example]` in `tasks/<name>.py` and register
  it in `tasks/__init__.py`. Any extra column your reward needs goes on `Example.to_row`.

## Verification status

Verified in CPU runs (`pytest`, 192 tests):
- Parsing, reward scoring and the TRL-facing wrapper; config validation and overrides; tasks,
  contamination removal, profiling selection; pass@k and gate; manifest/resume guards; memory estimates.
- **Real TRL 1.17.1 `GRPOTrainer` end-to-end on CPU** with a randomly initialised tiny Qwen2 (vLLM off,
  3 steps): rewards called and logged by name; a non-zero gradient on the step whose group had reward
  variance and zero gradient on zero-variance steps; non-zero LoRA B weights in the saved adapter;
  TensorBoard event files; checkpoints, merge, `COMPLETE`, idempotent re-run, crash-after-checkpoint
  recovery, and SIGTERM → checkpoint → resume to completion. This proves the plumbing, not learning:
  a random model can't improve on GSM8K.
- `GRPOConfig` constructed from every shipped profile against the real installed TRL.
- Profiler and eval stage logic against a deterministic fake vLLM engine.
- The full pinned set resolves without conflicts for Python 3.12 / Linux (dry-run resolution; vLLM and
  bitsandbytes were not installed or imported here).

**Not yet verified (needs a GPU host):** vLLM colocate/server generation, the real `vllm.LLM.chat`
call path in profile/eval, QLoRA weight sync to vLLM, ZeRO-3, the Docker build (base image tag
`nvidia/cuda:13.0.1-devel-ubuntu24.04` — override with `--build-arg BASE_IMAGE=...` if it differs),
and the memory estimates against measured VRAM. `scripts/smoke.sh` exercises all of the single-GPU items.
