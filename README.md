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

`requirements.lock` is the full 214-package closure, resolved with `uv pip compile` for Python 3.12 /
Linux x86_64 and checked with a dry-run install for that target.
`grpo preflight` refuses to run on any drift. Regenerate with `scripts/relock.sh`.

**Host requirement:** an NVIDIA driver that supports CUDA 13 (`nvidia-smi` header shows `CUDA Version: 13.x`).

## Quick start (GPU box)

```bash
# One command: detects GPU -> picks profile, checks driver (CUDA 13) and disk, installs uv + Python 3.12
# + the locked stack inside /workspace (or $HOME/grpo), verifies torch sees the GPU, runs preflight.
bash scripts/install.sh            # add --check to only inspect the machine; re-running is safe

# Every new shell afterwards:
source /workspace/grpo-activate.sh # (or $HOME/grpo/grpo-activate.sh)

bash scripts/smoke.sh $GRPO_PROFILE      # ~20-40 min end-to-end proof
bash scripts/pipeline.sh $GRPO_PROFILE   # the real run
```

Docker alternative: `docker build -t grpo-trainer .` (base image tag unverified; see below).

## One-script update + run (GridShare)

From your Mac, after downloading the newest `grpo-trainer.zip`:

```bash
bash ~/Downloads/mac_go.sh     # copy of scripts/mac_go.sh
```

Uploads the newest zip, updates the server code in place (`runs/` kept), re-gates the smoke results, and
-- only if you answer `y` -- starts the full pipeline detached (`scripts/run_all.sh --full`), logging to
`/workspace/pipeline-<timestamp>.log`. One password prompt (SSH connection sharing). For a different
machine: `GRPO_PORT=... GRPO_ALIAS=gridshare-<id> bash ~/Downloads/mac_go.sh`.

## Commands

| Command | What it does |
|---|---|
| `grpo validate --config configs/base.yaml --profile gpu24 [--print]` | Config only. No GPU, no network. |
| `grpo preflight ...` | Versions, CUDA, GPU count, VRAM, bf16, disk, memory estimate. Non-zero exit on FAIL. |
| `grpo profile ...` | Samples the base model K times per prompt; caches pass rates under `runs/_cache/profiles/`. |
| `scripts/train.sh gpu24 [--set k=v]` | `accelerate launch` training. Resumes automatically; refuses to resume with a changed config. |
| `grpo eval ... --model <id or path> --label base\|trained` | Held-out greedy accuracy, format rate, truncation, pass@k, per-example JSONL. |
| `grpo gate ...` | Compares the two models on the same questions (see Gate below); exit 6 if it fails. |
| `grpo baseline ... --provider NAME [--dry-run]` | **Paid.** Runs a frontier API model on the same held-out split, prompt and grader. |
| `grpo report ...` | Proof report (HTML + JSON) from saved outputs: paired tests, per-field accuracy, cost model, lineage. |
| `grpo bundle ...` | Packages a gate-passing model: weights, vLLM + gateway compose, smoke test, checksums. |
| `grpo spec init --data f.jsonl --out s.yaml --name x` | Drafts an extraction spec by inferring field types from labels. |
| `grpo spec check --spec s.yaml` | Validates every label and self-tests the compiled reward. No GPU. |
| `grpo synth invoices --out data/invoices` | Deterministic synthetic invoices for the extraction proof point. |

Any config value can be overridden: `--set grpo.max_steps=50 --set eval.limit=200`.
Exit codes: `2` config, `3` preflight, `4` data, `5` manifest mismatch, `6` gate failed, `7` budget,
`8` provider (API key, auth, circuit open, error rate).

## Structured extraction pilot (verified small-model replacement)

The commercial use of the harness: replace a frontier API on a narrow, checkable task with a small model
the customer owns, and prove it with numbers a buyer can audit. GSM8K stays the harness test; extraction is
the product. The task owns its grading (`Task.correctness_scorer`, `format_scorer`, `lenient_correct`,
`eval_detail`, `parse_prediction`), so eval, profiling, the gate, the report and the gateway are shared code.

```bash
grpo synth invoices --out data/invoices                 # or the customer's labelled JSONL
grpo spec check --spec specs/invoice.yaml               # labels + reward self-test, seconds, no GPU
CONFIG=configs/extraction.yaml bash scripts/pipeline.sh gpu16
# frontier comparison (PAID; dry-run prints the worst-case cost first):
grpo baseline --config configs/extraction.yaml --profile gpu16 --provider <name> --dry-run
grpo report   --config configs/extraction.yaml --profile gpu16
grpo bundle   --config configs/extraction.yaml --profile gpu16
```

**Spec (`specs/*.yaml`)** declares fields and how each is matched: `string` (exact / normalized / fuzzy),
`number` (absolute/relative tolerance), `integer`, `date` (ISO plus declared input formats), `enum`,
`boolean`, `list` (unordered, F1) and `object_list` (records aligned greedily, soft F1). Options that do not
apply to a type are errors. `grpo spec init` drafts one from labels and marks what to review.

**Reward.** Correctness = weighted mean of field scores (or all-or-nothing), exactly 1.0 only when every field
is right; null-aware (an invented value and a missed value are both wrong); duplicate JSON keys score 0;
list spraying scores 0. Values are coerced leniently for correctness (`"$1,234.50"` is 1234.5), while the
format reward demands the exact contract: tags, a bare JSON object, exactly the declared keys, JSON types,
ISO dates, canonical enum values. That split is what lets the report separate "learned the schema" from
"extracts the right values".

**`grpo spec check`** validates every label (all errors listed), checks duplicate ids and train/eval
overlap, flags fields with no signal (always null, constant), then self-tests the compiled reward on every
example: the perfect answer must score 1.0 on correctness, format and lenient; per-field mutations (wrong
value drawn from the same column, dropped value, invented value, dropped list item) must lower the score,
so a too-loose tolerance or fuzzy threshold is caught before training; adversarial outputs (empty, garbage,
array, duplicate keys, extra keys, code fence, no tags) must not earn credit they shouldn't. On the shipped
invoice spec: 47,042 mutations, all caught.

**Profiling** uses `profiler.select: reward_variance` for partial credit: a document never fully solved still
has gradient if its samples get different field scores.

**Baseline** (`baseline.providers`: `openai`, `azure_openai`, `anthropic`) uses the identical prompt and
grader and writes an eval-compatible report with measured cost and latency. Guards: worst-case estimate
refused above `baseline.max_cost_usd`; spend tracked and capped; every response cached so a re-run only pays
for what is missing; 401/403 abort; consecutive failures trip a circuit breaker; reports refused above
`max_error_rate`. Keys come from environment variables and are never written to disk. Prices are per
million tokens and must be copied from the provider's current price sheet.

**Report** (`<run>/report/proof-report.html` + `.json`) recomputes everything from saved outputs, pairs every
model with the trained one on the same questions (exact McNemar + 95% CI), and generates its claims from the
statistics: "significantly more/less accurate" only at p <= alpha, otherwise "no statistically significant
difference" with the interval. Per-field accuracy, error taxonomy (wrong / omitted / hallucinated / invalid),
cost per 1k requests and monthly cost at `economics.monthly_volume`, and full lineage. Input excerpts are off
by default (`report.include_examples`), since they are customer data.

**Bundle** (`<run>/bundle/`): refuses a failed or missing gate unless `--allow-failed-gate`. Contains the
merged weights, `docker-compose.yml` (vLLM on the GPU, unpublished; gateway on 8080), the gateway (FastAPI,
pinned), the exact training system prompt (the gateway refuses to start if its sha256 does not match),
held-out smoke examples with `smoke_test.py`, and `BUNDLE.json` with the sha256 of every file. API:
`POST /v1/predict {"input": ...}` returns typed `output`, `valid` (false = contract broken: route to review,
don't fail the batch), `errors`, `usage`, `latency_ms`; optional bearer auth, Idempotency-Key (409 on reuse
with a different body), 413 on oversize input, 429 when saturated, circuit breaker on the upstream,
`/healthz`, `/readyz`, `/v1/info`.

## Hardware profiles

| Profile | Default model | Mode | Notes |
|---|---|---|---|
| `gpu24` | Qwen2.5-1.5B-Instruct | LoRA + vLLM colocate, sleep mode | 7B possible with `adapter.mode=qlora`, batch 2 — tight |
| `gpu16` | Qwen2.5-1.5B-Instruct | LoRA, micro-batch 2 x 16 accum, vLLM 0.3 | RTX A4000 / 16 GB cards; est. ~12 GiB |
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

## Gate: format gain vs reasoning gain

Strict accuracy (answer must be inside `<answer>` tags) mixes two things: learning the output format and
actually knowing more answers. The gate separates them, recomputing everything from the saved
per-question outputs:

| Reported | Meaning | Default requirement |
|---|---|---|
| `format_delta` | change in full `<think>/<answer>` compliance | must not regress |
| `accuracy_delta` | strict accuracy change (format + knowledge) | >= +3 pts |
| `lenient_delta` | last number in the output correct, format ignored (knowledge only) | >= 0 (no reasoning regression) |
| `only_base_correct` / `only_candidate_correct` / `p_value` | exact paired McNemar test on the same questions | reported; enforce with `eval.gate.require_significant_gain=true` |

First real result (RTX A4000, Qwen2.5-1.5B, 20 steps, 100 test questions): format 58% -> 100%, strict
41% -> 76%, lenient 66% -> 76%; 14 questions fixed vs 4 broken, p = 0.031. Most of the strict gain was
format; the ~10-pt lenient gain was significant at 5% on this sample.

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

Verified in CPU runs (`pytest`; 280 tests, of which 6 need the full training stack):
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

- Extraction: spec validation, field matchers, grading, task loading, `spec check` self-tests (including the
  shipped invoice spec on generated data), `spec init`; baseline against a fake transport (request shapes for
  all three APIs, retries with Retry-After, auth abort, circuit breaker, budget refusal, cache resume);
  report statistics, claims, comparability guards and well-formed escaped HTML; bundle gate refusal,
  checksums and tamper detection; gateway contract, auth, idempotency, limits and breaker; the bundled
  gateway booted with real uvicorn from the bundle's own source.

**Not yet verified for extraction:** a real GPU run (profile, train, eval) on invoices, a live frontier
API call, the vLLM serving image tag `vllm/vllm-openai:v0.32.0` (override with `--vllm-image`), and the
compose stack end to end on a GPU host (`smoke_test.py` does that). The 6 training-stack tests did not run
in the environment where the extraction work was written; run `pytest` on the GPU box after installing.

**Not yet verified (needs a GPU host):** vLLM colocate/server generation, the real `vllm.LLM.chat`
call path in profile/eval, QLoRA weight sync to vLLM, ZeRO-3, the Docker build (base image tag
`nvidia/cuda:13.0.1-devel-ubuntu24.04` — override with `--build-arg BASE_IMAGE=...` if it differs),
and the memory estimates against measured VRAM. `scripts/smoke.sh` exercises all of the single-GPU items.
