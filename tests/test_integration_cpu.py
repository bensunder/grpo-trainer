"""End-to-end `run_train` on CPU with a randomly initialised tiny Qwen2 and a locally built tokenizer.

Exercises the real TRL GRPOTrainer path: dataset rows -> chat template -> generation -> our reward
functions (conversational completions, ground_truth column forwarding) -> LoRA update -> checkpoint ->
final adapter -> merge -> COMPLETE marker -> manifest -> resume. vLLM is disabled (no GPU here).
Skipped when the training stack is not installed.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("trl", reason="full training stack not installed")
pytest.importorskip("peft", reason="full training stack not installed")

from grpo_trainer import data as data_mod  # noqa: E402
from grpo_trainer import train as train_mod  # noqa: E402
from grpo_trainer.tasks import GSM8KTask  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = tmp_path_factory.mktemp("tiny-qwen2")
    specials = [
        "<|pad|>",
        "<|end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
    ]
    tok = Tokenizer(models.BPE(unk_token="<|pad|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["What is 2 plus 3? Natalia sold 48 clips. The answer is 72. reason step by step"] * 50
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=300, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        ),
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<|pad|>", eos_token="<|end|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(out)

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(out, safe_serialization=True)
    return out


def test_run_train_end_to_end_and_resume(tiny_model_dir, tmp_path, monkeypatch, config_factory):
    rows = [{"question": f"What is {i} plus 3?", "answer": f"#### {i + 3}"} for i in range(12)]
    test_rows = [{"question": "What is 100 plus 3?", "answer": "#### 103"}]
    task = GSM8KTask(loader=lambda split: {"train": rows, "test": test_rows}[split])
    monkeypatch.setattr(data_mod, "get_task", lambda name: task)
    monkeypatch.setattr(train_mod, "get_task", lambda name: task)

    # No GPU here: force CPU and fp32 in the mapped TRL arguments.
    real_kwargs = train_mod.grpo_config_kwargs
    monkeypatch.setattr(
        train_mod, "grpo_config_kwargs", lambda cfg, out: {**real_kwargs(cfg, out), "use_cpu": True}
    )

    # A random tiny model scores 0 on every real reward, so every group has zero std -> zero advantage ->
    # zero gradient, and "training" would silently update nothing. This probe reward varies with the
    # sampled text, giving non-zero advantages so the test proves weights actually move.
    import zlib

    from grpo_trainer.rewards import registry

    class TextParityProbe:
        def __call__(self, text, *, completion_ids, row):
            return float(zlib.crc32(text.encode()) % 2)

    monkeypatch.setitem(registry.REGISTRY, "probe", lambda params, ctx: TextParityProbe())

    cfg = config_factory(
        f"run.output_root={tmp_path}",
        "run.name=cpu-e2e",
        "rewards=[{name: correctness, weight: 1.0}, {name: format, weight: 0.2}, "
        "{name: length, weight: 0.1}, {name: probe, weight: 0.5}]",
        f"model.name_or_path={tiny_model_dir}",
        "model.dtype=float32",
        "model.attn_implementation=eager",
        "adapter.mode=lora",
        "adapter.r=4",
        "adapter.alpha=8",
        "grpo.num_generations=2",
        "grpo.per_device_train_batch_size=2",
        "grpo.gradient_accumulation_steps=1",
        "grpo.max_completion_length=16",
        "grpo.max_steps=3",
        "grpo.save_steps=2",
        "grpo.logging_steps=1",
        "grpo.gradient_checkpointing=false",
        "grpo.learning_rate=1.0e-2",
        # A random model never emits EOS, so every completion hits the 16-token cap. With the default
        # mask_truncated_completions=true those are all excluded from the loss (zero gradient).
        "grpo.mask_truncated_completions=false",
        "grpo.warmup_steps=0",
        "grpo.report_to=[tensorboard]",  # the shipped default backend, exercised for real
        "grpo.log_completions=false",
        "vllm.enabled=false",
        "profiler.enabled=false",
    )
    final = train_mod.run_train(cfg, skip_preflight=True)

    run_dir = tmp_path / "cpu-e2e"
    assert final == run_dir / "final"
    complete = json.loads((final / "COMPLETE").read_text())
    assert complete["global_step"] == 3
    assert (final / "adapter" / "adapter_config.json").is_file()
    merged = final / "merged"
    assert (merged / "config.json").is_file() and list(merged.glob("*.safetensors"))
    assert complete["eval_model_path"] == str(merged)
    assert (run_dir / "checkpoints" / "checkpoint-2").is_dir()

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["config_fingerprint"] == cfg.fingerprint()
    assert manifest["train_examples"] == 12

    # Our reward functions were actually called by TRL and logged per name.
    state = json.loads((run_dir / "checkpoints" / "checkpoint-3" / "trainer_state.json").read_text())
    logged = set().union(*(h.keys() for h in state["log_history"]))
    for name in ("correctness", "format", "length", "probe"):
        assert f"rewards/{name}/mean" in logged, sorted(logged)
    assert "frac_reward_zero_std" in logged

    # A real update happened: non-zero gradient on at least one step, and LoRA B (initialised to
    # zero by PEFT) is non-zero in the saved adapter.
    grad_norms = [float(h["grad_norm"]) for h in state["log_history"] if "grad_norm" in h]
    assert grad_norms and max(grad_norms) > 0, grad_norms
    from safetensors.torch import load_file

    weights = load_file(str(final / "adapter" / "adapter_model.safetensors"))
    lora_b = [t for k, t in weights.items() if "lora_B" in k]
    assert lora_b and any(t.abs().sum().item() > 0 for t in lora_b)

    # TensorBoard event files were written by the default report_to backend.
    assert list((run_dir / "checkpoints").rglob("events.out.tfevents.*"))

    def checkpoints():
        return sorted(p.name for p in (run_dir / "checkpoints").glob("checkpoint-*"))

    assert checkpoints() == ["checkpoint-2", "checkpoint-3"]

    # Re-running a completed run is a no-op: no extra training step, resume recorded in the manifest.
    train_mod.run_train(cfg, skip_preflight=True)
    assert checkpoints() == ["checkpoint-2", "checkpoint-3"]
    assert len(json.loads((run_dir / "manifest.json").read_text())["resumes"]) == 1

    # Crash after the last checkpoint but before finalisation: re-run finalises from the checkpoint.
    import shutil

    shutil.rmtree(final)
    train_mod.run_train(cfg, skip_preflight=True)
    assert checkpoints() == ["checkpoint-2", "checkpoint-3"]
    complete = json.loads((final / "COMPLETE").read_text())
    assert complete["global_step"] == 3
    assert "finalised_from_checkpoint" in complete["train_metrics"]
    assert (final / "adapter" / "adapter_config.json").is_file()
    assert not (final / "adapter" / "optimizer.pt").exists()
    assert list((final / "merged").glob("*.safetensors"))


def test_sigterm_checkpoints_then_resume_completes(tiny_model_dir, tmp_path, monkeypatch, config_factory):
    import os
    import signal

    rows = [{"question": f"What is {i} plus 3?", "answer": f"#### {i + 3}"} for i in range(12)]
    task = GSM8KTask(
        loader=lambda split: {"train": rows, "test": [{"question": "q?", "answer": "#### 1"}]}[split]
    )
    monkeypatch.setattr(data_mod, "get_task", lambda name: task)
    monkeypatch.setattr(train_mod, "get_task", lambda name: task)
    real_kwargs = train_mod.grpo_config_kwargs
    monkeypatch.setattr(train_mod, "grpo_config_kwargs", lambda c, o: {**real_kwargs(c, o), "use_cpu": True})
    monkeypatch.setattr(train_mod._StopFlag, "requested", False)

    real_cb = train_mod._graceful_stop_callback

    def preempting_callback():
        inner = real_cb()

        class Preempt(type(inner)):
            fired = False

            def on_step_end(self, args, state, control, **kw):
                if state.global_step == 1 and not Preempt.fired:
                    Preempt.fired = True
                    os.kill(os.getpid(), signal.SIGTERM)  # handled by run_train's handler
                return super().on_step_end(args, state, control, **kw)

        return Preempt()

    monkeypatch.setattr(train_mod, "_graceful_stop_callback", preempting_callback)
    old_term, old_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        cfg = config_factory(
            f"run.output_root={tmp_path}",
            "run.name=preempt",
            f"model.name_or_path={tiny_model_dir}",
            "model.dtype=float32",
            "model.attn_implementation=eager",
            "adapter.r=4",
            "adapter.alpha=8",
            "grpo.num_generations=2",
            "grpo.per_device_train_batch_size=2",
            "grpo.gradient_accumulation_steps=1",
            "grpo.max_completion_length=16",
            "grpo.max_steps=3",
            "grpo.save_steps=100",
            "grpo.gradient_checkpointing=false",
            "grpo.report_to=[none]",
            "grpo.log_completions=false",
            "vllm.enabled=false",
            "profiler.enabled=false",
        )
        run_dir = tmp_path / "preempt"
        out = train_mod.run_train(cfg, skip_preflight=True)
        assert out == run_dir / "checkpoints"
        assert (run_dir / "checkpoints" / "checkpoint-1").is_dir()  # saved despite save_steps=100
        assert not (run_dir / "final").exists()

        monkeypatch.setattr(train_mod._StopFlag, "requested", False)
        out = train_mod.run_train(cfg, skip_preflight=True)
        assert out == run_dir / "final"
        assert json.loads((run_dir / "final" / "COMPLETE").read_text())["global_step"] == 3
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
