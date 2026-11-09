import json

import pytest

from grpo_trainer import data as data_mod
from grpo_trainer.data import (
    Splits,
    eval_examples,
    profiling_sample,
    read_profile,
    select_learnable,
    training_examples,
)
from grpo_trainer.errors import DataError
from grpo_trainer.profiler import profile_key, summarize
from grpo_trainer.tasks import (
    DEFAULT_SYSTEM_PROMPT,
    Example,
    GSM8KTask,
    assert_disjoint,
    dataset_fingerprint,
    drop_contaminated,
)


def rows(n, offset=0):
    return [{"question": f"Question {i + offset}?", "answer": f"work\n#### {i + offset}"} for i in range(n)]


def make_task(train, test):
    data = {"train": train, "test": test}
    return GSM8KTask(loader=lambda split: data[split])


def test_gsm8k_parses_rows_and_skips_bad_ones(caplog):
    bad = [
        {"question": "", "answer": "#### 1"},
        {"question": "q", "answer": "no marker"},
        {"answer": "#### 2"},
    ]
    ex = make_task(rows(3) + bad, []).load("train")
    assert [e.ground_truth for e in ex] == ["0", "1", "2"]
    assert ex[0].id == "train-0"
    assert "skipped 3 rows" in caplog.text


def test_gsm8k_limit_is_deterministic():
    t = make_task(rows(10), [])
    assert t.load("train", 4) == t.load("train", 4)
    assert len(t.load("train", 4)) == 4


def test_gsm8k_empty_split_is_data_error():
    with pytest.raises(DataError):
        make_task([], []).load("train")


def test_gsm8k_loader_failure_is_data_error():
    def boom(split):
        raise ConnectionError("offline")

    with pytest.raises(DataError, match="offline"):
        GSM8KTask(loader=boom).load("train")


def test_example_row_shape():
    e = Example(id="train-1", question="What is 2+2?", ground_truth="4")
    row = e.to_row(DEFAULT_SYSTEM_PROMPT)
    assert row["ground_truth"] == "4"
    assert row["prompt"][0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert row["prompt"][1]["role"] == "user"
    assert "<think>" in DEFAULT_SYSTEM_PROMPT and "<answer>" in DEFAULT_SYSTEM_PROMPT


def test_contamination_drop_and_assert():
    train = [Example("a", "Same  question?", "1"), Example("b", "Other", "2")]
    test = [Example("t", "same question?", "1")]  # differs only in case/whitespace
    clean, dropped = drop_contaminated(train, test)
    assert dropped == 1 and [e.id for e in clean] == ["b"]
    with pytest.raises(DataError):
        assert_disjoint(train, test)
    assert_disjoint(clean, test)


def test_dataset_fingerprint_sensitive_to_content():
    a = [Example("1", "q", "1")]
    assert dataset_fingerprint(a) == dataset_fingerprint([Example("1", "q", "1")])
    assert dataset_fingerprint(a) != dataset_fingerprint([Example("1", "q", "2")])


def test_load_splits_drops_contamination(monkeypatch, config_factory):
    cfg = config_factory("task.train_limit=3")
    train = rows(5)
    test = [{"question": "Question 1?", "answer": "#### 1"}]
    monkeypatch.setattr(data_mod, "get_task", lambda name: make_task(train, test))
    splits = data_mod.load_splits(cfg)
    assert splits.contaminated_dropped == 1
    assert [e.question for e in splits.train] == ["Question 0?", "Question 2?", "Question 3?"]


def test_profiling_sample_is_seeded_and_capped(config_factory):
    cfg = config_factory("profiler.max_prompts=5")
    ex = [Example(str(i), f"q{i}", str(i)) for i in range(50)]
    s1, s2 = profiling_sample(cfg, ex), profiling_sample(cfg, ex)
    assert s1 == s2 and len(s1) == 5
    assert profiling_sample(config_factory("profiler.max_prompts=5", "run.seed=7"), ex) != s1


def test_eval_examples_limit(config_factory):
    held = [Example(str(i), f"q{i}", "1") for i in range(10)]
    assert len(eval_examples(config_factory("eval.limit=3"), held)) == 3
    assert len(eval_examples(config_factory(), held)) == 10


def _write_profile(path, rates):
    path.write_text("".join(json.dumps({"id": k, "pass_rate": v}) + "\n" for k, v in rates.items()))


def test_select_learnable_keeps_only_mixed_outcomes(tmp_path, config_factory):
    cfg = config_factory()
    sample = [Example(i, f"q{i}", "1") for i in ("a", "b", "c", "d")]
    p = tmp_path / "p.jsonl"
    _write_profile(p, {"a": 0.0, "b": 0.5, "c": 1.0, "d": 0.125})
    kept = select_learnable(cfg, sample, read_profile(p))
    assert [e.id for e in kept] == ["b", "d"]


def test_select_learnable_custom_band(config_factory):
    cfg = config_factory("profiler.keep_min_pass_rate=0.2", "profiler.keep_max_pass_rate=0.8")
    sample = [Example(i, i, "1") for i in ("a", "b", "c")]
    assert [e.id for e in select_learnable(cfg, sample, {"a": 0.125, "b": 0.5, "c": 0.875})] == ["b"]


def test_select_learnable_errors(config_factory):
    cfg = config_factory()
    sample = [Example("a", "q", "1")]
    with pytest.raises(DataError, match="missing"):
        select_learnable(cfg, sample, {})
    with pytest.raises(DataError, match="learnable band"):
        select_learnable(cfg, sample, {"a": 1.0})


def test_read_profile_rejects_corruption(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"id": "a", "pass_rate": 0.5}\nnot json\n')
    with pytest.raises(DataError, match="line 2"):
        read_profile(p)


def test_training_examples_requires_profile_when_enabled(config_factory):
    cfg = config_factory()
    splits = Splits(train=[Example("a", "q", "1")], held_out=[], contaminated_dropped=0)
    with pytest.raises(DataError, match="grpo profile"):
        training_examples(cfg, splits, None)
    assert training_examples(config_factory("profiler.enabled=false"), splits, None) == splits.train


def test_profile_key_tracks_sampling_settings(config_factory):
    sample = [Example("a", "q", "1")]
    base = profile_key(config_factory(), sample)
    assert base == profile_key(config_factory("grpo.learning_rate=3.0e-5"), sample)  # training-only knob
    assert base != profile_key(config_factory("grpo.temperature=0.7"), sample)
    assert base != profile_key(config_factory("model.name_or_path=other/model"), sample)
    assert base != profile_key(config_factory(), [Example("a", "q", "2")])


def test_summarize_histogram():
    s = summarize({"a": 0.0, "b": 0.5, "c": 1.0, "d": 1.0}, k=8, lo=0.0, hi=1.0)
    assert (s["prompts"], s["never_solved"], s["always_solved"], s["kept"]) == (4, 1, 2, 1)
    assert s["histogram_correct_of_k"]["4"] == 1
