import logging

import pytest

from grpo_trainer.config import RewardSpec
from grpo_trainer.errors import ConfigError
from grpo_trainer.rewards import RewardContext, build_rewards, build_scorer, make_reward
from grpo_trainer.rewards.scorers import CorrectnessScorer, FormatScorer, LengthPenaltyScorer

GOOD = "<think>Natalia sold 48 clips in April and half as many in May.</think><answer>72</answer>"
ROW = {"ground_truth": "72"}


def _call(scorer, text, ids=None, row=ROW):
    return scorer(text, completion_ids=ids, row=row)


# ------------------------------------------------------------------ correctness


@pytest.mark.parametrize(
    ("text", "score"),
    [
        (GOOD, 1.0),
        ("<answer>72</answer>", 1.0),  # correctness is independent of format by default
        ("<think>x</think><answer>$72.00</answer>", 1.0),
        ("<think>x</think><answer>73</answer>", 0.0),
        ("<think>x</think><answer>72 or 73</answer>", 0.0),  # spraying
        ("The answer is 72", 0.0),  # no answer tag
        ("<answer>72</answer><answer>72</answer>", 0.0),  # duplicate tags
        ("", 0.0),
    ],
)
def test_correctness(text, score):
    assert _call(CorrectnessScorer(), text) == score


def test_correctness_require_contract():
    s = CorrectnessScorer(require_contract=True)
    assert _call(s, GOOD) == 1.0
    assert _call(s, "<answer>72</answer>") == 0.0


def test_correctness_missing_column_raises_inside_scorer():
    with pytest.raises(KeyError):
        _call(CorrectnessScorer(), GOOD, row={})


# ------------------------------------------------------------------ format


@pytest.mark.parametrize(
    ("text", "score"),
    [
        (GOOD, 1.0),
        ("<think></think><answer>72</answer>", 0.0),  # empty-think hack
        ("<think>   \n  </think><answer>72</answer>", 0.0),
        ("<think>short</think><answer>72</answer>", 0.0),  # < 20 non-space chars
        ("<think>Natalia sold 48 clips in April and half.</think><answer></answer>", 0.0),
        ("<think>Natalia sold 48 clips in April and half.</think><answer>" + "9" * 65 + "</answer>", 0.0),
        ("<answer>72</answer>", 0.0),
        (GOOD + " extra", 0.0),
    ],
)
def test_format(text, score):
    assert _call(FormatScorer(), text) == score


# ------------------------------------------------------------------ length


@pytest.mark.parametrize(
    ("n_tokens", "expected"),
    [(0, 0.0), (300, 0.0), (576, 0.0), (672, -0.5), (768, -1.0), (2000, -1.0)],
)
def test_length_penalty_tokens(n_tokens, expected):
    s = LengthPenaltyScorer(max_tokens=768, soft_fraction=0.75)
    assert _call(s, "x", ids=list(range(n_tokens))) == pytest.approx(expected)


def test_length_penalty_falls_back_to_chars():
    s = LengthPenaltyScorer(max_tokens=100, soft_fraction=0.5)
    assert _call(s, "a" * 400, ids=None) == pytest.approx(-1.0)  # ~100 tokens
    assert _call(s, "a" * 100, ids=None) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [{"max_tokens": 0}, {"max_tokens": 10, "soft_fraction": 1.0}, {"max_tokens": 10, "soft_fraction": 0.0}],
)
def test_length_penalty_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        LengthPenaltyScorer(**kwargs)


# ------------------------------------------------------------------ TRL-facing wrapper


def test_wrapper_handles_conversational_and_string_completions():
    fn = make_reward("correctness", CorrectnessScorer())
    completions = [
        [{"role": "assistant", "content": GOOD}],
        GOOD,
        [{"role": "assistant", "content": "<answer>1</answer>"}],
    ]
    out = fn(
        prompts=[None] * 3, completions=completions, completion_ids=None, ground_truth=["72", "72", "72"]
    )
    assert out == [1.0, 1.0, 0.0]
    assert fn.__name__ == "correctness"


def test_wrapper_isolates_exceptions_and_reports_metric(caplog):
    class Boom:
        def __call__(self, text, *, completion_ids, row):
            if text == "bad":
                raise RuntimeError("boom")
            return 1.0

    logged = {}
    fn = make_reward("boom", Boom())
    with caplog.at_level(logging.ERROR):
        out = fn(
            prompts=[None] * 3,
            completions=["ok", "bad", "ok"],
            log_metric=lambda k, v: logged.__setitem__(k, v),
            trainer_state=object(),
        )
    assert out == [1.0, 0.0, 1.0]
    assert logged == {"reward_errors/boom": pytest.approx(1 / 3)}
    assert "boom failed" in caplog.text


def test_wrapper_rejects_non_finite_scores():
    class NaN:
        def __call__(self, text, *, completion_ids, row):
            return float("nan")

    assert make_reward("nan", NaN())(prompts=[None], completions=["x"]) == [0.0]


def test_wrapper_column_length_mismatch_scores_zero():
    fn = make_reward("correctness", CorrectnessScorer())
    assert fn(prompts=[None] * 2, completions=[GOOD, GOOD], ground_truth=["72"]) == [0.0, 0.0]


def test_wrapper_passes_completion_ids_per_item():
    fn = make_reward("length", LengthPenaltyScorer(max_tokens=10, soft_fraction=0.5))
    out = fn(prompts=[None] * 2, completions=["a", "b"], completion_ids=[[1] * 5, [1] * 10])
    assert out == [0.0, -1.0]


def test_wrapper_ignores_trl_injected_non_row_kwargs():
    fn = make_reward("correctness", CorrectnessScorer())
    out = fn(
        prompts=[None],
        completions=[GOOD],
        ground_truth=["72"],
        trainer_state=object(),
        log_extra=lambda *a: None,
        log_metric=lambda *a: None,
    )
    assert out == [1.0]


# ------------------------------------------------------------------ registry


def test_build_rewards_from_config(base_config):
    funcs, weights = build_rewards(base_config)
    assert [f.__name__ for f in funcs] == ["correctness", "format", "length"]
    assert weights == [1.0, 0.2, 0.1]


def test_length_factory_uses_run_max_completion_length():
    s = build_scorer(RewardSpec(name="length", weight=0.1), RewardContext(max_completion_length=256))
    assert isinstance(s, LengthPenaltyScorer) and s.max_tokens == 256


@pytest.mark.parametrize(
    "spec",
    [RewardSpec(name="nope", weight=1.0), RewardSpec(name="format", weight=0.1, params={"bogus": 1})],
)
def test_registry_errors_are_config_errors(spec):
    with pytest.raises(ConfigError):
        build_scorer(spec, RewardContext(max_completion_length=64))


def test_correct_answer_always_beats_well_formatted_wrong_one(base_config):
    funcs, weights = build_rewards(base_config)
    wrong = "<think>Natalia sold 48 clips in April and half as many in May.</think><answer>71</answer>"
    lazy_right = "<answer>72</answer>"

    def total(text, n_tokens):
        return sum(
            w * f(prompts=[None], completions=[text], completion_ids=[[0] * n_tokens], ground_truth=["72"])[0]
            for f, w in zip(funcs, weights, strict=True)
        )

    # Worst case for the correct answer: no format credit, max length penalty.
    # Best case for the wrong answer: full format credit, no length penalty.
    assert total(lazy_right, 10_000) > total(wrong, 10)
