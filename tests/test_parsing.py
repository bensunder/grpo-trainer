from decimal import Decimal

import pytest

from grpo_trainer.rewards.parsing import (
    completion_to_text,
    extract_answer,
    format_decimal,
    match_contract,
    numbers_equal,
    parse_gsm8k_ground_truth,
    parse_single_number,
)


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("plain", "plain"),
        ([{"role": "assistant", "content": "hi"}], "hi"),
        (
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                }
            ],
            "ab",
        ),
        ([{"role": "user", "content": "ignored"}, {"role": "assistant", "content": "kept"}], "kept"),
        ({"role": "assistant", "content": "dict"}, "dict"),
        ([{"role": "assistant", "content": None}], ""),
        (None, ""),
        (42, ""),
        ([], ""),
    ],
)
def test_completion_to_text(completion, expected):
    assert completion_to_text(completion) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("72", Decimal(72)),
        ("$72", Decimal(72)),
        ("72.00", Decimal(72)),
        ("72.", Decimal(72)),
        ("1,234", Decimal(1234)),
        ("1,234.50", Decimal("1234.5")),
        ("-5", Decimal(-5)),
        ("−5", Decimal(-5)),  # unicode minus
        ("72 apples", Decimal(72)),
        ("**72**", Decimal(72)),
        (r"\boxed{72}", Decimal(72)),
        ("７２", Decimal(72)),  # full-width digits via NFKC
        ("72 and 72", Decimal(72)),  # repeated same value is fine
        ("0", Decimal(0)),
        ("0.0", Decimal(0)),
        ("", None),
        ("none", None),
        ("72 or 73", None),  # answer spraying rejected
        ("x2", None),  # digit glued to a word is not a standalone number
        ("1,23", None),  # malformed thousands separator is ambiguous
    ],
)
def test_parse_single_number(text, expected):
    assert parse_single_number(text) == expected


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("Some reasoning <<2*3=6>>6\n#### 6", "6"),
        ("blah\n#### 1,234", "1234"),
        ("#### -7", "-7"),
        ("#### 2.50", "2.5"),
        ("no marker 42", None),
        ("#### ", None),
        (None, None),
    ],
)
def test_parse_gsm8k_ground_truth(field, expected):
    assert parse_gsm8k_ground_truth(field) == expected


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("<think>reason</think><answer>5</answer>", True),
        ("  <think>\nreason\n</think>\n\n<answer> 5 </answer>\n", True),
        ("<think></think><answer>5</answer>", True),  # contract shape only; FormatScorer enforces content
        ("<answer>5</answer>", False),
        ("<think>r</think>", False),
        ("preamble <think>r</think><answer>5</answer>", False),
        ("<think>r</think><answer>5</answer> trailing", False),
        ("<think>r</think><answer>5</answer><answer>6</answer>", False),
        ("<think>a<think>b</think><answer>5</answer>", False),
        ("<answer>5</answer><think>r</think>", False),
    ],
)
def test_match_contract(text, ok):
    assert (match_contract(text) is not None) is ok


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<answer>5</answer>", "5"),
        ("x <answer> 5 </answer> y", " 5 "),
        ("<answer>5</answer><answer>6</answer>", None),
        ("<answer>5", None),
        ("5</answer>", None),
        ("<answer><answer>5</answer>", None),
        ("", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(Decimal("72"), "72"), (Decimal("7.2E+2"), "720"), (Decimal("0.50"), "0.5"), (Decimal("-0"), "0")],
)
def test_format_decimal(value, expected):
    assert format_decimal(value) == expected


def test_numbers_equal():
    assert numbers_equal(Decimal("72.0"), "72")
    assert not numbers_equal(Decimal("72"), "73")
    assert not numbers_equal(None, "72")
    assert not numbers_equal(Decimal("72"), None)
    assert not numbers_equal(Decimal("72"), "seventy-two")
