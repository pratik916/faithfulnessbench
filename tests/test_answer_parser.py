"""Free-text answer parsing for real CoT (fb-b5t.2).

Synthetic CoT always ends with a clean `ANSWER: <X>` line, so the parser's messy-text
fallbacks were never exercised. Real models emit LaTeX `\\boxed{}`, "the answer is N",
"Final answer: N", and stray letters/digits in prose. These pin the precedence:
explicit ANSWER line > \\boxed{} > "answer is/:" phrase > weak last-resort fallback.
"""
from __future__ import annotations

from faithfulnessbench.models.anthropic_model import _parse_answer
from faithfulnessbench.problems import mixed_problems

_PS = mixed_problems(6, seed=0)
ARITH = next(p for p in _PS if p.domain == "arithmetic")
MCQ = next(p for p in _PS if p.domain == "mcq")


def test_explicit_answer_line_wins_over_stray_reasoning_digits():
    assert _parse_answer("we get 7 here, then 13\nANSWER: 42", ARITH) == "42"


def test_latex_boxed_integer():
    assert _parse_answer("After simplifying we obtain \\boxed{42}.", ARITH) == "42"


def test_answer_is_phrase_beats_last_integer_in_prose():
    assert _parse_answer("Intermediate values 12, 7, 99. The answer is 42. Logged at 3pm.", ARITH) == "42"


def test_final_answer_colon_phrase():
    assert _parse_answer("Final answer: 42", ARITH) == "42"


def test_boxed_beats_a_trailing_unrelated_integer():
    assert _parse_answer("compute 99 then correct to \\boxed{42}. Done at 5.", ARITH) == "42"


def test_arithmetic_no_answer_is_unknown():
    assert _parse_answer("I cannot solve this.", ARITH) == "?"


def test_last_integer_remains_the_final_fallback():
    assert _parse_answer("the running total reaches 42", ARITH) == "42"


def test_mcq_explicit_answer_line_wins():
    assert _parse_answer("Option B looks plausible. ANSWER: D", MCQ) == "D"


def test_mcq_answer_is_parenthesized_letter_beats_stray_letters():
    assert _parse_answer("A common trap is D, but the answer is (B).", MCQ) == "B"


def test_mcq_boxed_letter():
    assert _parse_answer("Therefore \\boxed{C}.", MCQ) == "C"


def test_mcq_unknown_when_no_letter():
    assert _parse_answer("none of these seem right", MCQ) == "?"
