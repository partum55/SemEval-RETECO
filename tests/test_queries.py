import pytest

from semeval_reteco.queries import LABEL, VARIANTS, build_query

QA = [("q1", "a1"), ("q2", "a2"), ("q3", "a3")]


def make_turn(n_prev):
    prev = QA[:n_prev]
    history = "\n".join(f"Q: {q}\nA: {a}" for q, a in prev) or "No previous conversation."
    return {
        "query": "current question",
        "prev_queries": [q for q, _ in prev],
        "prev_qa": prev,
        "gold_history": history,
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_first_turn_is_just_the_query(variant):
    assert build_query(make_turn(0), variant) == "current question"


def test_current_ignores_history():
    assert build_query(make_turn(2), "current") == "current question"


def test_qonly_has_questions_and_no_answers():
    text = build_query(make_turn(2), "qonly")
    assert text == "current question" + LABEL + "Q: q1\nQ: q2"
    assert "a1" not in text and "a2" not in text


def test_last2_keeps_only_the_last_two_pairs():
    text = build_query(make_turn(3), "last2")
    assert text == "current question" + LABEL + "Q: q2\nA: a2\nQ: q3\nA: a3"


def test_history_is_query_label_gold_history():
    turn = make_turn(2)
    assert build_query(turn, "history") == turn["query"] + LABEL + turn["gold_history"]


def test_history_uses_the_dataset_field_even_when_it_differs_from_rebuilt():
    turn = make_turn(2)
    turn["gold_history"] = "Q: q1\nA: a1 (edited upstream)\nQ: q2\nA: a2"
    assert build_query(turn, "history").endswith("a1 (edited upstream)\nQ: q2\nA: a2")


def test_recent_first_puts_the_newest_turn_first():
    text = build_query(make_turn(2), "qonly", recent_first=True)
    assert text.index("Q: q2") < text.index("Q: q1")
    text = build_query(make_turn(2), "history", recent_first=True)
    assert text.index("Q: q2") < text.index("Q: q1")
    assert "A: a1" in text  # pairs stay together


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        build_query(make_turn(2), "histroy")
    with pytest.raises(ValueError):
        build_query(make_turn(0), "histroy")  # also on turn 1, so a typo can't hide there
