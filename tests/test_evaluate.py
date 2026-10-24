import math

import pytest

from semeval_reteco.evaluate import evaluate, macro, write_run

# q1: relevant {a, b}, ranked a, x, b      q2: relevant {c}, ranked x, y, c
# q3 (turn 6 -> T5+): relevant {d}, nothing found
QRELS = {"c1_turn_1": {"a": 1, "b": 1}, "c2_turn_2": {"c": 1}, "c3_turn_6": {"d": 1}}
RUN = {
    "c1_turn_1": {"a": 3.0, "x": 2.0, "b": 1.0},
    "c2_turn_2": {"x": 3.0, "y": 2.0, "c": 1.0},
    "c3_turn_6": {"x": 3.0, "y": 2.0},
}


def test_hand_computed_metrics():
    res = evaluate(RUN, QRELS)
    ndcg1 = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    ndcg2 = (1 / math.log2(4)) / 1
    assert math.isclose(res["T1"]["nDCG@10"], ndcg1)
    assert math.isclose(res["T1"]["MAP@10"], (1 + 2 / 3) / 2)
    assert math.isclose(res["T1"]["Recall@10"], 1)
    assert math.isclose(res["T1"]["MRR"], 1)
    assert math.isclose(res["T2"]["nDCG@10"], ndcg2)
    assert math.isclose(res["T2"]["MAP@10"], 1 / 3)
    assert math.isclose(res["T2"]["MRR"], 1 / 3)
    assert res["T5+"]["nDCG@10"] == 0
    assert math.isclose(res["all"]["nDCG@10"], (ndcg1 + ndcg2) / 3)
    assert res["all"]["n"] == 3 and "T3" not in res


def test_topic_missing_from_run_scores_zero():
    res = evaluate({"c1_turn_1": RUN["c1_turn_1"]}, QRELS)
    assert res["all"]["n"] == 3
    assert res["T2"]["nDCG@10"] == 0


def test_ranks_by_score_regardless_of_dict_order():
    unsorted = {"c2_turn_2": {"x": 3.0, "y": 2.0, "c": 9.0}}
    assert evaluate(unsorted, {"c2_turn_2": {"c": 1}})["all"]["nDCG@10"] == 1


def test_ties_keep_the_run_order_not_doc_id_order():
    qrels = {"c_turn_1": {"a": 1}}
    first = {"c_turn_1": {"z": 0.0, "a": 0.0}}  # tied, z first: a is at rank 2
    last = {"c_turn_1": {"a": 0.0, "z": 0.0}}
    assert evaluate(first, qrels)["all"]["nDCG@10"] == pytest.approx(1 / math.log2(3))
    assert evaluate(last, qrels)["all"]["nDCG@10"] == 1


def test_macro_is_mean_of_domain_means():
    big = evaluate(RUN, QRELS)
    small = evaluate({"c2_turn_2": {"c": 1.0}}, {"c2_turn_2": {"c": 1}})  # nDCG 1.0, one topic
    got = macro({"big": big, "small": small})["nDCG@10"]
    assert math.isclose(got, (big["all"]["nDCG@10"] + 1.0) / 2)


def test_write_run_format_and_untied_scores(tmp_path):
    write_run(tmp_path / "sub" / "r.run", {"c1_turn_1": {"a": 0.0, "b": 0.0, "c": 5.0}}, "tag")
    lines = (tmp_path / "sub" / "r.run").read_text().splitlines()
    assert lines == ["c1_turn_1\tQ0\tc\t1\t3.0\ttag", "c1_turn_1\tQ0\ta\t2\t2.0\ttag", "c1_turn_1\tQ0\tb\t3\t1.0\ttag"]
