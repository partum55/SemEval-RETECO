import pytest

from semeval_reteco.stats import paired_bootstrap


def metrics(per_qid):
    return {("d", "dev"): {qid: {"ndcg_cut_10": x} for qid, x in per_qid.items()}}


def test_identical_systems_have_a_zero_interval():
    a = metrics({"c1_turn_1": 0.5, "c2_turn_1": 0.2})
    assert paired_bootstrap(a, a, resamples=200) == (0, 0, 0)


def test_point_is_the_turn_mean_and_a_constant_gap_has_no_spread():
    a = metrics({"c1_turn_1": 0.6, "c1_turn_2": 0.6, "c2_turn_1": 0.3})
    b = metrics({"c1_turn_1": 0.5, "c1_turn_2": 0.5, "c2_turn_1": 0.2})
    point, low, high = paired_bootstrap(a, b, resamples=200)
    assert point == pytest.approx(0.1) and low == pytest.approx(0.1) and high == pytest.approx(0.1)


def test_macro_weights_domains_equally():
    a = {("big", "dev"): {f"c{i}_turn_1": {"ndcg_cut_10": 1.0} for i in range(9)},
        ("small", "dev"): {"c0_turn_1": {"ndcg_cut_10": 0.0}}}
    b = {key: {qid: {"ndcg_cut_10": 0.0} for qid in topics} for key, topics in a.items()}
    assert paired_bootstrap(a, b, resamples=200)[0] == pytest.approx(0.5)
