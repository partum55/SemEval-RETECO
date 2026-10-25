import pytest

from semeval_reteco.bm25 import Bm25, analyze, run_bm25
from semeval_reteco.config import DATA_DIR
from semeval_reteco.data import load_qrels, load_turns
from semeval_reteco.evaluate import evaluate

needs_data = pytest.mark.skipif(not DATA_DIR.exists(), reason="data not downloaded")


def test_analyzer_matches_the_official_one():
    assert analyze(["The PWM frequency on my ESC, running 3.5mm motors; don't"]) == [
        ["pwm", "frequenc", "my", "esc", "run", "3.5mm", "motor", "don't"]]


def test_analyzer_keeps_one_letter_tokens_and_drops_possessive_s():
    assert analyze(["a x y drone's"]) == [["x", "y", "drone"]]


@needs_data
@pytest.mark.parametrize("variant, expected", [("current", 0.1695), ("history", 0.2954)])
def test_drones_dev_matches_official_baseline(variant, expected):
    turns = load_turns("drones", "dev")
    run = run_bm25(Bm25("drones"), turns, variant)
    got = evaluate(run, load_qrels("drones", "dev"))["all"]["nDCG@10"]
    assert got == pytest.approx(expected, abs=0.001)
