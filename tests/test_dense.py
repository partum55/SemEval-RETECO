import numpy as np
import pytest

from semeval_reteco.config import DATA_DIR, EMB_DIR
from semeval_reteco.data import load_qrels, load_turns
from semeval_reteco.embeddings import encode_queries, run_dense, system_name
from semeval_reteco.evaluate import evaluate


def test_system_name_shows_only_non_default_settings():
    assert system_name("minilm", "history") == "minilm-history"
    assert system_name("minilm", "history", max_len=256) == "minilm-history"
    name = system_name("bge-base", "qonly", recent_first=True, window=2, prefix=False)
    assert name == "bge-base-qonly-last2-newest-noprefix"
    assert system_name("minilm", "qonly", max_len=512, prefix=False) == "minilm-qonly-len512"
    assert system_name("minilm", "history", rebuild=True) == "minilm-history-rebuilt"
    assert system_name("minilm", "history", rebuild=True, recent_first=True) == "minilm-history-newest"
    assert system_name("minilm", "qonly", rebuild=True) == "minilm-qonly"


def test_query_vectors_are_normalized_and_the_prefix_matters():
    texts = ["what voltage does my drone battery need", "what voltage does my drone battery need"]
    vectors = encode_queries("bge-base", texts)
    assert vectors.shape == (2, 768)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3)
    assert not np.allclose(encode_queries("bge-base", texts[:1], prefix=False), vectors[:1])


@pytest.mark.skipif(not (DATA_DIR.exists() and EMB_DIR.exists()), reason="data or embeddings not downloaded")
@pytest.mark.parametrize("model, variant, expected", [("minilm", "history", 0.3213), ("bge-base", "qonly", 0.2838)])
def test_drones_dev_matches_reference(model, variant, expected):
    run = run_dense(model, "drones", load_turns("drones", "dev"), variant)
    assert evaluate(run, load_qrels("drones", "dev"))["all"]["nDCG@10"] == pytest.approx(expected, abs=0.002)
