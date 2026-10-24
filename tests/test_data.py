import pytest

from semeval_reteco.config import DATA_DIR, DOMAINS
from semeval_reteco.data import load_corpus, load_turns, load_qrels

pytestmark = pytest.mark.skipif(not DATA_DIR.exists(), reason="data not downloaded")


def test_counts():
    n_docs = sum(len(load_corpus(d)[0]) for d in DOMAINS)
    n_train = sum(len(load_turns(d, "train")) for d in DOMAINS)
    n_dev = sum(len(load_turns(d, "dev")) for d in DOMAINS)
    assert n_docs == 507_141
    assert n_train == 2_113
    assert n_dev == 858


def test_every_turn_has_qrels():
    for d in DOMAINS:
        for split in ("train", "dev"):
            qrels = load_qrels(d, split)
            for turn in load_turns(d, split):
                assert turn["qid"] in qrels, (d, split, turn["qid"])