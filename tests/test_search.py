import faiss
import random

import pytest
from sentence_transformers import SentenceTransformer

from semeval_reteco.config import DATA_DIR, EMB_DIR
from semeval_reteco.data import load_corpus
from semeval_reteco.embeddings import ENCODERS, search

pytestmark = pytest.mark.skipif(not (DATA_DIR.exists() and EMB_DIR.exists()),
    reason="data or embeddings not downloaded")


def check_model(model_name):
    doc_ids, texts = load_corpus("drones")

    random.seed(0)
    picked = random.sample(range(len(doc_ids)), 20)
    picked_texts = [texts[i] for i in picked]

    cfg = ENCODERS[model_name]
    model = SentenceTransformer(cfg["hf"])
    model.max_seq_length = cfg["max_len"]
    query_vecs = model.encode(picked_texts, normalize_embeddings=True)

    found_ids, _ = search(model_name, "drones", query_vecs, k=5)

    for i, ids in zip(picked, found_ids):
        assert ids[0] == doc_ids[i]


def test_minilm_finds_itself():
    check_model("minilm")


def test_bge_finds_itself():
    check_model("bge-base")
