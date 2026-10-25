import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from semeval_reteco.config import EMB_DIR
from semeval_reteco.data import load_corpus
from semeval_reteco.queries import build_query

ENCODERS = {
    "minilm": dict(
        hf="sentence-transformers/all-MiniLM-L6-v2",
        max_len=256,
        query_prefix="",
    ),
    "bge-base": dict(
        hf="BAAI/bge-base-en-v1.5",
        max_len=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}


def encode_corpus(model_name, domain, device=None, batch_size=64):
    out_dir = EMB_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"{domain}.npy"
    ids_path = out_dir / f"{domain}.ids.json"

    if npy_path.exists() and ids_path.exists():
        print("skip (already done):", model_name, domain)
        return npy_path

    doc_ids, texts = load_corpus(domain)

    cfg = ENCODERS[model_name]
    model = SentenceTransformer(cfg["hf"], device=device)
    model.max_seq_length = cfg["max_len"]

    vectors = model.encode(texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True)

    tmp_npy = out_dir / f"{domain}.tmp.npy"
    tmp_ids = out_dir / f"{domain}.tmp.json"
    np.save(tmp_npy, vectors.astype(np.float16))
    with open(tmp_ids, "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    tmp_npy.rename(npy_path)
    tmp_ids.rename(ids_path)

    return npy_path

_index_cache = {}


def load_index(model_name, domain):
    key = (model_name, domain)
    if key not in _index_cache:
        vectors = np.load(EMB_DIR / model_name / f"{domain}.npy")
        vectors = vectors.astype(np.float32)
        with open(EMB_DIR / model_name / f"{domain}.ids.json", encoding="utf-8") as f:
            doc_ids = json.load(f)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        _index_cache[key] = (index, doc_ids)
    return _index_cache[key]


def search(model_name, domain, query_vecs, k=100):
    index, doc_ids = load_index(model_name, domain)
    query_vecs = np.asarray(query_vecs, dtype=np.float32)
    scores, rows = index.search(query_vecs, k)

    result_ids = []
    for row_list in rows:
        result_ids.append([doc_ids[r] for r in row_list])
    return result_ids, scores


_model_cache = {}
_query_cache = {}


def load_model(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(ENCODERS[model_name]["hf"])
    return _model_cache[model_name]


def encode_queries(model_name, texts, max_len=None, prefix=True, batch_size=32):
    cfg = ENCODERS[model_name]
    max_len = max_len or cfg["max_len"]
    if prefix:
        texts = [cfg["query_prefix"] + t for t in texts]

    todo = [t for t in dict.fromkeys(texts) if (model_name, max_len, t) not in _query_cache]
    if todo:
        model = load_model(model_name)
        model.max_seq_length = max_len
        vectors = model.encode(todo, batch_size=batch_size, normalize_embeddings=True)
        _query_cache.update(((model_name, max_len, t), v) for t, v in zip(todo, vectors))
    return np.stack([_query_cache[model_name, max_len, t] for t in texts])


def system_name(model_name, variant, recent_first=False, window=None, max_len=None, prefix=True, rebuild=False):
    cfg = ENCODERS[model_name]
    name = f"{model_name}-{variant}"
    if rebuild and variant == "history" and not recent_first and window is None:
        name += "-rebuilt"
    if window is not None:
        name += f"-last{window}"
    if recent_first:
        name += "-newest"
    if max_len and max_len != cfg["max_len"]:
        name += f"-len{max_len}"
    if not prefix and cfg["query_prefix"]:
        name += "-noprefix"
    return name


def run_dense(model_name, domain, turns, variant, recent_first=False, window=None, max_len=None, prefix=True, rebuild=False, k=100):
    queries = [build_query(t, variant, recent_first, window, rebuild) for t in turns]
    result_ids, result_scores = search(model_name, domain, encode_queries(model_name, queries, max_len, prefix), k)
    return {t["qid"]: dict(zip(ids, map(float, scores)))
        for t, ids, scores in zip(turns, result_ids, result_scores)}
