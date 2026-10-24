import json
import faiss

import numpy as np
from sentence_transformers import SentenceTransformer

from semeval_reteco.config import EMB_DIR
from semeval_reteco.data import load_corpus

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