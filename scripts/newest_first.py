"""Does putting the newest turns first help a dense model that truncates long queries?

    uv run python scripts/newest_first.py

all-MiniLM-L6-v2 reads at most 256 tokens and silently drops the rest. A query with history
is "query + label + oldest turn ... newest turn", so the cut removes the NEWEST turns, the ones
the current question usually refers to. With recent_first the OLDEST turns are cut instead.
Compares history and q-only in both orders. Needs data/emb/minilm/ (see embeddings.encode_corpus).

Writes run files to results/runs/ (git-ignored) and the tables to results/newest_first/newest_first.md.
"""
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from semeval_reteco.config import DOMAINS, ROOT, SPLITS
from semeval_reteco.data import load_qrels, load_turns
from semeval_reteco.embeddings import ENCODERS, search
from semeval_reteco.evaluate import (
    BUCKETS,
    METRICS,
    bucket,
    evaluate,
    macro,
    topic_scores,
    write_run,
)
from semeval_reteco.queries import build_query

MODEL = "minilm"
OUT = ROOT / "results" / "newest_first" / "newest_first.md"
# name -> (variant, recent_first); history uses the dataset's gold answers (R4)
CONDITIONS = {
    "history (gold answers)": ("history", False),
    "history (gold answers), newest first": ("history", True),
    "q-only": ("qonly", False),
    "q-only, newest first": ("qonly", True),
}
PAIRS = [("history (gold answers), newest first", "history (gold answers)"),
    ("q-only, newest first", "q-only")]
RESAMPLES = 10_000


def conv_id(qid):
    return qid.rsplit("_turn_", 1)[0]


def run_dense(model, domain, turns, variant, recent_first):
    texts = [build_query(t, variant, recent_first) for t in turns]
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
    result_ids, result_scores = search(MODEL, domain, vectors)
    return {t["qid"]: dict(zip(ids, map(float, scores)))
        for t, ids, scores in zip(turns, result_ids, result_scores)}


def paired_bootstrap(deltas, rng):
    """deltas: {domain: {qid: ndcg_a - ndcg_b}} -> (mean, low, high) of the macro delta.

    Resamples whole conversations within each domain (turns of one conversation are not independent).
    """
    boot = np.zeros(RESAMPLES)
    for per_qid in deltas.values():
        by_conv = {}
        for qid, d in per_qid.items():
            by_conv.setdefault(conv_id(qid), []).append(d)
        conv_means = np.array([np.mean(v) for v in by_conv.values()])
        boot += conv_means[rng.integers(len(conv_means), size=(RESAMPLES, len(conv_means)))].mean(axis=1)
    boot /= len(deltas)
    point = np.mean([np.mean(list(d.values())) for d in deltas.values()])
    return point, *np.percentile(boot, [2.5, 97.5])


def main():
    model = SentenceTransformer(ENCODERS[MODEL]["hf"])
    model.max_seq_length = ENCODERS[MODEL]["max_len"]

    results = {(c, s): {} for c in CONDITIONS for s in SPLITS}  # -> {domain: evaluate()}
    per_topic = {(c, s): {} for c in CONDITIONS for s in SPLITS}  # -> {domain: {qid: ndcg}}
    n_tokens = {s: {} for s in SPLITS}  # split -> depth -> token counts of the history queries
    for domain in DOMAINS:
        start = time.time()
        for split in SPLITS:
            turns, qrels = load_turns(domain, split), load_qrels(domain, split)
            for t in turns:
                text = build_query(t, "history")
                n_tokens[split].setdefault(bucket(t["qid"]), []).append(len(model.tokenizer(text)["input_ids"]))
            for name, (variant, recent_first) in CONDITIONS.items():
                run = run_dense(model, domain, turns, variant, recent_first)
                tag = f"dense-{MODEL}-{variant}" + ("-newest" if recent_first else "")
                write_run(ROOT / "results" / "runs" / domain / f"{split}_{tag}.run", run, tag)
                results[name, split][domain] = evaluate(run, qrels)
                per_topic[name, split][domain] = {q: m["ndcg_cut_10"] for q, m in topic_scores(run, qrels).items()}
        print(f"{domain}: {time.time() - start:.0f}s", flush=True)

    lines = [f"# Newest-first history order, dense {MODEL} (max {ENCODERS[MODEL]['max_len']} tokens), 11 domains", "",
        ("`history` feeds the dataset's gold answers to the retriever (R4); `q-only` does not. "
         "Default order is oldest turn first, so truncation cuts the newest turns."), "",
        "## Macro average over domains", "",
        "| split | condition | " + " | ".join(METRICS.values()) + " |", "|---|---|" + "---|" * len(METRICS)]
    for split in ("dev", "train"):
        for name in CONDITIONS:
            m = macro(results[name, split])
            lines.append(f"| {split} | {name} | " + " | ".join(f"{m[n]:.4f}" for n in METRICS.values()) + " |")

    for split in ("dev", "train"):
        lines += ["", f"## nDCG@10 by turn depth, {split} (macro over domains)", "",
            "| condition | " + " | ".join(BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
        for name in CONDITIONS:
            row = [macro(results[name, split], b)["nDCG@10"] for b in BUCKETS]
            lines.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in row) + " |")

    rng = np.random.default_rng(0)
    lines += ["", f"## Paired difference in nDCG@10 (A − B), {RESAMPLES:,} conversation-level bootstrap resamples, 95% CI", "",
        "No setting was fitted on dev, so pooling train and dev is not optimistic.", "",
        "| A | B | split | Δ | 95% CI |", "|---|---|---|---|---|"]
    for a, b in PAIRS:
        for label, splits in (("dev", ("dev",)), ("train+dev", SPLITS)):
            deltas = {d: {q: per_topic[a, s][d][q] - per_topic[b, s][d][q] for s in splits for q in per_topic[a, s][d]}
                for d in DOMAINS}
            point, low, high = paired_bootstrap(deltas, rng)
            lines.append(f"| {a} | {b} | {label} | {point:+.4f} | [{low:+.4f}, {high:+.4f}] |")

    lines += ["", f"## Query length: share of history queries over {ENCODERS[MODEL]['max_len']} tokens (train + dev)", "",
        "| depth | queries | > limit | median tokens |", "|---|---|---|---|"]
    for b in BUCKETS[1:]:
        counts = np.array(n_tokens["dev"].get(b, []) + n_tokens["train"].get(b, []))
        lines.append(f"| {b} | {len(counts)} | {np.mean(counts > ENCODERS[MODEL]['max_len']):.1%} | {np.median(counts):.0f} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
