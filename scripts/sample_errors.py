"""Picks 30 dev turns (6 from each of 5 groups) for the error analysis.

Run: uv run python scripts/sample_errors.py   (after the BM25 and dense runs)

Writes results/error_analysis/sample.csv. The labels for these turns are in results/error_analysis/labels.csv.
"""
import csv

import numpy as np

from semeval_reteco.config import DOMAINS, ROOT
from semeval_reteco.data import load_corpus, load_qrels, load_turns
from semeval_reteco.evaluate import bucket, load_metrics, read_run

OUT = ROOT / "results" / "error_analysis" / "sample.csv"
RUNS_DIR = ROOT / "results" / "runs"
PER_STRATUM = 6
SNIPPET = 300
SYSTEMS = [f"{r}-{v}" for r in ("bm25", "minilm", "bge-base") for v in ("current", "qonly", "history")]


def stratum(s, turn):
    if all(x == 0 for x in s.values()):
        return "everything fails"
    if s["bge-base-history"] - s["bm25-history"] >= 0.5:
        return "dense >> BM25"
    if s["bm25-history"] - s["bge-base-history"] >= 0.5:
        return "BM25 >> dense"
    if s["bge-base-history"] - s["bge-base-current"] >= 0.5:
        return "history fixes it"
    if turn >= 2 and s["bge-base-history"] <= s["bge-base-current"] < 0.5:
        return "history no help"
    return None


def snippet(text):
    return " ".join(text.split())[:SNIPPET]


def main():
    metrics = {s: load_metrics(s) for s in SYSTEMS}
    candidates = {}
    for domain in DOMAINS:
        for turn in load_turns(domain, "dev"):
            scores = {s: metrics[s][domain, "dev"][turn["qid"]]["ndcg_cut_10"] for s in SYSTEMS}
            name = stratum(scores, turn["turn"])
            if name:
                candidates.setdefault(name, []).append((domain, turn, scores))

    rng = np.random.default_rng(0)
    rows = []
    for name in ("everything fails", "dense >> BM25", "BM25 >> dense", "history fixes it", "history no help"):
        pool = candidates.get(name, [])
        picked = sorted(rng.choice(len(pool), size=min(PER_STRATUM, len(pool)), replace=False))
        print(f"{name}: {len(pool)} dev turns, sampled {len(picked)}")
        rows += [(name, *pool[i]) for i in picked]

    texts, runs, qrels = {}, {}, {}
    for domain in {d for _, d, _, _ in rows}:
        doc_ids, docs = load_corpus(domain)
        texts[domain] = dict(zip(doc_ids, docs))
        qrels[domain] = load_qrels(domain, "dev")
        for system in ("bm25-history", "bge-base-history"):
            runs[domain, system] = read_run(RUNS_DIR / domain / f"dev_{system}.run")

    def top3(domain, system, qid):
        ranked = list(runs[domain, system][qid])[:3]
        return "\n\n".join(f"[{'GOLD' if d in qrels[domain][qid] else 'x'}] {snippet(texts[domain][d])}"
            for d in ranked)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stratum", "domain", "qid", "depth", *(f"ndcg {s}" for s in SYSTEMS), "query",
            "earlier questions", "bge-base-history top 3", "bm25-history top 3", "gold documents"])
        for name, domain, turn, scores in rows:
            qid = turn["qid"]
            gold = [d for d, rel in qrels[domain][qid].items() if rel > 0]
            writer.writerow([name, domain, qid, bucket(qid), *(f"{scores[s]:.3f}" for s in SYSTEMS), turn["query"],
                "\n".join(f"{i}. {q}" for i, q in enumerate(turn["prev_queries"], 1)),
                top3(domain, "bge-base-history", qid), top3(domain, "bm25-history", qid),
                "\n\n".join(snippet(texts[domain][d]) for d in gold if d in texts[domain])])
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} turns)")


if __name__ == "__main__":
    main()
