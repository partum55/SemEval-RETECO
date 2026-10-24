from collections import defaultdict
from pathlib import Path

import pytrec_eval

# MRR is pytrec's recip_rank over the whole run (depth 100), like the official scorer.
METRICS = {"ndcg_cut_10": "nDCG@10", "map_cut_10": "MAP@10", "recall_10": "Recall@10", "recip_rank": "MRR"}
BUCKETS = ("all", "T1", "T2", "T3", "T4", "T5+")


def bucket(qid):
    n = int(qid.rsplit("_turn_", 1)[1])
    return "T5+" if n >= 5 else f"T{n}"


def strictly_decreasing(docs):
    """{doc_id: score} -> same docs with scores that never tie, in (score desc, given order) order."""
    ranked = sorted(docs, key=lambda d: -docs[d])  # stable
    return {d: float(len(ranked) - i) for i, d in enumerate(ranked)}


def evaluate(run, qrels):
    """run, qrels: {qid: {doc_id: score or rel}} for ONE domain and split.

    -> {bucket: {"nDCG@10": mean, ..., "n": topics}}. Every topic in qrels is scored;
    one missing from the run counts as 0. Ranking = score descending, ties keep the run's own
    order (pytrec alone would break ties by doc id, which leaks relevance).
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "map_cut.10", "recall.10", "recip_rank"})
    per_topic = evaluator.evaluate({qid: strictly_decreasing(run.get(qid, {})) for qid in qrels})

    groups = defaultdict(list)
    for qid, scores in per_topic.items():
        groups["all"].append(scores)
        groups[bucket(qid)].append(scores)
    return {b: {**{name: sum(s[key] for s in group) / len(group) for key, name in METRICS.items()},
        "n": len(group)} for b, group in groups.items()}


def macro(results, bucket="all"):
    """results: {domain: evaluate(...)} -> {metric name: mean over domains of the per-domain mean}.

    Macro over domains is the leaderboard number: a small domain counts as much as a big one.
    """
    return {name: sum(r[bucket][name] for r in results.values()) / len(results)
        for name in METRICS.values()}


def write_run(path, run, tag):
    """TREC run file: qid Q0 doc_id rank score tag.

    Scores are rewritten as strictly decreasing numbers, so scorers that sort by score
    (pytrec_eval) and scorers that follow the rank column give the same result.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in run.items():
            f.writelines(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.1f}\t{tag}\n"
                for rank, (doc_id, score) in enumerate(strictly_decreasing(docs).items(), 1))
