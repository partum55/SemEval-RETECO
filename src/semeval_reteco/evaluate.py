import csv
from collections import defaultdict
from pathlib import Path

import pytrec_eval

from semeval_reteco.config import ROOT

METRICS_DIR = ROOT / "results" / "metrics"

METRICS = {"ndcg_cut_10": "nDCG@10", "map_cut_10": "MAP@10", "recall_10": "Recall@10", "recall_100": "Recall@100",
    "recip_rank": "MRR"}

BUCKETS = ("all", "T1", "T2", "T3", "T4", "T5+")


def bucket(qid):
    n = int(qid.rsplit("_turn_", 1)[1])
    return "T5+" if n >= 5 else f"T{n}"


def strictly_decreasing(docs):
    ranked = sorted(docs, key=lambda d: -docs[d])
    return {d: float(len(ranked) - i) for i, d in enumerate(ranked)}


def topic_scores(run, qrels):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels,
        {"ndcg_cut.10", "map_cut.10", "recall.10", "recall.100", "recip_rank"})
    # equal scores keep our order, pytrec_eval would sort them by doc id, which gives away which docs are relevant
    return evaluator.evaluate({qid: strictly_decreasing(run.get(qid, {})) for qid in qrels})


def evaluate(run, qrels):
    return summarize(topic_scores(run, qrels))


def summarize(per_topic):
    groups = defaultdict(list)
    for qid, scores in per_topic.items():
        groups["all"].append(scores)
        groups[bucket(qid)].append(scores)
    return {b: {**{name: sum(s[key] for s in group) / len(group) for key, name in METRICS.items()},
        "n": len(group)} for b, group in groups.items()}


def macro(results, bucket="all"):
    return {name: sum(r[bucket][name] for r in results.values()) / len(results)
        for name in METRICS.values()}


def write_run(path, run, tag):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, docs in run.items():
            f.writelines(f"{qid}\tQ0\t{doc_id}\t{rank}\t{score:.1f}\t{tag}\n"
                for rank, (doc_id, score) in enumerate(strictly_decreasing(docs).items(), 1))


def read_run(path):
    run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, _, score, _ = line.split()
            run.setdefault(qid, {})[doc_id] = float(score)
    return run


def save_metrics(system, per_topic):
    path = METRICS_DIR / f"{system}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "split", "qid", *METRICS])
        for (domain, split), scores in sorted(per_topic.items()):
            writer.writerows([domain, split, qid, *(f"{s[key]:.6f}" for key in METRICS)]
                for qid, s in scores.items())


def load_metrics(system):
    per_topic = defaultdict(dict)
    with open(METRICS_DIR / f"{system}.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            per_topic[row["domain"], row["split"]][row["qid"]] = {key: float(row[key]) for key in METRICS}
    return dict(per_topic)
