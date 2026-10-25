"""Tries 25 BM25 settings (k1 x b), picks the best one on train and compares it with the official one on dev.

Run: uv run python scripts/bm25_grid.py   (about 1 hour on a CPU)

Writes results/bm25/grid.md, and results/metrics/bm25-<variant>-tuned.csv if a non-official setting wins.
"""
import time

from semeval_reteco.bm25 import K1, B, Bm25, analyze_corpus, run_bm25
from semeval_reteco.config import DOMAINS, ROOT
from semeval_reteco.data import load_qrels, load_turns
from semeval_reteco.evaluate import BUCKETS, METRICS_DIR, macro, save_metrics, summarize, topic_scores

K1S = (0.6, 0.9, 1.2, 1.5, 2.0)
BS = (0.3, 0.4, 0.55, 0.75, 0.9)
PAIRS = [(K1, B)] + [(k1, b) for k1 in K1S for b in BS if (k1, b) != (K1, B)]
VARIANTS = ("current", "qonly", "history")
NOTE = {"current": "current", "qonly": "q-only", "history": "history (gold answers)"}
OUT = ROOT / "results" / "bm25" / "grid.md"


def macro_of(per_topic, split, bucket="all"):
    return macro({d: summarize(per_topic[d, split]) for d in DOMAINS}, bucket)["nDCG@10"]


def main():
    train = {(p, v): {} for p in PAIRS for v in VARIANTS}
    for domain in DOMAINS:
        start = time.time()
        analyzed = analyze_corpus(domain)
        turns, qrels = load_turns(domain, "train"), load_qrels(domain, "train")
        for pair in PAIRS:
            bm25 = Bm25(domain, *pair, analyzed=analyzed)
            for variant in VARIANTS:
                train[pair, variant][domain, "train"] = topic_scores(run_bm25(bm25, turns, variant), qrels)
        print(f"{domain}: {time.time() - start:.0f}s", flush=True)

    scores = {key: macro_of(per_topic, "train") for key, per_topic in train.items()}
    # differences below 0.0001 are noise, so they count as a tie and the official setting wins
    best = {v: max(PAIRS, key=lambda p: round(scores[p, v], 4)) for v in VARIANTS}

    dev = {(p, v): {} for v in VARIANTS for p in {(K1, B), best[v]}}
    for domain in DOMAINS:
        analyzed = analyze_corpus(domain)
        turns, qrels = load_turns(domain, "dev"), load_qrels(domain, "dev")
        for pair in {p for p, _ in dev}:
            bm25 = Bm25(domain, *pair, analyzed=analyzed)
            for variant in VARIANTS:
                if (pair, variant) in dev:
                    dev[pair, variant][domain, "dev"] = topic_scores(run_bm25(bm25, turns, variant), qrels)
    for variant in VARIANTS:
        if best[variant] != (K1, B):
            save_metrics(f"bm25-{variant}-tuned", train[best[variant], variant] | dev[best[variant], variant])
        else:
            (METRICS_DIR / f"bm25-{variant}-tuned.csv").unlink(missing_ok=True)

    lines = ["# BM25 grid search over k1 and b", "",
        f"Train macro nDCG@10 over 11 domains, rows k1, columns b. **Bold** = selected on train, *(official)* = "
        f"k1={K1}, b={B}. Only those two are run on dev (rule R1). `history` feeds the dataset's gold answers to "
        "the retriever (R4)."]
    for variant in VARIANTS:
        lines += ["", f"## {NOTE[variant]}", "", "| k1 \\ b | " + " | ".join(map(str, BS)) + " |",
            "|---|" + "---|" * len(BS)]
        for k1 in K1S:
            cells = []
            for b in BS:
                cell = f"{scores[(k1, b), variant]:.4f}"
                cell = f"**{cell}**" if (k1, b) == best[variant] else cell
                cells.append(cell + (" *(official)*" if (k1, b) == (K1, B) else ""))
            lines.append(f"| {k1} | " + " | ".join(cells) + " |")

    lines += ["", "## Dev: selected vs official (macro nDCG@10)", "",
        "| query | k1, b | " + " | ".join(BUCKETS) + " |", "|---|---|" + "---|" * len(BUCKETS)]
    for variant in VARIANTS:
        for pair, label in ((best[variant], "selected"), ((K1, B), "official")):
            row = [macro_of(dev[pair, variant], "dev", b) for b in BUCKETS]
            lines.append(f"| {NOTE[variant]} | {label}: {pair[0]}, {pair[1]} | "
                + " | ".join(f"{x:.4f}" for x in row) + " |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
