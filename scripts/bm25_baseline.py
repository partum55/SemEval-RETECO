"""BM25 baseline with and without history, all domains, train and dev.

    uv run python scripts/bm25_baseline.py

Writes run files to results/runs/<domain>/<split>_bm25-<variant>.run (git-ignored), the
tables to results/bm25/baseline.md and the plot to results/bm25/ndcg_by_depth.png.
"""
import time

import matplotlib.pyplot as plt

from semeval_reteco.bm25 import Bm25, run_bm25
from semeval_reteco.config import DOMAINS, ROOT, SPLITS
from semeval_reteco.data import load_qrels, load_turns
from semeval_reteco.evaluate import BUCKETS, METRICS, evaluate, macro, write_run
from semeval_reteco.queries import VARIANTS

RUNS_DIR = ROOT / "results" / "runs"
OUT = ROOT / "results" / "bm25" / "baseline.md"
PLOT = ROOT / "results" / "bm25" / "ndcg_by_depth.png"

# history and last2 use the dataset's gold answers (rule R4)
NOTE = {"current": "current", "qonly": "q-only", "last2": "last2 (gold answers)", "history": "history (gold answers)"}


def plot_by_depth(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, split in zip(axes, ("dev", "train")):
        for variant in VARIANTS:
            ys = [macro(results[variant, split], b)["nDCG@10"] for b in BUCKETS[1:]]
            ax.plot(BUCKETS[1:], ys, marker="o", linestyle="--" if variant in ("last2", "history") else "-",
                label=NOTE[variant])
        ax.set_title(f"BM25, {split} (macro over 11 domains)")
        ax.set_xlabel("turn depth")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("nDCG@10")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(VARIANTS))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT, dpi=150)


def main():
    results = {(v, s): {} for v in VARIANTS for s in SPLITS}  # (variant, split) -> {domain: evaluate()}
    for domain in DOMAINS:
        start = time.time()
        bm25 = Bm25(domain)
        for split in SPLITS:
            turns, qrels = load_turns(domain, split), load_qrels(domain, split)
            for variant in VARIANTS:
                run = run_bm25(bm25, turns, variant)
                write_run(RUNS_DIR / domain / f"{split}_bm25-{variant}.run", run, f"bm25-{variant}")
                results[variant, split][domain] = evaluate(run, qrels)
        print(f"{domain}: {time.time() - start:.0f}s", flush=True)

    lines = ["# BM25 baseline (official k1=0.9, b=0.4), 11 domains", "",
        "`last2` and `history` feed the dataset's gold answers to the retriever (R4); `current` and `q-only` do not.", "",
        "## Macro average over domains", "",
        "| split | variant | " + " | ".join(METRICS.values()) + " |", "|---|---|" + "---|" * len(METRICS)]
    for split in ("dev", "train"):
        for variant in VARIANTS:
            m = macro(results[variant, split])
            lines.append(f"| {split} | {NOTE[variant]} | " + " | ".join(f"{m[n]:.4f}" for n in METRICS.values()) + " |")

    for split in ("dev", "train"):
        lines += ["", f"## nDCG@10 per domain, {split}", "",
            "| domain | topics | " + " | ".join(NOTE[v] for v in VARIANTS) + " |", "|---|---|" + "---|" * len(VARIANTS)]
        for domain in DOMAINS:
            n = results["current", split][domain]["all"]["n"]
            scores = [results[v, split][domain]["all"]["nDCG@10"] for v in VARIANTS]
            lines.append(f"| {domain} | {n} | " + " | ".join(f"{x:.4f}" for x in scores) + " |")

    for split in ("dev", "train"):
        lines += ["", f"## nDCG@10 by turn depth, {split} (macro over domains)", "",
            "| variant | " + " | ".join(BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
        for variant in VARIANTS:
            row = [macro(results[variant, split], b)["nDCG@10"] for b in BUCKETS]
            lines.append(f"| {NOTE[variant]} | " + " | ".join(f"{x:.4f}" for x in row) + " |")

    plot_by_depth(results)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
