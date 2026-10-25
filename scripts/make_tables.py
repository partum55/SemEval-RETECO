"""Builds all comparison tables and figures from the per-turn scores in results/metrics/.

Run: uv run python scripts/make_tables.py   (after the BM25 and dense scripts)

Writes results/tables/*.md and results/figures/*.png, and updates the main table in README.md.
"""
import csv
import re
from collections import Counter
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from semeval_reteco.config import DOMAINS, ROOT, SPLITS
from semeval_reteco.data import load_turns
from semeval_reteco.embeddings import ENCODERS
from semeval_reteco.evaluate import (
    BUCKETS,
    METRICS,
    METRICS_DIR,
    bucket,
    load_metrics,
    macro,
    summarize,
)
from semeval_reteco.queries import build_query
from semeval_reteco.stats import RESAMPLES, paired_bootstrap

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
README = ROOT / "README.md"

RETRIEVERS = {"bm25": "BM25", "minilm": "MiniLM", "bge-base": "bge-base"}
REGIMES = {"current": "current", "qonly": "q-only", "history": "history (gold answers)"}
COLORS = {"bm25": "#2a78d6", "minilm": "#eb6834", "bge-base": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

COMPARISONS = [
    ("bge-base-current", "bm25-current", "unadapted dense vs BM25, current turn"),
    ("bge-base-qonly", "bm25-qonly", "unadapted dense vs BM25, q-only"),
    ("bge-base-history", "bm25-history", "unadapted dense vs BM25, history"),
    ("bge-base-qonly", "minilm-qonly", "bigger encoder, q-only"),
    ("bge-base-history", "minilm-history", "bigger encoder, history"),
    ("bge-base-qonly", "bge-base-current", "earlier questions help dense"),
    ("bge-base-history", "bge-base-qonly", "gold answers on top of the questions"),
    ("minilm-qonly-tuned", "minilm-qonly", "grid-selected vs default"),
    ("minilm-history-tuned", "minilm-history", "grid-selected vs default"),
    ("bge-base-qonly-tuned", "bge-base-qonly", "grid-selected vs default"),
    ("bge-base-history-tuned", "bge-base-history", "grid-selected vs default"),
    ("bm25-qonly-tuned", "bm25-qonly", "grid-selected k1, b vs official"),
    ("bm25-history-tuned", "bm25-history", "grid-selected k1, b vs official"),
]

CONTEXT = {
    "minilm": [("minilm-history", "256 tokens, dataset field, oldest first (default)"),
        ("minilm-history-rebuilt", "256 tokens, all turns, oldest first"),
        ("minilm-history-newest", "256 tokens, all turns, newest first"),
        ("minilm-history-len512", "512 tokens, dataset field, oldest first")],
    "bge-base": [("bge-base-history", "512 tokens, dataset field, oldest first (default)"),
        ("bge-base-history-rebuilt", "512 tokens, all turns, oldest first"),
        ("bge-base-history-newest", "512 tokens, all turns, newest first")],
}

CONTEXT_PAIRS = [
    ("minilm-history-newest", "minilm-history-rebuilt", "order: newest turns first"),
    ("bge-base-history-newest", "bge-base-history-rebuilt", "order: newest turns first"),
    ("minilm-history-len512", "minilm-history", "length: 512 tokens instead of 256"),
    ("minilm-history-rebuilt", "minilm-history", "content: all turns instead of the dataset field"),
    ("bge-base-history-rebuilt", "bge-base-history", "content: all turns instead of the dataset field"),
]

_cache, missing = {}, []


def metrics(system):
    if system not in _cache:
        found = (METRICS_DIR / f"{system}.csv").exists()
        _cache[system] = load_metrics(system) if found else None
        if not found and not system.endswith("-tuned"):
            missing.append(system)
    return _cache[system]


def score(per_topic, splits, bucket_name="all", metric="nDCG@10"):
    per_domain = {d: summarize({f"{s}/{q}": m for s in splits for q, m in per_topic[d, s].items()})
        for d in DOMAINS}
    return macro(per_domain, bucket_name)[metric]


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def systems_for(regime, tuned=True):
    rows = []
    for retriever in RETRIEVERS:
        rows.append((f"{retriever}-{regime}", retriever, RETRIEVERS[retriever]))
        if tuned:
            rows.append((f"{retriever}-{regime}-tuned", retriever, f"{RETRIEVERS[retriever]}, grid-selected"))
    return [r for r in rows if metrics(r[0])]


def main_table():
    rows = []
    for regime, regime_label in REGIMES.items():
        for system, _, label in systems_for(regime):
            m = metrics(system)
            rows.append([regime_label, label, *(f"{score(m, ('dev',), metric=n):.4f}" for n in METRICS.values()),
                f"{score(m, ('train',)):.4f}"])
    return table(["query", "retriever", *(f"dev {n}" for n in METRICS.values()), "train nDCG@10"], rows)


def by_depth_table(split):
    counts = {b: 0 for b in BUCKETS}
    for domain in DOMAINS:
        for turn in load_turns(domain, split):
            counts["all"] += 1
            counts[bucket(turn["qid"])] += 1
    rows = [[REGIMES[regime], label, *(f"{score(metrics(s), (split,), b):.4f}" for b in BUCKETS)]
        for regime in REGIMES for s, _, label in systems_for(regime)]
    return table(["query", "retriever", *(f"{b} (n={counts[b]})" for b in BUCKETS)], rows)


def by_domain_table():
    columns = [(s, f"{label}, {REGIMES[regime]}") for regime in REGIMES for s, _, label in systems_for(regime, False)]
    rows = []
    for domain in DOMAINS:
        n = len(metrics(columns[0][0])[domain, "dev"])
        rows.append([domain, str(n), *(f"{np.mean([t['ndcg_cut_10'] for t in metrics(s)[domain, 'dev'].values()]):.4f}"
            for s, _ in columns)])
    return table(["domain", "dev topics", *(c for _, c in columns)], rows)


def comparisons_table():
    rows = []
    for a, b, question in COMPARISONS:
        if metrics(a) and metrics(b):
            point, low, high = paired_bootstrap(metrics(a), metrics(b), ("dev",))
            verdict = "A better" if low > 0 else "B better" if high < 0 else "no clear difference"
            rows.append([question, f"`{a}`", f"`{b}`", f"{point:+.4f}", f"[{low:+.4f}, {high:+.4f}]", verdict])
        elif a.endswith("-tuned") and metrics(b):
            rows.append([question, f"`{a}`", f"`{b}`", "—", "—", "the grid kept the default"])
    return table(["question", "A", "B", "Δ nDCG@10 (A − B)", "95% CI", "verdict"], rows)


def token_lengths(turns):
    lengths = {}
    for model, cfg in ENCODERS.items():
        tokenizer = AutoTokenizer.from_pretrained(cfg["hf"])
        texts = [cfg["query_prefix"] + build_query(t, "history") for t in turns]
        for t, ids in zip(turns, tokenizer(texts, verbose=False)["input_ids"]):
            lengths.setdefault((model, bucket(t["qid"])), []).append(len(ids))
    return lengths


def context_length_lines():
    turns = [t for d in DOMAINS for s in SPLITS for t in load_turns(d, s)]
    lengths = token_lengths(turns)
    deepest = [t for t in turns if bucket(t["qid"]) == "T5+"]
    short_field = np.mean([build_query(t, "history") != build_query(t, "history", rebuild=True) for t in deepest])
    intro = ("Dense encoders read a fixed number of tokens and cut the rest from the END of the text. The default "
        "`history` query is \"question + oldest turn ... newest turn\", so a long query loses its newest turns. "
        "*Newest first* reverses the earlier turns, so the oldest ones are cut instead; *512 tokens* lets MiniLM "
        "read twice as much (its documents stay encoded at 256). The default query is the dataset's "
        f"`conversation_history` field, which keeps only the last 2 turns for {short_field:.0%} of T5+ turns (none "
        "before T5); newest first is rebuilt from all turns, so its effect is measured against *all turns, oldest "
        "first*, and each pair below changes one thing. `history` contains the dataset's gold answers (R4).")
    lines = ["# Context length: what truncating long history queries costs", "", intro, "",
        "## History queries over the token limit (train + dev)", ""]
    rows = []
    for b in BUCKETS[1:]:
        mini, bge = np.array(lengths["minilm", b]), np.array(lengths["bge-base", b])
        rows.append([b, str(len(mini)), f"{np.median(mini):.0f}", f"{np.mean(mini > 256):.1%}",
            f"{np.mean(mini > 512):.1%}", f"{np.median(bge):.0f}", f"{np.mean(bge > 512):.1%}"])
    lines += table(["depth", "turns", "MiniLM median tokens", "MiniLM > 256", "MiniLM > 512",
        "bge median tokens (with prefix)", "bge > 512"], rows)

    for splits, title in ((("dev",), "dev"), (("train",), "train"), (SPLITS, "train + dev")):
        lines += ["", f"## nDCG@10 by turn depth, {title} (macro over domains)", ""]
        rows = [[RETRIEVERS[model], label, *(f"{score(metrics(s), splits, b):.4f}" for b in BUCKETS)]
            for model, conditions in CONTEXT.items() for s, label in conditions if metrics(s)]
        lines += table(["model", "query", *BUCKETS], rows)

    lines += ["", f"## Paired difference in nDCG@10, train + dev, {RESAMPLES:,} conversation bootstrap resamples", "",
        "None of these settings was fitted on any split, so pooling train and dev is not optimistic.", ""]
    rows = []
    for a, b, change in CONTEXT_PAIRS:
        if metrics(a) and metrics(b):
            deep = {key: {q: m for q, m in topics.items() if bucket(q) in ("T4", "T5+")}
                for key, topics in metrics(a).items()}
            deep_b = {key: {q: metrics(b)[key][q] for q in topics} for key, topics in deep.items()}
            for label, (x, y) in (("all turns", (metrics(a), metrics(b))), ("T4 + T5+", (deep, deep_b))):
                point, low, high = paired_bootstrap(x, y, SPLITS)
                rows.append([change, f"`{a}`", f"`{b}`", label, f"{point:+.4f}", f"[{low:+.4f}, {high:+.4f}]"])
    lines += table(["what changes", "A", "B", "turns", "Δ (A − B)", "95% CI"], rows)
    return lines


def error_types_lines():
    folder = ROOT / "results" / "error_analysis"
    if not (folder / "labels.csv").exists():
        return None
    with open(folder / "sample.csv", encoding="utf-8", newline="") as f:
        strata = {(r["domain"], r["qid"]): r["stratum"] for r in csv.DictReader(f)}
    with open(folder / "labels.csv", encoding="utf-8", newline="") as f:
        labels = [(strata[r["domain"], r["qid"]], r["label"]) for r in csv.DictReader(f)]
    columns = list(dict.fromkeys(s for s, _ in labels))
    counts = Counter(labels)
    rows = [[label, *(str(counts[s, label] or "") for s in columns), str(sum(counts[s, label] for s in columns))]
        for label, _ in Counter(label for _, label in labels).most_common()]
    intro = (f"{len(labels)} dev turns from `scripts/sample_errors.py`, labelled by hand in "
        "[`labels.csv`](../error_analysis/labels.csv) (one main cause per turn; for 'dense >> BM25' and "
        "'BM25 >> dense' the label explains why the weaker retriever failed).")
    return ["# First error analysis: error types", "", intro, "", *table(["error type", *columns, "total"], rows)]


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_by_depth():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (regime, regime_label) in zip(axes, REGIMES.items()):
        style(ax)
        ends = []
        for system, retriever, label in systems_for(regime, False):
            ys = [score(metrics(system), ("dev",), b) for b in BUCKETS[1:]]
            ax.plot(BUCKETS[1:], ys, color=COLORS[retriever], linewidth=2, marker="o", markersize=7, label=label)
            ends.append([ys[-1], label])
        ends.sort()
        for lower, upper in pairwise(ends):
            upper[0] = max(upper[0], lower[0] + 0.02)
        for y, label in ends:
            ax.annotate(label, (len(BUCKETS) - 2, y), xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=9, color=INK)
        ax.set_title(regime_label, color=INK, fontsize=11)
        ax.set_xlabel("turn depth", color=MUTED)
        ax.set_xlim(-0.3, len(BUCKETS) - 1.2)
    axes[0].set_ylabel("dev nDCG@10 (macro over 11 domains)", color=MUTED)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "ndcg_by_depth.png", dpi=150)


def plot_context_length():
    pairs = [("minilm", "minilm-history-newest", "minilm-history-rebuilt", "MiniLM: newest turns first", ""),
        ("minilm", "minilm-history-len512", "minilm-history", "MiniLM: read 512 tokens instead of 256", "//"),
        ("bge-base", "bge-base-history-newest", "bge-base-history-rebuilt", "bge-base: newest turns first", "")]
    pairs = [p for p in pairs if metrics(p[1]) and metrics(p[2])]
    if not pairs:
        return
    depths = ("all", "T3", "T4", "T5+")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style(ax)
    x = np.arange(len(depths))
    width = 0.8 / len(pairs)
    for i, (model, a, b, label, hatch) in enumerate(pairs):
        deltas = [score(metrics(a), SPLITS, d) - score(metrics(b), SPLITS, d) for d in depths]
        ax.bar(x + (i - (len(pairs) - 1) / 2) * width, deltas, width - 0.03, color=COLORS[model], hatch=hatch,
            edgecolor="white", linewidth=0, label=label)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(x, ["all turns", "T3", "T4", "T5+"])
    ax.set_xlabel("turn depth (T1 and T2 are never truncated)", color=MUTED)
    ax.set_ylabel("Δ nDCG@10 vs the same query without the change", color=MUTED)
    ax.set_title("History query, train + dev (macro over 11 domains)", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "context_length.png", dpi=150)


def write(name, lines):
    path = TABLES / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    note = ("Macro average over the 11 domains (each domain counts equally, like the leaderboard). "
        "`history` feeds the dataset's gold answers to the retriever (R4); `current` and `q-only` do not. "
        "*Grid-selected* rows appear only where a grid search on train (R1) picked something other than the default "
        "(results/dense_grid/, results/bm25/grid.md); the official BM25 (k1=0.9, b=0.4) stays the baseline.")

    main_lines = main_table()
    write("main.md", ["# Main results", "", note, "", *main_lines])
    write("by_depth.md", ["# nDCG@10 by turn depth", "", note, "", "## dev", "", *by_depth_table("dev"),
        "", "## train", "", *by_depth_table("train")])
    write("by_domain.md", ["# dev nDCG@10 per domain (default settings)", "", note, "", *by_domain_table()])
    method = (f"Paired difference in macro nDCG@10, {RESAMPLES:,} bootstrap resamples of whole conversations within "
        "each domain, 95% percentile interval. Each comparison was fixed before the results it compares were seen.")
    write("comparisons.md", ["# Pre-declared comparisons, dev", "", method, "", *comparisons_table()])
    write("context_length.md", context_length_lines())
    if error_lines := error_types_lines():
        write("error_types.md", error_lines)
    plot_by_depth()
    plot_context_length()

    readme = README.read_text(encoding="utf-8")
    block = "<!-- main-table -->\n" + "\n".join(main_lines) + "\n<!-- /main-table -->"
    README.write_text(re.sub(r"<!-- main-table -->.*?<!-- /main-table -->", lambda _: block, readme, flags=re.DOTALL),
        encoding="utf-8")

    if missing:
        print("skipped (no results/metrics/<system>.csv yet; see README, Reproduce):", ", ".join(sorted(set(missing))))


if __name__ == "__main__":
    main()
