"""Tries different ways to build the query for a dense model, picks the best on train and reports it on dev.

Run: uv run python scripts/dense_grid.py --model minilm   (or --model bge-base)

Writes results/dense_grid/<model>.md, and results/metrics/<model>-<variant>-tuned.csv if a non-default setting wins.
"""
import argparse
import time

from semeval_reteco.config import DOMAINS, ROOT
from semeval_reteco.data import load_qrels, load_turns
from semeval_reteco.embeddings import ENCODERS, run_dense, system_name
from semeval_reteco.evaluate import (
    BUCKETS,
    METRICS,
    METRICS_DIR,
    macro,
    save_metrics,
    summarize,
    topic_scores,
)

REGIMES = ("current", "qonly", "history")
SHAPES = [(None, False, False), (None, False, True), (None, True, False), (2, False, False), (2, True, False),
    (1, False, False)]


def grid(model, regime):
    cfg = ENCODERS[model]
    lengths = (cfg["max_len"], 512) if cfg["max_len"] < 512 else (cfg["max_len"],)
    prefixes = (True, False) if cfg["query_prefix"] else (True,)
    if regime == "current":
        shapes = SHAPES[:1]
    elif regime == "qonly":
        shapes = [s for s in SHAPES if not s[2]]
    else:
        shapes = SHAPES
    return [{"variant": regime, "window": w, "recent_first": r, "rebuild": rb, "max_len": n, "prefix": p}
        for w, r, rb in shapes for n in lengths for p in prefixes]


def evaluate_config(model, config, split):
    per_topic = {}
    for domain in DOMAINS:
        turns, qrels = load_turns(domain, split), load_qrels(domain, split)
        per_topic[domain, split] = topic_scores(run_dense(model, domain, turns, **config), qrels)
    return per_topic


def macro_of(per_topic, split, bucket="all"):
    return macro({d: summarize(per_topic[d, split]) for d in DOMAINS}, bucket)


def describe(config):
    window = "all" if config["window"] is None else f"last {config['window']}"
    if config["variant"] == "history" and config["window"] is None and not config["recent_first"]:
        window += " (rebuilt)" if config["rebuild"] else " (dataset field)"
    order = "newest first" if config["recent_first"] else "oldest first"
    return f"{window}, {order}, {config['max_len']} tokens" + ("" if config["prefix"] else ", no prefix")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ENCODERS))
    model = parser.parse_args().model

    intro = ("Selected by **train** macro nDCG@10 over 11 domains; only the selected setting and the default are run "
        "on dev (rule R1). `history` feeds the dataset's gold answers to the retriever (R4). The default `history` "
        "query is the dataset's own `conversation_history` field (the official query), which for long conversations "
        "keeps only the last 2 turns; every other history setting is rebuilt from all earlier turns, so compare "
        "newest first with \"all (rebuilt), oldest first\". Read the table as an ablation: what each query-side "
        "choice is worth.")
    lines = [f"# Encoder grid search, {model} ({ENCODERS[model]['hf']})", "", intro, ""]
    for regime in REGIMES:
        start = time.time()
        configs = grid(model, regime)
        train = [evaluate_config(model, c, "train") for c in configs]
        scores = [macro_of(t, "train") for t in train]
        # differences below 0.0001 are noise, so they count as a tie and the default wins
        best = max(range(len(configs)), key=lambda i: round(scores[i]["nDCG@10"], 4))

        header = "| setting | system | " + " | ".join(f"train {m}" for m in METRICS.values()) + " |"
        lines += [f"## {regime}", "", header, "|---|---|" + "---|" * len(METRICS)]
        for i in sorted(range(len(configs)), key=lambda i: -scores[i]["nDCG@10"]):
            mark = " **(selected)**" if i == best else ""
            mark += " (default)" if i == 0 else ""
            lines.append(f"| {describe(configs[i])}{mark} | `{system_name(model, **configs[i])}` | "
                + " | ".join(f"{scores[i][m]:.4f}" for m in METRICS.values()) + " |")

        dev = {i: evaluate_config(model, configs[i], "dev") for i in {0, best}}
        if best:
            save_metrics(f"{model}-{regime}-tuned", train[best] | dev[best])
        else:
            (METRICS_DIR / f"{model}-{regime}-tuned.csv").unlink(missing_ok=True)
        lines += ["", "Dev, selected vs default (macro nDCG@10):", "",
            "| setting | " + " | ".join(BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
        for i, label in ((best, "selected"), (0, "default")):
            row = [macro_of(dev[i], "dev", b)["nDCG@10"] for b in BUCKETS]
            lines.append(f"| {label}: {describe(configs[i])} | " + " | ".join(f"{x:.4f}" for x in row) + " |")
        lines.append("")
        print(f"{model} {regime}: {len(configs)} settings, {time.time() - start:.0f}s, "
            f"selected {system_name(model, **configs[best])}", flush=True)

    out = ROOT / "results" / "dense_grid" / f"{model}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
