"""Runs dense retrieval (MiniLM or bge-base) on all 11 domains, train and dev.

Run: uv run python scripts/run_dense.py --model minilm --variants current qonly history

Writes run files to results/runs/ and scores per turn to results/metrics/<system>.csv.
"""
import argparse
import time

from semeval_reteco.config import DOMAINS, ROOT, SPLITS
from semeval_reteco.data import load_qrels, load_turns
from semeval_reteco.embeddings import ENCODERS, run_dense, system_name
from semeval_reteco.evaluate import (
    BUCKETS,
    METRICS,
    macro,
    save_metrics,
    summarize,
    topic_scores,
    write_run,
)
from semeval_reteco.queries import VARIANTS

RUNS_DIR = ROOT / "results" / "runs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ENCODERS))
    parser.add_argument("--variants", nargs="+", default=["current", "qonly", "history"], choices=VARIANTS)
    parser.add_argument("--recent-first", action="store_true", help="newest earlier turn first")
    parser.add_argument("--window", type=int, help="keep only the last N earlier turns")
    parser.add_argument("--query-max-len", type=int, help="query token limit (default: the model's)")
    parser.add_argument("--no-prefix", action="store_true", help="drop the bge query instruction")
    parser.add_argument("--rebuild", action="store_true",
        help="history from all earlier turns instead of the dataset field (implied by --recent-first / --window)")
    args = parser.parse_args()

    for variant in args.variants:
        config = {"variant": variant, "recent_first": args.recent_first, "window": args.window,
            "max_len": args.query_max_len, "prefix": not args.no_prefix, "rebuild": args.rebuild}
        system = system_name(args.model, **config)
        start = time.time()
        per_topic = {}
        for domain in DOMAINS:
            for split in SPLITS:
                turns, qrels = load_turns(domain, split), load_qrels(domain, split)
                run = run_dense(args.model, domain, turns, **config)
                write_run(RUNS_DIR / domain / f"{split}_{system}.run", run, system)
                per_topic[domain, split] = topic_scores(run, qrels)
        save_metrics(system, per_topic)

        print(f"\n{system} ({time.time() - start:.0f}s)")
        print("split  " + "  ".join(f"{m:>9}" for m in METRICS.values()) + "  nDCG@10 " + " ".join(BUCKETS[1:]))
        for split in SPLITS:
            results = {d: summarize(per_topic[d, split]) for d in DOMAINS}
            m = macro(results)
            by_depth = " ".join(f"{macro(results, b)['nDCG@10']:.3f}" for b in BUCKETS[1:])
            print(f"{split:<6} " + "  ".join(f"{m[n]:>9.4f}" for n in METRICS.values()) + f"  {by_depth}")


if __name__ == "__main__":
    main()
