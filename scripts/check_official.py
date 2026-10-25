"""Checks our run files with the official format checker and scorer.

Run: uv run python scripts/check_official.py   (after the run files exist)

Downloads the official tools into data/starter_kit/ once and prints every file that fails.
"""
import json
import os
import subprocess
import sys
import urllib.request

from semeval_reteco.config import DATA_DIR, ROOT
from semeval_reteco.data import load_qrels
from semeval_reteco.evaluate import BUCKETS, evaluate, read_run

KIT_URL = "https://raw.githubusercontent.com/DataScienceUIBK/RETECO/main/starter_kit/"
KIT_DIR = ROOT / "data" / "starter_kit"
TOLERANCE = 1e-4


def fetch_starter_kit():
    KIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("scorer.py", "ir_metrics.py", "format_checker.py"):
        if not (KIT_DIR / name).exists():
            urllib.request.urlretrieve(KIT_URL + name, KIT_DIR / name)


def official(script, *args):
    return subprocess.run([sys.executable, KIT_DIR / script, *map(str, args)],
        capture_output=True, text=True, encoding="utf-8", check=True, env=os.environ | {"PYTHONUTF8": "1"}).stdout


def main():
    run_files = sorted((ROOT / "results" / "runs").glob("*/*.run"))
    if not run_files:
        sys.exit("no run files, run scripts/bm25_baseline.py first")
    fetch_starter_kit()

    worst, problems = 0.0, 0
    for path in run_files:
        domain, split = path.parent.name, path.name.split("_")[0]
        qrels_path = DATA_DIR / domain / f"qrels_{split}.txt"

        checked = official("format_checker.py", path, "--qrels", qrels_path,
            "--corpus", DATA_DIR / domain / "documents.jsonl", "--doc-key", "doc_id")
        if "RESULT: VALID" not in checked:
            problems += 1
            print(f"{path.relative_to(ROOT)}: format check failed\n{checked}")

        theirs = json.loads(official("scorer.py", "--track", 2, "--run", path, "--qrels", qrels_path, "--json"))
        ours = evaluate(read_run(path), load_qrels(domain, split))
        assert set(theirs["by_turn"]) == set(ours) - {"all"}, f"{path}: turn buckets differ"
        pairs = [(theirs["nDCG@10"], ours["all"]["nDCG@10"])] + [
            (theirs["by_turn"][b], ours[b]["nDCG@10"]) for b in BUCKETS[1:] if b in ours]
        diff = max(abs(a - b) for a, b in pairs)
        worst = max(worst, diff)
        if diff > TOLERANCE:
            problems += 1
            print(f"{path.relative_to(ROOT)}: nDCG@10 differs by {diff:.4f}  (official, ours): {pairs}")

    print(f"{len(run_files)} run files, worst |official - ours| = {worst:.5f}, problems: {problems}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
