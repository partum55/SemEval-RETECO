"""Prints one document, one conversation and one relevance line, to see what the raw data looks like.

Run: uv run python scripts/inspect_data.py --domain drones
"""
import argparse
import json

from semeval_reteco.config import DATA_DIR, DOMAINS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="economics", choices=DOMAINS)
    folder = DATA_DIR / parser.parse_args().domain

    with open(folder / "documents.jsonl", encoding="utf-8") as f:
        print("Document:", json.loads(f.readline()))
    with open(folder / "benchmark_train.json", encoding="utf-8") as f:
        print("Conversation:", json.load(f)[0])
    with open(folder / "qrels_train.txt", encoding="utf-8") as f:
        print("Qrels:", repr(f.readline()))


if __name__ == "__main__":
    main()
