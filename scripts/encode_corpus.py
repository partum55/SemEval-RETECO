"""Turns all documents into vectors with MiniLM or bge-base. This was done once and shared, so you rarely need it.

Run: uv run python scripts/encode_corpus.py --model minilm

Writes data/emb/<model>/<domain>.npy and <domain>.ids.json.
"""
import argparse

from semeval_reteco.config import DOMAINS
from semeval_reteco.embeddings import ENCODERS, encode_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(ENCODERS))
    parser.add_argument("--domains", nargs="*", default=list(DOMAINS))
    args = parser.parse_args()

    for domain in args.domains:
        print("encoding", args.model, domain)
        encode_corpus(args.model, domain)


if __name__ == "__main__":
    main()
