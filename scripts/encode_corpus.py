import argparse

from semeval_reteco.config import DOMAINS
from semeval_reteco.embeddings import encode_corpus

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["minilm", "bge-base"])
parser.add_argument("--domains", nargs="*", default=list(DOMAINS))
args = parser.parse_args()

for domain in args.domains:
    print("encoding", args.model, domain)
    encode_corpus(args.model, domain)