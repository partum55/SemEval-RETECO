from semeval_reteco.config import DOMAINS
from semeval_reteco.data import load_corpus, load_turns, load_qrels
import re
import matplotlib.pyplot as plt
from semeval_reteco.config import ROOT
import random
random.seed(0)

for d in DOMAINS:
    n_docs = len(load_corpus(d)[0])
    n_train = len(load_turns(d, "train"))
    n_dev = len(load_turns(d, "dev"))

    qrels = load_qrels(d, "train")
    gold_per_turn = sum(len(docs) for docs in qrels.values()) / len(qrels)

    print(d)
    print("  documents:", n_docs)
    print("  train turns:", n_train)
    print("  dev turns:", n_dev)
    print("  gold docs per turn:", round(gold_per_turn, 2))

print()
print("Turn depth:")
depth_counts = {}

for split in ("train", "dev"):
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for d in DOMAINS:
        for turn in load_turns(d, split):
            depth = min(turn["turn"], 5)
            counts[depth] += 1
    print(split, counts)
    depth_counts[split] = counts

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def make_query(turn, variant):
    if variant == "current":
        return turn["query"]
    if variant == "qonly":
        history = "\n".join("Q: " + q for q in turn["prev_queries"])
    else:
        history = turn["gold_history"]
    return turn["query"] + "\n\nConversation History:\n" + history


train_turns = []
for d in DOMAINS:
    train_turns += load_turns(d, "train")

print()
print("Query length in tokens (MiniLM tokenizer), train:")

over256_pct = {}
for variant in ("current", "qonly", "history"):
    for depth in (1, 2, 3, 4, 5):
        texts = []
        for turn in train_turns:
            if min(turn["turn"], 5) == depth:
                texts.append(make_query(turn, variant))

        lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
        n = len(lengths)
        mean = round(sum(lengths) / n)
        over256 = round(100 * sum(l > 256 for l in lengths) / n)
        over256_pct[(variant, depth)] = over256
        over512 = round(100 * sum(l > 512 for l in lengths) / n)

        print(variant, "depth", depth, "-> mean:", mean,
              "-> over 256:", str(over256) + "%",
              "-> over 512:", str(over512) + "%")


print()
print("Document length in tokens (MiniLM tokenizer), sample of 3000 docs per domain:")

for d in DOMAINS:
    all_texts = load_corpus(d)[1]
    texts = random.sample(all_texts, min(3000, len(all_texts)))

    lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    lengths.sort()
    n = len(lengths)
    mean = round(sum(lengths) / n)
    median = lengths[n // 2]
    p95 = lengths[int(n * 0.95)]
    over256 = round(100 * sum(l > 256 for l in lengths) / n)
    over512 = round(100 * sum(l > 512 for l in lengths) / n)

    print(d, "-> mean:", mean,
          "-> median:", median,
          "-> p95:", p95,
          "-> over 256:", str(over256) + "%",
          "-> over 512:", str(over512) + "%")

    

def words(text):
    return set(w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3)


variants = ("current", "qonly", "history")
shared_sum = {"current": 0, "qonly": 0, "history": 0}
share_sum = {"current": 0, "qonly": 0, "history": 0}
n_turns = 0

for d in DOMAINS:
    doc_ids, texts = load_corpus(d)
    doc_text = dict(zip(doc_ids, texts))
    qrels = load_qrels(d, "train")

    for turn in load_turns(d, "train"):
        gold_words = set()
        for doc_id in qrels[turn["qid"]]:
            if doc_id in doc_text:
                gold_words |= words(doc_text[doc_id])

        for variant in variants:
            query_words = words(make_query(turn, variant))
            shared = len(query_words & gold_words)
            shared_sum[variant] += shared
            share_sum[variant] += shared / len(query_words)
        n_turns += 1

print()
print("Word overlap between the query and its gold documents, train:")

for variant in variants:
    mean_shared = round(shared_sum[variant] / n_turns, 1)
    mean_share = round(100 * share_sum[variant] / n_turns)
    print(variant, "-> shared words per turn:", mean_shared,
          "-> share of query words found in gold docs:", str(mean_share) + "%")

out_dir = ROOT / "results" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
labels = ["T1", "T2", "T3", "T4", "T5+"]

for split in ("train", "dev"):
    plt.plot(labels, [depth_counts[split][k] for k in (1, 2, 3, 4, 5)],
             marker="o", label=split)
plt.ylabel("number of turns")
plt.title("Turns per conversation depth")
plt.legend()
plt.savefig(out_dir / "turn_depth.png")
plt.close()

for variant in ("current", "qonly", "history"):
    plt.plot(labels, [over256_pct[(variant, k)] for k in (1, 2, 3, 4, 5)],
             marker="o", label=variant)
plt.ylabel("queries longer than 256 tokens (%)")
plt.title("Share of queries truncated by MiniLM (train)")
plt.legend()
plt.savefig(out_dir / "query_length.png")
plt.close()

