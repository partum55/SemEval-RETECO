from semeval_reteco.config import DOMAINS
from semeval_reteco.data import load_corpus, load_turns, load_qrels

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

for split in ("train", "dev"):
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for d in DOMAINS:
        for turn in load_turns(d, split):
            depth = min(turn["turn"], 5)
            counts[depth] += 1
    print(split, counts)

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
        over512 = round(100 * sum(l > 512 for l in lengths) / n)

        print(variant, "depth", depth, "-> mean:", mean,
              "-> over 256:", str(over256) + "%",
              "-> over 512:", str(over512) + "%")