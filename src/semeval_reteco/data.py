import json

from semeval_reteco.config import DATA_DIR


def load_corpus(domain):
    doc_ids = []
    texts = []
    with open(DATA_DIR / domain / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_ids.append(doc["doc_id"])
            texts.append(doc["content"])
    return doc_ids, texts


def load_turns(domain, split):
    with open(DATA_DIR / domain / f"benchmark_{split}.json", encoding="utf-8") as f:
        conversations = json.load(f)

    turns = []
    for conv in conversations:
        prev_queries = []
        prev_qa = []
        for t in conv["turns"]:
            turns.append({
                "domain": domain,
                "split": split,
                "qid": f"{conv['id']}_turn_{t['turn_id']}",
                "conv_id": conv["id"],
                "turn": t["turn_id"],
                "query": t["query"],
                "prev_queries": list(prev_queries),
                "prev_qa": list(prev_qa),
                "gold_history": t["conversation_history"],
            })
            prev_queries.append(t["query"])
            prev_qa.append((t["query"], t["answer"]))
    return turns


def load_qrels(domain, split):
    qrels = {}
    with open(DATA_DIR / domain / f"qrels_{split}.txt", encoding="utf-8") as f:
        for line in f:
            qid, _, doc_id, rel = line.strip().split("\t")
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels