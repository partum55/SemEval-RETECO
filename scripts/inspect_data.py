import json

folder = "data/reteco/track2_recor/economics/"

with open(folder + "documents.jsonl", encoding="utf-8") as f:
    doc = json.loads(f.readline())
print("Document:", doc)

with open(folder + "benchmark_train.json", encoding="utf-8") as f:
    conversations = json.load(f)
print("Conversation:", conversations[0])

with open(folder + "qrels_train.txt", encoding="utf-8") as f:
    print("Qrels:", repr(f.readline()))