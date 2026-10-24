LABEL = "\n\nConversation History:\n"  # separator of the official "2a_hist" query

# "history" and "last2" feed the dataset's gold answers to the retriever (rule R4):
# report them separately and label them as such.
VARIANTS = ("current", "qonly", "last2", "history")


def build_query(turn, variant, recent_first=False):
    """Text given to the retriever for one turn (a dict from data.load_turns).

    Reads only the turn's query, earlier queries and, for last2/history, the earlier
    gold answers. Never gold_doc_ids or subquestion_reasoning.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if variant == "current" or not turn["prev_queries"]:
        return turn["query"]

    if variant == "history" and not recent_first:
        # The dataset's own field, so numbers match the official baseline exactly. For ~4% of
        # turns (long conversations) it holds only the last 2 turns, not all of prev_qa.
        return turn["query"] + LABEL + turn["gold_history"]

    if variant == "qonly":
        lines = [f"Q: {q}" for q in turn["prev_queries"]]
    elif variant == "last2":
        lines = [f"Q: {q}\nA: {a}" for q, a in turn["prev_qa"][-2:]]
    else:
        lines = [f"Q: {q}\nA: {a}" for q, a in turn["prev_qa"]]
    if recent_first:
        lines.reverse()
    return turn["query"] + LABEL + "\n".join(lines)
