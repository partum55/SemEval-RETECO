LABEL = "\n\nConversation History:\n"
VARIANTS = ("current", "qonly", "last2", "history")


def build_query(turn, variant, recent_first=False, window=None, rebuild=False):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if window is not None and window < 1:
        raise ValueError(f"window must be >= 1 or None, got {window}")
    if variant == "current" or not turn["prev_queries"]:
        return turn["query"]

    if variant == "history" and not recent_first and window is None and not rebuild:
        return turn["query"] + LABEL + turn["gold_history"]

    if variant == "qonly":
        lines = [f"Q: {q}" for q in turn["prev_queries"]]
    else:
        lines = [f"Q: {q}\nA: {a}" for q, a in turn["prev_qa"]]
    if variant == "last2":
        lines = lines[-2:]
    if window is not None:
        lines = lines[-window:]
    if recent_first:
        lines.reverse()
    return turn["query"] + LABEL + "\n".join(lines)
