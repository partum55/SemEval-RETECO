from collections import defaultdict

import numpy as np

RESAMPLES = 10_000


def conv_id(qid):
    return qid.rsplit("_turn_", 1)[0]


def paired_bootstrap(a, b, splits=("dev",), metric="ndcg_cut_10", resamples=RESAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    domains = sorted({domain for domain, split in a if split in splits})
    point, boot = 0.0, np.zeros(resamples)
    for domain in domains:
        by_conv = defaultdict(list)
        for split in splits:
            sa, sb = a[domain, split], b[domain, split]
            for qid in sa:
                by_conv[split, conv_id(qid)].append(sa[qid][metric] - sb[qid][metric])
        sums = np.array([sum(v) for v in by_conv.values()])
        counts = np.array([len(v) for v in by_conv.values()])
        picks = rng.integers(len(sums), size=(resamples, len(sums)))
        point += sums.sum() / counts.sum()
        boot += sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    point, boot = point / len(domains), boot / len(domains)
    return point, *np.percentile(boot, [2.5, 97.5])
