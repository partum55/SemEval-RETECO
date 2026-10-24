# Newest-first history order, dense minilm (max 256 tokens), 11 domains

`history` feeds the dataset's gold answers to the retriever (R4); `q-only` does not. Default order is oldest turn first, so truncation cuts the newest turns.

## Macro average over domains

| split | condition | nDCG@10 | MAP@10 | Recall@10 | MRR |
|---|---|---|---|---|---|
| dev | history (gold answers) | 0.3413 | 0.2583 | 0.4852 | 0.3598 |
| dev | history (gold answers), newest first | 0.3404 | 0.2578 | 0.4831 | 0.3608 |
| dev | q-only | 0.2913 | 0.2172 | 0.4195 | 0.3130 |
| dev | q-only, newest first | 0.2902 | 0.2172 | 0.4145 | 0.3132 |
| train | history (gold answers) | 0.3735 | 0.2934 | 0.5067 | 0.3958 |
| train | history (gold answers), newest first | 0.3712 | 0.2909 | 0.5058 | 0.3928 |
| train | q-only | 0.3164 | 0.2396 | 0.4484 | 0.3364 |
| train | q-only, newest first | 0.3158 | 0.2386 | 0.4502 | 0.3337 |

## nDCG@10 by turn depth, dev (macro over domains)

| condition | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| history (gold answers) | 0.3413 | 0.2805 | 0.3375 | 0.3713 | 0.4070 | 0.3219 |
| history (gold answers), newest first | 0.3404 | 0.2805 | 0.3375 | 0.3726 | 0.3933 | 0.3415 |
| q-only | 0.2913 | 0.2805 | 0.2784 | 0.2879 | 0.3351 | 0.2880 |
| q-only, newest first | 0.2902 | 0.2805 | 0.2784 | 0.2858 | 0.3282 | 0.2923 |

## nDCG@10 by turn depth, train (macro over domains)

| condition | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| history (gold answers) | 0.3735 | 0.2910 | 0.3768 | 0.4096 | 0.4107 | 0.4339 |
| history (gold answers), newest first | 0.3712 | 0.2910 | 0.3768 | 0.4011 | 0.4203 | 0.4294 |
| q-only | 0.3164 | 0.2910 | 0.3001 | 0.3217 | 0.3312 | 0.3831 |
| q-only, newest first | 0.3158 | 0.2910 | 0.3001 | 0.3211 | 0.3347 | 0.3749 |

## Paired difference in nDCG@10 (A − B), 10,000 conversation-level bootstrap resamples, 95% CI

No setting was fitted on dev, so pooling train and dev is not optimistic.

| A | B | split | Δ | 95% CI |
|---|---|---|---|---|
| history (gold answers), newest first | history (gold answers) | dev | -0.0009 | [-0.0107, +0.0073] |
| history (gold answers), newest first | history (gold answers) | train+dev | -0.0022 | [-0.0068, +0.0032] |
| q-only, newest first | q-only | dev | -0.0011 | [-0.0042, +0.0028] |
| q-only, newest first | q-only | train+dev | -0.0008 | [-0.0025, +0.0010] |

## Query length: share of history queries over 256 tokens (train + dev)

| depth | queries | > limit | median tokens |
|---|---|---|---|
| T1 | 707 | 0.0% | 25 |
| T2 | 707 | 0.0% | 141 |
| T3 | 707 | 42.1% | 250 |
| T4 | 411 | 99.8% | 363 |
| T5+ | 439 | 100.0% | 477 |
