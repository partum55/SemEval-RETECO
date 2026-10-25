# Encoder grid search, bge-base (BAAI/bge-base-en-v1.5)

Selected by **train** macro nDCG@10 over 11 domains; only the selected setting and the default are run on dev (rule R1). `history` feeds the dataset's gold answers to the retriever (R4). The default `history` query is the dataset's own `conversation_history` field (the official query), which for long conversations keeps only the last 2 turns; every other history setting is rebuilt from all earlier turns, so compare newest first with "all (rebuilt), oldest first". Read the table as an ablation: what each query-side choice is worth.

## current

| setting | system | train nDCG@10 | train MAP@10 | train Recall@10 | train Recall@100 | train MRR |
|---|---|---|---|---|---|---|
| all, oldest first, 512 tokens **(selected)** (default) | `bge-base-current` | 0.2572 | 0.1951 | 0.3657 | 0.6678 | 0.2770 |
| all, oldest first, 512 tokens, no prefix | `bge-base-current-noprefix` | 0.2143 | 0.1599 | 0.3066 | 0.6079 | 0.2352 |

Dev, selected vs default (macro nDCG@10):

| setting | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| selected: all, oldest first, 512 tokens | 0.2353 | 0.3084 | 0.2231 | 0.1893 | 0.2447 | 0.1688 |
| default: all, oldest first, 512 tokens | 0.2353 | 0.3084 | 0.2231 | 0.1893 | 0.2447 | 0.1688 |

## qonly

| setting | system | train nDCG@10 | train MAP@10 | train Recall@10 | train Recall@100 | train MRR |
|---|---|---|---|---|---|---|
| all, oldest first, 512 tokens **(selected)** (default) | `bge-base-qonly` | 0.3640 | 0.2836 | 0.4991 | 0.8154 | 0.3857 |
| all, newest first, 512 tokens | `bge-base-qonly-newest` | 0.3612 | 0.2804 | 0.4966 | 0.8163 | 0.3836 |
| last 2, newest first, 512 tokens | `bge-base-qonly-last2-newest` | 0.3420 | 0.2631 | 0.4763 | 0.7903 | 0.3617 |
| last 2, oldest first, 512 tokens | `bge-base-qonly-last2` | 0.3413 | 0.2635 | 0.4730 | 0.7905 | 0.3617 |
| last 1, oldest first, 512 tokens | `bge-base-qonly-last1` | 0.3115 | 0.2379 | 0.4377 | 0.7525 | 0.3313 |
| all, oldest first, 512 tokens, no prefix | `bge-base-qonly-noprefix` | 0.3032 | 0.2307 | 0.4284 | 0.7439 | 0.3234 |
| all, newest first, 512 tokens, no prefix | `bge-base-qonly-newest-noprefix` | 0.2989 | 0.2262 | 0.4256 | 0.7424 | 0.3172 |
| last 2, oldest first, 512 tokens, no prefix | `bge-base-qonly-last2-noprefix` | 0.2719 | 0.2034 | 0.3933 | 0.7105 | 0.2892 |
| last 2, newest first, 512 tokens, no prefix | `bge-base-qonly-last2-newest-noprefix` | 0.2694 | 0.2006 | 0.3922 | 0.7100 | 0.2859 |
| last 1, oldest first, 512 tokens, no prefix | `bge-base-qonly-last1-noprefix` | 0.2344 | 0.1738 | 0.3420 | 0.6489 | 0.2509 |

Dev, selected vs default (macro nDCG@10):

| setting | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| selected: all, oldest first, 512 tokens | 0.3222 | 0.3084 | 0.3046 | 0.3341 | 0.3652 | 0.3406 |
| default: all, oldest first, 512 tokens | 0.3222 | 0.3084 | 0.3046 | 0.3341 | 0.3652 | 0.3406 |

## history

| setting | system | train nDCG@10 | train MAP@10 | train Recall@10 | train Recall@100 | train MRR |
|---|---|---|---|---|---|---|
| all (rebuilt), oldest first, 512 tokens **(selected)** | `bge-base-history-rebuilt` | 0.4439 | 0.3573 | 0.5869 | 0.8653 | 0.4670 |
| all (dataset field), oldest first, 512 tokens (default) | `bge-base-history` | 0.4413 | 0.3548 | 0.5846 | 0.8621 | 0.4639 |
| all, newest first, 512 tokens | `bge-base-history-newest` | 0.4398 | 0.3525 | 0.5857 | 0.8627 | 0.4622 |
| last 2, oldest first, 512 tokens | `bge-base-history-last2` | 0.4251 | 0.3388 | 0.5697 | 0.8507 | 0.4475 |
| last 2, newest first, 512 tokens | `bge-base-history-last2-newest` | 0.4235 | 0.3367 | 0.5696 | 0.8515 | 0.4447 |
| all (rebuilt), oldest first, 512 tokens, no prefix | `bge-base-history-rebuilt-noprefix` | 0.4120 | 0.3284 | 0.5519 | 0.8475 | 0.4335 |
| all (dataset field), oldest first, 512 tokens, no prefix | `bge-base-history-noprefix` | 0.4090 | 0.3256 | 0.5489 | 0.8447 | 0.4299 |
| all, newest first, 512 tokens, no prefix | `bge-base-history-newest-noprefix` | 0.4062 | 0.3240 | 0.5425 | 0.8435 | 0.4288 |
| last 2, oldest first, 512 tokens, no prefix | `bge-base-history-last2-noprefix` | 0.3984 | 0.3156 | 0.5364 | 0.8338 | 0.4225 |
| last 2, newest first, 512 tokens, no prefix | `bge-base-history-last2-newest-noprefix` | 0.3959 | 0.3137 | 0.5332 | 0.8337 | 0.4185 |
| last 1, oldest first, 512 tokens | `bge-base-history-last1` | 0.3923 | 0.3096 | 0.5296 | 0.8241 | 0.4172 |
| last 1, oldest first, 512 tokens, no prefix | `bge-base-history-last1-noprefix` | 0.3667 | 0.2870 | 0.5003 | 0.8036 | 0.3906 |

Dev, selected vs default (macro nDCG@10):

| setting | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| selected: all (rebuilt), oldest first, 512 tokens | 0.3874 | 0.3084 | 0.3752 | 0.4260 | 0.4778 | 0.4055 |
| default: all (dataset field), oldest first, 512 tokens | 0.3862 | 0.3084 | 0.3752 | 0.4260 | 0.4778 | 0.3965 |
