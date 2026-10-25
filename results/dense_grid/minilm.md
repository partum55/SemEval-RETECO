# Encoder grid search, minilm (sentence-transformers/all-MiniLM-L6-v2)

Selected by **train** macro nDCG@10 over 11 domains; only the selected setting and the default are run on dev (rule R1). `history` feeds the dataset's gold answers to the retriever (R4). The default `history` query is the dataset's own `conversation_history` field (the official query), which for long conversations keeps only the last 2 turns; every other history setting is rebuilt from all earlier turns, so compare newest first with "all (rebuilt), oldest first". Read the table as an ablation: what each query-side choice is worth.

## current

| setting | system | train nDCG@10 | train MAP@10 | train Recall@10 | train Recall@100 | train MRR |
|---|---|---|---|---|---|---|
| all, oldest first, 256 tokens **(selected)** (default) | `minilm-current` | 0.1998 | 0.1459 | 0.2954 | 0.5945 | 0.2194 |
| all, oldest first, 512 tokens | `minilm-current-len512` | 0.1998 | 0.1459 | 0.2954 | 0.5945 | 0.2194 |

Dev, selected vs default (macro nDCG@10):

| setting | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| selected: all, oldest first, 256 tokens | 0.1837 | 0.2805 | 0.1499 | 0.1403 | 0.1606 | 0.1546 |
| default: all, oldest first, 256 tokens | 0.1837 | 0.2805 | 0.1499 | 0.1403 | 0.1606 | 0.1546 |

## qonly

| setting | system | train nDCG@10 | train MAP@10 | train Recall@10 | train Recall@100 | train MRR |
|---|---|---|---|---|---|---|
| all, oldest first, 512 tokens | `minilm-qonly-len512` | 0.3164 | 0.2396 | 0.4484 | 0.7647 | 0.3364 |
| all, oldest first, 256 tokens **(selected)** (default) | `minilm-qonly` | 0.3164 | 0.2396 | 0.4484 | 0.7647 | 0.3364 |
| all, newest first, 512 tokens | `minilm-qonly-newest-len512` | 0.3159 | 0.2386 | 0.4503 | 0.7652 | 0.3337 |
| all, newest first, 256 tokens | `minilm-qonly-newest` | 0.3158 | 0.2386 | 0.4502 | 0.7652 | 0.3337 |
| last 2, oldest first, 256 tokens | `minilm-qonly-last2` | 0.2959 | 0.2218 | 0.4248 | 0.7478 | 0.3162 |
| last 2, oldest first, 512 tokens | `minilm-qonly-last2-len512` | 0.2959 | 0.2218 | 0.4248 | 0.7478 | 0.3162 |
| last 2, newest first, 256 tokens | `minilm-qonly-last2-newest` | 0.2959 | 0.2215 | 0.4260 | 0.7472 | 0.3160 |
| last 2, newest first, 512 tokens | `minilm-qonly-last2-newest-len512` | 0.2959 | 0.2215 | 0.4260 | 0.7472 | 0.3160 |
| last 1, oldest first, 256 tokens | `minilm-qonly-last1` | 0.2660 | 0.1969 | 0.3874 | 0.7073 | 0.2852 |
| last 1, oldest first, 512 tokens | `minilm-qonly-last1-len512` | 0.2660 | 0.1969 | 0.3874 | 0.7073 | 0.2852 |

Dev, selected vs default (macro nDCG@10):

| setting | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| selected: all, oldest first, 256 tokens | 0.2913 | 0.2805 | 0.2784 | 0.2879 | 0.3351 | 0.2880 |
| default: all, oldest first, 256 tokens | 0.2913 | 0.2805 | 0.2784 | 0.2879 | 0.3351 | 0.2880 |

## history

| setting | system | train nDCG@10 | train MAP@10 | train Recall@10 | train Recall@100 | train MRR |
|---|---|---|---|---|---|---|
| all (rebuilt), oldest first, 512 tokens **(selected)** | `minilm-history-rebuilt-len512` | 0.3840 | 0.3032 | 0.5197 | 0.8104 | 0.4051 |
| all, newest first, 512 tokens | `minilm-history-newest-len512` | 0.3809 | 0.2998 | 0.5177 | 0.8088 | 0.4012 |
| all (dataset field), oldest first, 512 tokens | `minilm-history-len512` | 0.3803 | 0.2993 | 0.5170 | 0.8090 | 0.4009 |
| all (rebuilt), oldest first, 256 tokens | `minilm-history-rebuilt` | 0.3740 | 0.2932 | 0.5088 | 0.8033 | 0.3956 |
| all (dataset field), oldest first, 256 tokens (default) | `minilm-history` | 0.3735 | 0.2934 | 0.5067 | 0.8026 | 0.3958 |
| all, newest first, 256 tokens | `minilm-history-newest` | 0.3712 | 0.2909 | 0.5058 | 0.8034 | 0.3928 |
| last 2, newest first, 256 tokens | `minilm-history-last2-newest` | 0.3678 | 0.2879 | 0.5012 | 0.8029 | 0.3904 |
| last 2, oldest first, 256 tokens | `minilm-history-last2` | 0.3676 | 0.2881 | 0.5013 | 0.8026 | 0.3902 |
| last 2, oldest first, 512 tokens | `minilm-history-last2-len512` | 0.3675 | 0.2879 | 0.5025 | 0.8032 | 0.3887 |
| last 2, newest first, 512 tokens | `minilm-history-last2-newest-len512` | 0.3671 | 0.2874 | 0.5004 | 0.8018 | 0.3897 |
| last 1, oldest first, 256 tokens | `minilm-history-last1` | 0.3364 | 0.2588 | 0.4665 | 0.7781 | 0.3626 |
| last 1, oldest first, 512 tokens | `minilm-history-last1-len512` | 0.3364 | 0.2588 | 0.4665 | 0.7781 | 0.3626 |

Dev, selected vs default (macro nDCG@10):

| setting | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| selected: all (rebuilt), oldest first, 512 tokens | 0.3474 | 0.2805 | 0.3375 | 0.3786 | 0.4286 | 0.3254 |
| default: all (dataset field), oldest first, 256 tokens | 0.3413 | 0.2805 | 0.3375 | 0.3713 | 0.4070 | 0.3219 |
