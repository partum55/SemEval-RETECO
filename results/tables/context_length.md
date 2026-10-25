# Context length: what truncating long history queries costs

Dense encoders read a fixed number of tokens and cut the rest from the END of the text. The default `history` query is "question + oldest turn ... newest turn", so a long query loses its newest turns. *Newest first* reverses the earlier turns, so the oldest ones are cut instead; *512 tokens* lets MiniLM read twice as much (its documents stay encoded at 256). The default query is the dataset's `conversation_history` field, which keeps only the last 2 turns for 26% of T5+ turns (none before T5); newest first is rebuilt from all turns, so its effect is measured against *all turns, oldest first*, and each pair below changes one thing. `history` contains the dataset's gold answers (R4).

## History queries over the token limit (train + dev)

| depth | turns | MiniLM median tokens | MiniLM > 256 | MiniLM > 512 | bge median tokens (with prefix) | bge > 512 |
|---|---|---|---|---|---|---|
| T1 | 707 | 25 | 0.0% | 0.0% | 33 | 0.0% |
| T2 | 707 | 141 | 0.0% | 0.0% | 149 | 0.0% |
| T3 | 707 | 250 | 42.1% | 0.0% | 258 | 0.0% |
| T4 | 411 | 363 | 99.8% | 0.0% | 371 | 0.0% |
| T5+ | 439 | 477 | 100.0% | 38.0% | 485 | 40.3% |

## nDCG@10 by turn depth, dev (macro over domains)

| model | query | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|---|
| MiniLM | 256 tokens, dataset field, oldest first (default) | 0.3413 | 0.2805 | 0.3375 | 0.3713 | 0.4070 | 0.3219 |
| MiniLM | 256 tokens, all turns, oldest first | 0.3408 | 0.2805 | 0.3375 | 0.3713 | 0.4070 | 0.3219 |
| MiniLM | 256 tokens, all turns, newest first | 0.3404 | 0.2805 | 0.3375 | 0.3726 | 0.3933 | 0.3415 |
| MiniLM | 512 tokens, dataset field, oldest first | 0.3479 | 0.2805 | 0.3375 | 0.3786 | 0.4286 | 0.3287 |
| bge-base | 512 tokens, dataset field, oldest first (default) | 0.3862 | 0.3084 | 0.3752 | 0.4260 | 0.4778 | 0.3965 |
| bge-base | 512 tokens, all turns, oldest first | 0.3874 | 0.3084 | 0.3752 | 0.4260 | 0.4778 | 0.4055 |
| bge-base | 512 tokens, all turns, newest first | 0.3901 | 0.3084 | 0.3752 | 0.4385 | 0.4954 | 0.3562 |

## nDCG@10 by turn depth, train (macro over domains)

| model | query | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|---|
| MiniLM | 256 tokens, dataset field, oldest first (default) | 0.3735 | 0.2910 | 0.3768 | 0.4096 | 0.4107 | 0.4339 |
| MiniLM | 256 tokens, all turns, oldest first | 0.3740 | 0.2910 | 0.3768 | 0.4096 | 0.4107 | 0.4341 |
| MiniLM | 256 tokens, all turns, newest first | 0.3712 | 0.2910 | 0.3768 | 0.4011 | 0.4203 | 0.4294 |
| MiniLM | 512 tokens, dataset field, oldest first | 0.3803 | 0.2910 | 0.3768 | 0.4084 | 0.4322 | 0.4654 |
| bge-base | 512 tokens, dataset field, oldest first (default) | 0.4413 | 0.3485 | 0.4312 | 0.4823 | 0.4834 | 0.5151 |
| bge-base | 512 tokens, all turns, oldest first | 0.4439 | 0.3484 | 0.4312 | 0.4823 | 0.4834 | 0.5263 |
| bge-base | 512 tokens, all turns, newest first | 0.4398 | 0.3484 | 0.4312 | 0.4817 | 0.4777 | 0.5054 |

## nDCG@10 by turn depth, train + dev (macro over domains)

| model | query | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|---|
| MiniLM | 256 tokens, dataset field, oldest first (default) | 0.3654 | 0.2881 | 0.3653 | 0.3982 | 0.4113 | 0.4181 |
| MiniLM | 256 tokens, all turns, oldest first | 0.3656 | 0.2881 | 0.3653 | 0.3982 | 0.4113 | 0.4177 |
| MiniLM | 256 tokens, all turns, newest first | 0.3632 | 0.2881 | 0.3653 | 0.3926 | 0.4131 | 0.4153 |
| MiniLM | 512 tokens, dataset field, oldest first | 0.3720 | 0.2881 | 0.3653 | 0.3995 | 0.4324 | 0.4416 |
| bge-base | 512 tokens, dataset field, oldest first (default) | 0.4265 | 0.3367 | 0.4147 | 0.4654 | 0.4826 | 0.4911 |
| bge-base | 512 tokens, all turns, oldest first | 0.4287 | 0.3367 | 0.4147 | 0.4654 | 0.4826 | 0.5017 |
| bge-base | 512 tokens, all turns, newest first | 0.4264 | 0.3367 | 0.4147 | 0.4687 | 0.4808 | 0.4801 |

## Paired difference in nDCG@10, train + dev, 10,000 conversation bootstrap resamples

None of these settings was fitted on any split, so pooling train and dev is not optimistic.

| what changes | A | B | turns | Δ (A − B) | 95% CI |
|---|---|---|---|---|---|
| order: newest turns first | `minilm-history-newest` | `minilm-history-rebuilt` | all turns | -0.0024 | [-0.0091, +0.0042] |
| order: newest turns first | `minilm-history-newest` | `minilm-history-rebuilt` | T4 + T5+ | -0.0017 | [-0.0228, +0.0184] |
| order: newest turns first | `bge-base-history-newest` | `bge-base-history-rebuilt` | all turns | -0.0023 | [-0.0073, +0.0029] |
| order: newest turns first | `bge-base-history-newest` | `bge-base-history-rebuilt` | T4 + T5+ | -0.0119 | [-0.0264, +0.0032] |
| length: 512 tokens instead of 256 | `minilm-history-len512` | `minilm-history` | all turns | +0.0067 | [+0.0031, +0.0102] |
| length: 512 tokens instead of 256 | `minilm-history-len512` | `minilm-history` | T4 + T5+ | +0.0236 | [+0.0109, +0.0359] |
| content: all turns instead of the dataset field | `minilm-history-rebuilt` | `minilm-history` | all turns | +0.0002 | [-0.0017, +0.0022] |
| content: all turns instead of the dataset field | `minilm-history-rebuilt` | `minilm-history` | T4 + T5+ | +0.0003 | [-0.0058, +0.0063] |
| content: all turns instead of the dataset field | `bge-base-history-rebuilt` | `bge-base-history` | all turns | +0.0022 | [+0.0002, +0.0042] |
| content: all turns instead of the dataset field | `bge-base-history-rebuilt` | `bge-base-history` | T4 + T5+ | +0.0067 | [-0.0001, +0.0131] |
