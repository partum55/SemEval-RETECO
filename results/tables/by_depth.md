# nDCG@10 by turn depth

Macro average over the 11 domains (each domain counts equally, like the leaderboard). `history` feeds the dataset's gold answers to the retriever (R4); `current` and `q-only` do not. *Grid-selected* rows appear only where a grid search on train (R1) picked something other than the default (results/dense_grid/, results/bm25/grid.md); the official BM25 (k1=0.9, b=0.4) stays the baseline.

## dev

| query | retriever | all (n=858) | T1 (n=211) | T2 (n=211) | T3 (n=211) | T4 (n=121) | T5+ (n=104) |
|---|---|---|---|---|---|---|---|
| current | BM25 | 0.1819 | 0.2619 | 0.1794 | 0.1557 | 0.1290 | 0.1126 |
| current | BM25, grid-selected | 0.1936 | 0.2696 | 0.1945 | 0.1732 | 0.1324 | 0.1249 |
| current | MiniLM | 0.1837 | 0.2805 | 0.1499 | 0.1403 | 0.1606 | 0.1546 |
| current | bge-base | 0.2353 | 0.3084 | 0.2231 | 0.1893 | 0.2447 | 0.1688 |
| q-only | BM25 | 0.2921 | 0.2619 | 0.2932 | 0.3125 | 0.3126 | 0.2491 |
| q-only | BM25, grid-selected | 0.3015 | 0.2736 | 0.2974 | 0.3201 | 0.3222 | 0.2683 |
| q-only | MiniLM | 0.2913 | 0.2805 | 0.2784 | 0.2879 | 0.3351 | 0.2880 |
| q-only | bge-base | 0.3222 | 0.3084 | 0.3046 | 0.3341 | 0.3652 | 0.3406 |
| history (gold answers) | BM25 | 0.4377 | 0.2619 | 0.4979 | 0.5087 | 0.4978 | 0.4355 |
| history (gold answers) | BM25, grid-selected | 0.4330 | 0.2420 | 0.4911 | 0.5119 | 0.5006 | 0.4499 |
| history (gold answers) | MiniLM | 0.3413 | 0.2805 | 0.3375 | 0.3713 | 0.4070 | 0.3219 |
| history (gold answers) | MiniLM, grid-selected | 0.3474 | 0.2805 | 0.3375 | 0.3786 | 0.4286 | 0.3254 |
| history (gold answers) | bge-base | 0.3862 | 0.3084 | 0.3752 | 0.4260 | 0.4778 | 0.3965 |
| history (gold answers) | bge-base, grid-selected | 0.3874 | 0.3084 | 0.3752 | 0.4260 | 0.4778 | 0.4055 |

## train

| query | retriever | all (n=2113) | T1 (n=496) | T2 (n=496) | T3 (n=496) | T4 (n=290) | T5+ (n=335) |
|---|---|---|---|---|---|---|---|
| current | BM25 | 0.1846 | 0.2641 | 0.1856 | 0.1551 | 0.1406 | 0.1445 |
| current | BM25, grid-selected | 0.1955 | 0.2688 | 0.2011 | 0.1672 | 0.1533 | 0.1520 |
| current | MiniLM | 0.1998 | 0.2910 | 0.1878 | 0.1756 | 0.1825 | 0.1454 |
| current | bge-base | 0.2572 | 0.3485 | 0.2487 | 0.2295 | 0.2094 | 0.2258 |
| q-only | BM25 | 0.2993 | 0.2641 | 0.2928 | 0.2927 | 0.3292 | 0.3507 |
| q-only | BM25, grid-selected | 0.3091 | 0.2685 | 0.3101 | 0.2998 | 0.3362 | 0.3592 |
| q-only | MiniLM | 0.3164 | 0.2910 | 0.3001 | 0.3217 | 0.3312 | 0.3831 |
| q-only | bge-base | 0.3640 | 0.3485 | 0.3430 | 0.3665 | 0.3723 | 0.4319 |
| history (gold answers) | BM25 | 0.4543 | 0.2641 | 0.5030 | 0.5175 | 0.5211 | 0.5121 |
| history (gold answers) | BM25, grid-selected | 0.4573 | 0.2531 | 0.5008 | 0.5357 | 0.5261 | 0.5202 |
| history (gold answers) | MiniLM | 0.3735 | 0.2910 | 0.3768 | 0.4096 | 0.4107 | 0.4339 |
| history (gold answers) | MiniLM, grid-selected | 0.3840 | 0.2910 | 0.3768 | 0.4084 | 0.4322 | 0.4865 |
| history (gold answers) | bge-base | 0.4413 | 0.3485 | 0.4312 | 0.4823 | 0.4834 | 0.5151 |
| history (gold answers) | bge-base, grid-selected | 0.4439 | 0.3485 | 0.4312 | 0.4823 | 0.4834 | 0.5263 |
