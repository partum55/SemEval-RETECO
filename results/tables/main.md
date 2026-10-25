# Main results

Macro average over the 11 domains (each domain counts equally, like the leaderboard). `history` feeds the dataset's gold answers to the retriever (R4); `current` and `q-only` do not. *Grid-selected* rows appear only where a grid search on train (R1) picked something other than the default (results/dense_grid/, results/bm25/grid.md); the official BM25 (k1=0.9, b=0.4) stays the baseline.

| query | retriever | dev nDCG@10 | dev MAP@10 | dev Recall@10 | dev Recall@100 | dev MRR | train nDCG@10 |
|---|---|---|---|---|---|---|---|
| current | BM25 | 0.1819 | 0.1353 | 0.2782 | 0.5551 | 0.1844 | 0.1846 |
| current | BM25, grid-selected | 0.1936 | 0.1472 | 0.2890 | 0.5705 | 0.1967 | 0.1955 |
| current | MiniLM | 0.1837 | 0.1344 | 0.2723 | 0.5934 | 0.2004 | 0.1998 |
| current | bge-base | 0.2353 | 0.1778 | 0.3410 | 0.6612 | 0.2515 | 0.2572 |
| q-only | BM25 | 0.2921 | 0.2273 | 0.4205 | 0.7302 | 0.2944 | 0.2993 |
| q-only | BM25, grid-selected | 0.3015 | 0.2376 | 0.4275 | 0.7390 | 0.3036 | 0.3091 |
| q-only | MiniLM | 0.2913 | 0.2172 | 0.4195 | 0.7574 | 0.3130 | 0.3164 |
| q-only | bge-base | 0.3222 | 0.2410 | 0.4765 | 0.8076 | 0.3271 | 0.3640 |
| history (gold answers) | BM25 | 0.4377 | 0.3610 | 0.5639 | 0.8240 | 0.4613 | 0.4543 |
| history (gold answers) | BM25, grid-selected | 0.4330 | 0.3532 | 0.5647 | 0.8326 | 0.4583 | 0.4573 |
| history (gold answers) | MiniLM | 0.3413 | 0.2583 | 0.4852 | 0.8014 | 0.3598 | 0.3735 |
| history (gold answers) | MiniLM, grid-selected | 0.3474 | 0.2659 | 0.4865 | 0.8069 | 0.3697 | 0.3840 |
| history (gold answers) | bge-base | 0.3862 | 0.3006 | 0.5398 | 0.8542 | 0.3950 | 0.4413 |
| history (gold answers) | bge-base, grid-selected | 0.3874 | 0.3020 | 0.5407 | 0.8526 | 0.3961 | 0.4439 |
