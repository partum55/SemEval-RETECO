# dev nDCG@10 per domain (default settings)

Macro average over the 11 domains (each domain counts equally, like the leaderboard). `history` feeds the dataset's gold answers to the retriever (R4); `current` and `q-only` do not. *Grid-selected* rows appear only where a grid search on train (R1) picked something other than the default (results/dense_grid/, results/bm25/grid.md); the official BM25 (k1=0.9, b=0.4) stays the baseline.

| domain | dev topics | BM25, current | MiniLM, current | bge-base, current | BM25, q-only | MiniLM, q-only | bge-base, q-only | BM25, history (gold answers) | MiniLM, history (gold answers) | bge-base, history (gold answers) |
|---|---|---|---|---|---|---|---|---|---|---|
| biology | 115 | 0.2034 | 0.1779 | 0.2457 | 0.3866 | 0.3296 | 0.3391 | 0.6078 | 0.4181 | 0.4442 |
| drones | 38 | 0.1695 | 0.1658 | 0.2322 | 0.2642 | 0.2843 | 0.2838 | 0.2954 | 0.3213 | 0.3254 |
| earth_science | 133 | 0.2209 | 0.1426 | 0.2464 | 0.4483 | 0.2512 | 0.3574 | 0.6240 | 0.3401 | 0.4433 |
| economics | 92 | 0.1628 | 0.1652 | 0.2167 | 0.2455 | 0.2883 | 0.3343 | 0.4753 | 0.3453 | 0.4194 |
| hardware | 58 | 0.1586 | 0.1963 | 0.2603 | 0.2148 | 0.2961 | 0.2951 | 0.3315 | 0.2971 | 0.3132 |
| law | 66 | 0.1147 | 0.2147 | 0.2260 | 0.2300 | 0.3148 | 0.2904 | 0.3437 | 0.3252 | 0.3149 |
| medicalsciences | 44 | 0.1168 | 0.2678 | 0.2449 | 0.1072 | 0.3616 | 0.3560 | 0.1730 | 0.3564 | 0.3562 |
| politics | 54 | 0.1483 | 0.2685 | 0.3383 | 0.2056 | 0.3826 | 0.4403 | 0.3302 | 0.3780 | 0.4546 |
| psychology | 99 | 0.2635 | 0.1622 | 0.2422 | 0.3945 | 0.2700 | 0.3380 | 0.5760 | 0.3648 | 0.4430 |
| robotics | 75 | 0.1483 | 0.0543 | 0.1009 | 0.3157 | 0.0964 | 0.1863 | 0.5051 | 0.2552 | 0.3138 |
| sustainable_living | 84 | 0.2937 | 0.2058 | 0.2349 | 0.4003 | 0.3288 | 0.3229 | 0.5530 | 0.3523 | 0.4206 |
