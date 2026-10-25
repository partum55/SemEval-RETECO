# BM25 baseline (official k1=0.9, b=0.4), 11 domains

`last2` and `history` feed the dataset's gold answers to the retriever (R4); `current` and `q-only` do not.

## Macro average over domains

| split | variant | nDCG@10 | MAP@10 | Recall@10 | Recall@100 | MRR |
|---|---|---|---|---|---|---|
| dev | current | 0.1819 | 0.1353 | 0.2782 | 0.5551 | 0.1844 |
| dev | q-only | 0.2921 | 0.2273 | 0.4205 | 0.7302 | 0.2944 |
| dev | last2 (gold answers) | 0.4341 | 0.3577 | 0.5577 | 0.8119 | 0.4601 |
| dev | history (gold answers) | 0.4377 | 0.3610 | 0.5639 | 0.8240 | 0.4613 |
| train | current | 0.1846 | 0.1369 | 0.2787 | 0.5460 | 0.1906 |
| train | q-only | 0.2993 | 0.2338 | 0.4246 | 0.7401 | 0.3070 |
| train | last2 (gold answers) | 0.4438 | 0.3645 | 0.5782 | 0.8179 | 0.4622 |
| train | history (gold answers) | 0.4543 | 0.3740 | 0.5918 | 0.8286 | 0.4692 |

## nDCG@10 per domain, dev

| domain | topics | current | q-only | last2 (gold answers) | history (gold answers) |
|---|---|---|---|---|---|
| biology | 115 | 0.2034 | 0.3866 | 0.6119 | 0.6078 |
| drones | 38 | 0.1695 | 0.2642 | 0.2880 | 0.2954 |
| earth_science | 133 | 0.2209 | 0.4483 | 0.6065 | 0.6240 |
| economics | 92 | 0.1628 | 0.2455 | 0.4742 | 0.4753 |
| hardware | 58 | 0.1586 | 0.2148 | 0.3312 | 0.3315 |
| law | 66 | 0.1147 | 0.2300 | 0.3171 | 0.3437 |
| medicalsciences | 44 | 0.1168 | 0.1072 | 0.1731 | 0.1730 |
| politics | 54 | 0.1483 | 0.2056 | 0.3585 | 0.3302 |
| psychology | 99 | 0.2635 | 0.3945 | 0.5713 | 0.5760 |
| robotics | 75 | 0.1483 | 0.3157 | 0.5093 | 0.5051 |
| sustainable_living | 84 | 0.2937 | 0.4003 | 0.5340 | 0.5530 |

## nDCG@10 per domain, train

| domain | topics | current | q-only | last2 (gold answers) | history (gold answers) |
|---|---|---|---|---|---|
| biology | 247 | 0.2252 | 0.4084 | 0.6288 | 0.6337 |
| drones | 104 | 0.1400 | 0.1474 | 0.2985 | 0.3164 |
| earth_science | 321 | 0.2769 | 0.4905 | 0.6192 | 0.6354 |
| economics | 196 | 0.1589 | 0.2807 | 0.4623 | 0.4614 |
| hardware | 130 | 0.1497 | 0.1638 | 0.2835 | 0.2957 |
| law | 164 | 0.1066 | 0.1385 | 0.2753 | 0.2760 |
| medicalsciences | 139 | 0.1221 | 0.1513 | 0.2199 | 0.2264 |
| politics | 159 | 0.1609 | 0.3124 | 0.4086 | 0.4427 |
| psychology | 234 | 0.2222 | 0.3684 | 0.5445 | 0.5534 |
| robotics | 184 | 0.1826 | 0.3818 | 0.5773 | 0.5786 |
| sustainable_living | 235 | 0.2856 | 0.4493 | 0.5638 | 0.5773 |

## nDCG@10 by turn depth, dev (macro over domains)

| variant | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| current | 0.1819 | 0.2619 | 0.1794 | 0.1557 | 0.1290 | 0.1126 |
| q-only | 0.2921 | 0.2619 | 0.2932 | 0.3125 | 0.3126 | 0.2491 |
| last2 (gold answers) | 0.4341 | 0.2619 | 0.4979 | 0.5087 | 0.5006 | 0.3829 |
| history (gold answers) | 0.4377 | 0.2619 | 0.4979 | 0.5087 | 0.4978 | 0.4355 |

## nDCG@10 by turn depth, train (macro over domains)

| variant | all | T1 | T2 | T3 | T4 | T5+ |
|---|---|---|---|---|---|---|
| current | 0.1846 | 0.2641 | 0.1856 | 0.1551 | 0.1406 | 0.1445 |
| q-only | 0.2993 | 0.2641 | 0.2928 | 0.2927 | 0.3292 | 0.3507 |
| last2 (gold answers) | 0.4438 | 0.2641 | 0.5030 | 0.5175 | 0.5035 | 0.4635 |
| history (gold answers) | 0.4543 | 0.2641 | 0.5030 | 0.5175 | 0.5211 | 0.5121 |
