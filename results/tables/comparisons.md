# Pre-declared comparisons, dev

Paired difference in macro nDCG@10, 10,000 bootstrap resamples of whole conversations within each domain, 95% percentile interval. Each comparison was fixed before the results it compares were seen.

| question | A | B | Δ nDCG@10 (A − B) | 95% CI | verdict |
|---|---|---|---|---|---|
| unadapted dense vs BM25, current turn | `bge-base-current` | `bm25-current` | +0.0535 | [+0.0285, +0.0775] | A better |
| unadapted dense vs BM25, q-only | `bge-base-qonly` | `bm25-qonly` | +0.0301 | [-0.0045, +0.0633] | no clear difference |
| unadapted dense vs BM25, history | `bge-base-history` | `bm25-history` | -0.0515 | [-0.0839, -0.0199] | B better |
| bigger encoder, q-only | `bge-base-qonly` | `minilm-qonly` | +0.0309 | [+0.0044, +0.0564] | A better |
| bigger encoder, history | `bge-base-history` | `minilm-history` | +0.0450 | [+0.0199, +0.0693] | A better |
| earlier questions help dense | `bge-base-qonly` | `bge-base-current` | +0.0868 | [+0.0640, +0.1099] | A better |
| gold answers on top of the questions | `bge-base-history` | `bge-base-qonly` | +0.0641 | [+0.0461, +0.0826] | A better |
| grid-selected vs default | `minilm-qonly-tuned` | `minilm-qonly` | — | — | the grid kept the default |
| grid-selected vs default | `minilm-history-tuned` | `minilm-history` | +0.0061 | [-0.0002, +0.0125] | no clear difference |
| grid-selected vs default | `bge-base-qonly-tuned` | `bge-base-qonly` | — | — | the grid kept the default |
| grid-selected vs default | `bge-base-history-tuned` | `bge-base-history` | +0.0012 | [-0.0014, +0.0042] | no clear difference |
| grid-selected k1, b vs official | `bm25-qonly-tuned` | `bm25-qonly` | +0.0095 | [+0.0039, +0.0150] | A better |
| grid-selected k1, b vs official | `bm25-history-tuned` | `bm25-history` | -0.0047 | [-0.0122, +0.0026] | no clear difference |
