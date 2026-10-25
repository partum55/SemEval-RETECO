# First error analysis: error types

30 dev turns from `scripts/sample_errors.py`, labelled by hand in [`labels.csv`](../error_analysis/labels.csv) (one main cause per turn; for 'dense >> BM25' and 'BM25 >> dense' the label explains why the weaker retriever failed).

| error type | everything fails | dense >> BM25 | BM25 >> dense | history fixes it | history no help | total |
|---|---|---|---|---|---|---|
| near-duplicate wrong doc | 3 | 4 | 1 |  |  | 8 |
| reference / pronoun |  |  | 3 | 5 |  | 8 |
| needs reasoning | 2 |  |  |  | 2 | 4 |
| vocabulary mismatch |  | 1 |  |  | 3 | 4 |
| topic shift | 1 | 1 | 1 |  |  | 3 |
| annotation issue |  |  | 1 | 1 | 1 | 3 |
