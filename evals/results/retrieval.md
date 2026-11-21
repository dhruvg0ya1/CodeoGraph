| Retrieval | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.40 | 0.75 | 0.80 | 0.537 | 0.565 | 1 ms |
| Dense only | 0.62 | 0.88 | 0.95 | 0.729 | 0.745 | 8 ms |
| **Hybrid, weighted 0.5 / 0.5** | 0.68 | 0.90 | 0.95 | **0.772** | 0.775 | 10 ms |
| Hybrid, weighted 0.6 dense / 0.4 BM25 (default) | 0.68 | 0.93 | 0.93 | 0.760 | 0.763 | 10 ms |
| Hybrid, weighted 0.7 / 0.3 | 0.68 | 0.93 | 0.93 | 0.757 | 0.757 | 10 ms |
| Hybrid, RRF | 0.62 | 0.90 | 0.97 | 0.728 | 0.750 | 10 ms |
| Hybrid, RRF + MS MARCO cross-encoder | 0.62 | 0.80 | 0.93 | 0.719 | 0.713 | 2913 ms |

40 questions on [`requests` v2.32.3](https://github.com/psf/requests/tree/v2.32.3) (695 chunks), embedder `BAAI/bge-small-en-v1.5` on a GitHub Actions CPU runner. Reproduce: `python evals/retrieval_eval.py`.
