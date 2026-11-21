| Embedder | Retrieval | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency |
|---|---|---|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | BM25 only | 0.40 | 0.75 | 0.80 | 0.537 | 0.565 | 0 ms |
| `BAAI/bge-small-en-v1.5` | Dense only | 0.62 | 0.88 | 0.95 | 0.729 | 0.745 | 6 ms |
| `BAAI/bge-small-en-v1.5` | **Hybrid, weighted 0.5 / 0.5** | 0.68 | 0.90 | 0.95 | **0.772** | 0.775 | 7 ms |
| `BAAI/bge-small-en-v1.5` | Hybrid, weighted 0.6 dense / 0.4 BM25 (default) | 0.68 | 0.93 | 0.93 | 0.760 | 0.763 | 7 ms |
| `BAAI/bge-small-en-v1.5` | Hybrid, weighted 0.7 / 0.3 | 0.68 | 0.93 | 0.93 | 0.757 | 0.757 | 7 ms |
| `BAAI/bge-small-en-v1.5` | Hybrid, RRF | 0.62 | 0.90 | 0.97 | 0.728 | 0.750 | 7 ms |
| `BAAI/bge-small-en-v1.5` | Hybrid, RRF + MS MARCO cross-encoder | 0.62 | 0.80 | 0.93 | 0.719 | 0.713 | 2123 ms |
| `jinaai/jina-embeddings-v2-base-code` | BM25 only | 0.40 | 0.75 | 0.80 | 0.537 | 0.565 | 0 ms |
| `jinaai/jina-embeddings-v2-base-code` | Dense only | 0.65 | 0.88 | 0.90 | 0.748 | 0.765 | 49 ms |
| `jinaai/jina-embeddings-v2-base-code` | Hybrid, weighted 0.5 / 0.5 | 0.62 | 0.93 | 0.93 | 0.735 | 0.749 | 65 ms |
| `jinaai/jina-embeddings-v2-base-code` | **Hybrid, weighted 0.6 dense / 0.4 BM25 (default)** | 0.70 | 0.88 | 0.95 | **0.772** | 0.796 | 37 ms |
| `jinaai/jina-embeddings-v2-base-code` | Hybrid, weighted 0.7 / 0.3 | 0.68 | 0.88 | 0.93 | 0.762 | 0.782 | 50 ms |
| `jinaai/jina-embeddings-v2-base-code` | Hybrid, RRF | 0.62 | 0.82 | 0.90 | 0.725 | 0.738 | 50 ms |
| `jinaai/jina-embeddings-v2-base-code` | Hybrid, RRF + MS MARCO cross-encoder | 0.62 | 0.82 | 0.93 | 0.722 | 0.716 | 2194 ms |

40 questions on [`requests` v2.32.3](https://github.com/psf/requests/tree/v2.32.3) (695 chunks), embedder `BAAI/bge-small-en-v1.5` on a GitHub Actions CPU runner. Reproduce: `python evals/retrieval_eval.py`.
