"""Retrieval benchmark: does search put the right function first?

40 hand-written questions about psf/requests (pinned at v2.32.3), each with the function(s) that answer it.
Most questions describe behaviour in plain English without naming the function, so keyword overlap alone is not
enough. Every configuration is scored on the same index:

    Recall@1 / @5 / @10   a correct function in the top 1 / 5 / 10
    MRR@10                mean reciprocal rank of the first correct function
    nDCG@10               ranking quality with several acceptable answers
    p50 ms                median query latency on the runner's CPU

    python evals/retrieval_eval.py                         # default embedder, all configs
    python evals/retrieval_eval.py --models BAAI/bge-small-en-v1.5,jinaai/jina-embeddings-v2-base-code
    python evals/retrieval_eval.py --write-readme --min-recall5 0.8     # CI: update README, fail on regression
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

CONFIGS = {
    "BM25 only": dict(mode="bm25", rerank=False),
    "Dense only": dict(mode="dense", rerank=False),
    "Hybrid, weighted 0.6/0.4 (v1)": dict(mode="hybrid", fusion="weighted", rerank=False),
    "Hybrid, RRF": dict(mode="hybrid", fusion="rrf", rerank=False),
    "Hybrid, RRF + cross-encoder rerank": dict(mode="hybrid", fusion="rrf", rerank=True),
}


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkout(url: str, ref: str, cache: Path) -> Path:
    dst = cache / f"{re.sub(r'[^a-z0-9]+', '-', url.lower())}-{ref}"
    if not dst.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", ref, url, str(dst)], check=True)
    return dst


def score(ranked: list[str], targets: set[str]) -> dict:
    first = next((i for i, n in enumerate(ranked[:10]) if n in targets), None)
    dcg = sum(1 / math.log2(i + 2) for i, n in enumerate(ranked[:10]) if n in targets)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(targets), 10)))
    return {"r1": first == 0, "r5": first is not None and first < 5, "r10": first is not None,
            "mrr": 0.0 if first is None else 1 / (first + 1), "ndcg": dcg / idcg if idcg else 0.0, "rank": first}


def run(models: list[str], dataset: Path, url: str, ref: str, cache: Path) -> list[dict]:
    from codeograph import indexer, search, store

    repo = checkout(url, ref, cache)
    data = load_dataset(dataset)
    results = []
    for model in models:
        os.environ["CODEOGRAPH_EMBED_MODEL"] = model
        ns = "eval-" + re.sub(r"[^a-z0-9]+", "-", f"{Path(url).name}-{model}".lower())
        t0 = time.time()
        meta = indexer.index_repository(str(repo), namespace=ns, force=True)
        index_s = time.time() - t0
        idx = store.load(ns)
        names = {f"{c.file_path}::{c.qualified_name}" for c in idx.chunks}
        missing = sorted({t for q in data for t in q["targets"]} - names)
        if missing:
            raise SystemExit(f"dataset targets not in the index (fix the dataset): {missing}")
        search.search(idx, "warm up", k=5)                 # load models before timing
        for label, cfg in CONFIGS.items():
            rows, lat = [], []
            for q in data:
                t = time.perf_counter()
                hits = search.search(idx, q["question"], k=10, **cfg)
                lat.append((time.perf_counter() - t) * 1000)
                ranked = [f"{h.chunk.file_path}::{h.chunk.qualified_name}" for h in hits]
                rows.append({"id": q["id"], **score(ranked, set(q["targets"])), "top": ranked[:3]})
            n = len(rows)
            results.append({
                "model": model, "config": label, "questions": n, "chunks": meta["chunk_count"],
                "index_seconds": round(index_s, 1),
                "recall@1": round(sum(r["r1"] for r in rows) / n, 3),
                "recall@5": round(sum(r["r5"] for r in rows) / n, 3),
                "recall@10": round(sum(r["r10"] for r in rows) / n, 3),
                "mrr@10": round(sum(r["mrr"] for r in rows) / n, 3),
                "ndcg@10": round(sum(r["ndcg"] for r in rows) / n, 3),
                "p50_ms": round(statistics.median(lat), 1),
                "misses": [r["id"] for r in rows if not r["r5"]],
            })
            print(f"{model:40s} {label:38s} R@1 {results[-1]['recall@1']:.3f}  R@5 {results[-1]['recall@5']:.3f}  "
                  f"MRR {results[-1]['mrr@10']:.3f}  p50 {results[-1]['p50_ms']} ms", flush=True)
    return results


def table(results: list[dict], url: str, ref: str) -> str:
    multi = len({r["model"] for r in results}) > 1
    head = ("| Embedder " if multi else "") + "| Retrieval | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | p50 latency |"
    sep = ("|---" if multi else "") + "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    best = max(r["mrr@10"] for r in results)
    for r in results:
        b = "**" if r["mrr@10"] == best else ""
        lines.append((f"| `{r['model']}` " if multi else "") +
                     f"| {b}{r['config']}{b} | {r['recall@1']:.2f} | {r['recall@5']:.2f} | {r['recall@10']:.2f} | "
                     f"{b}{r['mrr@10']:.3f}{b} | {r['ndcg@10']:.3f} | {r['p50_ms']:.0f} ms |")
    r0 = results[0]
    where = "a GitHub Actions CPU runner" if os.environ.get("GITHUB_ACTIONS") else "a laptop CPU"
    lines.append("")
    lines.append(f"{r0['questions']} questions on [`{Path(url).name}` {ref}]({url}/tree/{ref}) ({r0['chunks']} chunks), "
                 f"embedder `{r0['model']}` on {where}. Reproduce: `python evals/retrieval_eval.py`.")
    return "\n".join(lines)


def write_readme(md: str) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = "<!-- RETRIEVAL_EVAL:START -->", "<!-- RETRIEVAL_EVAL:END -->"
    if start in text and end in text:
        text = text.split(start)[0] + f"{start}\n{md}\n{end}" + text.split(end, 1)[1]
        readme.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("CODEOGRAPH_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    ap.add_argument("--dataset", default=str(HERE / "datasets" / "requests_v2.32.3.jsonl"))
    ap.add_argument("--repo", default="https://github.com/psf/requests")
    ap.add_argument("--ref", default="v2.32.3")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "codeograph-evals"))
    ap.add_argument("--out", default=str(HERE / "results" / "retrieval.json"))
    ap.add_argument("--write-readme", action="store_true")
    ap.add_argument("--min-recall5", type=float, default=0.0,
                    help="exit 1 if the default configuration's Recall@5 falls below this")
    a = ap.parse_args()
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    results = run([m.strip() for m in a.models.split(",") if m.strip()], Path(a.dataset), a.repo, a.ref, Path(a.cache))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    md = table(results, a.repo, a.ref)
    (out.parent / "retrieval.md").write_text(md + "\n")
    print("\n" + md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("## Retrieval eval\n\n" + md + "\n")
    if a.write_readme:
        write_readme(md)
    default = next(r for r in results if r["config"] == "Hybrid, RRF + cross-encoder rerank")
    if default["recall@5"] < a.min_recall5:
        raise SystemExit(f"Recall@5 {default['recall@5']} is below the gate {a.min_recall5}")


if __name__ == "__main__":
    main()
