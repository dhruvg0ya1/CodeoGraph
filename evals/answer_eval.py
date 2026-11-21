"""Answer benchmark: are the agent's answers grounded in the code it retrieved, and do they cite the right place?

Runs the LangGraph agent on questions from the retrieval set and scores every answer two ways:

  deterministic   cited the expected function (target hit), has at least one valid citation, citations removed
                  by the verify step (pointed at a source that was not retrieved), latency
  RAGAS           faithfulness (claims supported by the cited code) and context precision (were the cited
                  sources useful), judged by an LLM

Needs an LLM. In CI it uses GitHub Models through the workflow's own token (CODEOGRAPH_LLM=github), so it runs
with no paid key; locally any provider in codeograph/llm.py works.

    CODEOGRAPH_LLM=github GITHUB_TOKEN=... python evals/answer_eval.py --n 12
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def main() -> None:
    from retrieval_eval import checkout, load_dataset, write_readme  # noqa: F401  (same repo, same cache)

    from codeograph import agent, indexer, llm, store

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--dataset", default=str(HERE / "datasets" / "requests_v2.32.3.jsonl"))
    ap.add_argument("--repo", default="https://github.com/psf/requests")
    ap.add_argument("--ref", default="v2.32.3")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "codeograph-evals"))
    ap.add_argument("--no-ragas", action="store_true")
    ap.add_argument("--write-readme", action="store_true")
    a = ap.parse_args()

    model = llm.get_model()
    if model is None:
        raise SystemExit("answer_eval needs an LLM - set CODEOGRAPH_LLM (github, gemini or openai).")
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    repo = checkout(a.repo, a.ref, Path(a.cache))
    ns = "eval-answers-requests"
    indexer.index_repository(str(repo), namespace=ns)
    idx = store.load(ns)
    data = load_dataset(Path(a.dataset))[: a.n]

    rows, samples = [], []
    for q in data:
        t = time.perf_counter()
        r = agent.ask(q["question"], ns)
        secs = time.perf_counter() - t
        cited = [f"{c['file_path']}::{c['symbol_name']}" for c in r["citations"] if c.get("cited")]
        contexts = []
        for c in r["citations"][:4]:
            ch = next((x for x in idx.chunks if x.file_path == c["file_path"] and x.qualified_name == c["symbol_name"]),
                      None)
            if ch:
                contexts.append(f"{ch.file_path} `{ch.qualified_name}`\n{ch.code[:2500]}")
        rows.append({"id": q["id"], "target_cited": bool(set(cited) & set(q["targets"])), "has_citation": bool(cited),
                     "removed_citations": len(r["invalid_citations"]), "seconds": round(secs, 1),
                     "answer": r["answer"][:600], "cited": cited})
        samples.append({"user_input": q["question"], "response": r["answer"], "retrieved_contexts": contexts or [""]})
        print(f"{q['id']} target_cited={rows[-1]['target_cited']} cites={len(cited)} {secs:.1f}s", flush=True)

    n = len(rows)
    summary = {
        "model": llm.model_label(), "questions": n,
        "target_cited": round(sum(r["target_cited"] for r in rows) / n, 3),
        "has_citation": round(sum(r["has_citation"] for r in rows) / n, 3),
        "removed_citations": sum(r["removed_citations"] for r in rows),
        "p50_seconds": round(statistics.median(r["seconds"] for r in rows), 1),
    }

    if not a.no_ragas:
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference

        res = evaluate(EvaluationDataset.from_list(samples),
                       metrics=[Faithfulness(), LLMContextPrecisionWithoutReference()],
                       llm=LangchainLLMWrapper(model),
                       run_config=RunConfig(max_workers=2, max_retries=8, max_wait=90, timeout=180),
                       show_progress=False)
        df = res.to_pandas()
        for col, key in (("faithfulness", "ragas_faithfulness"),
                         ("llm_context_precision_without_reference", "ragas_context_precision")):
            if col in df:
                vals = [float(v) for v in df[col] if v == v]       # drop NaN (a judge call that failed)
                summary[key] = round(sum(vals) / len(vals), 3) if vals else None
                summary[key + "_scored"] = len(vals)

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "answers.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    md = ("| Metric | Value |\n|---|---|\n"
          f"| Answers that cite the expected function | {summary['target_cited']:.0%} |\n"
          f"| Answers with at least one valid citation | {summary['has_citation']:.0%} |\n"
          f"| Citations removed by the verify step | {summary['removed_citations']} |\n"
          + (f"| RAGAS faithfulness | {summary['ragas_faithfulness']:.3f} |\n"
             if summary.get("ragas_faithfulness") is not None else "")
          + (f"| RAGAS context precision | {summary['ragas_context_precision']:.3f} |\n"
             if summary.get("ragas_context_precision") is not None else "")
          + f"| Median time per answer | {summary['p50_seconds']} s |\n\n"
          f"{n} questions from the retrieval set, answered by the agent with `{summary['model']}`, which also "
          f"judges the RAGAS metrics. Reproduce: `python evals/answer_eval.py`.")
    (out / "answers.md").write_text(md + "\n")
    print("\n" + md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("## Answer eval\n\n" + md + "\n")
    if a.write_readme:
        readme = ROOT / "README.md"
        text = readme.read_text()
        s, e = "<!-- ANSWER_EVAL:START -->", "<!-- ANSWER_EVAL:END -->"
        if s in text and e in text:
            readme.write_text(text.split(s)[0] + f"{s}\n{md}\n{e}" + text.split(e, 1)[1])


if __name__ == "__main__":
    main()
