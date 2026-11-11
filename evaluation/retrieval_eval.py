"""Measure retrieval and choose the relevance threshold from DEVELOPMENT data (FR-03).

    python -m evaluation.retrieval_eval --input data/development_tickets.json --output evaluation/results/

Uses labels.answerable_from_docs and labels.expected_doc_ids. Never run threshold sweeps on
validation or hidden data: tuning there would measure fit, not generalisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.ingest import load_tickets
from src.retrieval import Retriever

THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.71, 0.05)]


def evaluate(path: str, k_values=(1, 3, 5)) -> dict:
    retriever = Retriever()
    rows = []
    for t in load_tickets(path):
        lab = t.labels or {}
        res = retriever.search(t.text, top_k=15, min_score=-1.0)  # no threshold: rank everything
        rows.append(
            {
                "ticket_id": t.ticket_id,
                "intent": lab.get("intent"),
                "answerable": bool(lab.get("answerable_from_docs")),
                "expected": list(lab.get("expected_doc_ids") or []),
                "ranked": res.doc_ids,
                "top_score": res.top_score,
                "error": res.error,
            }
        )
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    out: dict = {
        "n": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "errors": sum(1 for r in rows if r["error"]),
    }
    for k in k_values:
        hits = sum(1 for r in ans if set(r["expected"]) & set(r["ranked"][:k]))
        out[f"hit_rate_at_{k}_docs"] = round(hits / len(ans), 4)
    # reciprocal rank of the first expected doc
    rr = []
    for r in ans:
        pos = [i for i, d in enumerate(r["ranked"], 1) if d in r["expected"]]
        rr.append(1 / pos[0] if pos else 0.0)
    out["mrr"] = round(float(np.mean(rr)), 4)

    sweep = []
    for th in THRESHOLDS:
        ret_ans = [r for r in ans if r["top_score"] >= th]
        hit_ans = [r for r in ret_ans if set(r["expected"]) & set([d for d in r["ranked"][:5]])]
        false_ret = [r for r in unans if r["top_score"] >= th]
        sweep.append(
            {
                "threshold": th,
                "answerable_returned_something": round(len(ret_ans) / len(ans), 4),
                "answerable_hit_at_5_after_threshold": round(len(hit_ans) / len(ans), 4),
                "unanswerable_wrongly_returned": round(len(false_ret) / len(unans), 4),
            }
        )
    out["threshold_sweep"] = sweep
    by_intent: dict[str, list[float]] = {}
    for r in ans:
        by_intent.setdefault(r["intent"], []).append(1.0 if set(r["expected"]) & set(r["ranked"][:3]) else 0.0)
    out["hit_at_3_by_intent"] = {k: round(float(np.mean(v)), 3) for k, v in sorted(by_intent.items())}
    out["score_stats"] = {
        "answerable_top_score_median": round(float(np.median([r["top_score"] for r in ans])), 4),
        "unanswerable_top_score_median": round(float(np.median([r["top_score"] for r in unans])), 4),
    }
    out["_rows"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/development_tickets.json")
    ap.add_argument("--output", default="evaluation/results/")
    a = ap.parse_args()
    result = evaluate(a.input)
    rows = result.pop("_rows")
    outdir = Path(a.output)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "retrieval_eval.json").write_text(json.dumps(result, indent=2))
    (outdir / "retrieval_rows.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k not in ("threshold_sweep", "hit_at_3_by_intent")}, indent=2))
    print("threshold  ans_returned  ans_hit@5  unans_wrongly_returned")
    for s in result["threshold_sweep"]:
        print(
            f"{s['threshold']:>8}  {s['answerable_returned_something']:>12}  "
            f"{s['answerable_hit_at_5_after_threshold']:>9}  {s['unanswerable_wrongly_returned']:>10}"
        )
    print("hit@3 by intent:", json.dumps(result["hit_at_3_by_intent"]))


if __name__ == "__main__":
    main()
