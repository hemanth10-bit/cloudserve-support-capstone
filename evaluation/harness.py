"""Unattended evaluation run (FR-12, Build Spec A9 and A10).

    python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/

- Takes ANY file with the ticket schema (the hidden set is a file we have never seen).
- Processes every ticket; a ticket that misbehaves is escalated and recorded, never skipped.
- Writes results incrementally (results.jsonl), then metrics.json and metrics.md with no manual step.
- Exit code 0 means the run completed and reconciled; 1 means it completed but counts did not reconcile;
  2 means the input file could not be read at all.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.decision_log import DecisionLog
from src.ingest import load_raw_tickets, normalise
from src.pipeline import Pipeline

from .metrics import compute_metrics, to_markdown

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("harness")


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def _dev_bodies() -> set[str] | None:
    p = ROOT / "data" / "development_tickets.json"
    try:
        return {_norm(t.get("body", "")) for t in json.loads(p.read_text())}
    except Exception:
        return None


def run(input_path: str, output_dir: str, limit: int | None = None) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    raw = load_raw_tickets(input_path)  # raises ValueError if the file is unreadable
    if limit:
        raw = raw[:limit]
    dlog = DecisionLog(f"sqlite:///{(out / 'decisions.db').resolve()}", reset=True)
    pipeline = Pipeline(decision_log=dlog)

    results: list[dict] = []
    labels_by_id: dict[str, dict] = {}
    bodies_by_id: dict[str, str] = {}
    results_path = out / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for i, item in enumerate(raw, 1):
            ticket = normalise(item)
            res = pipeline.process(ticket).to_dict()
            results.append(res)
            if ticket.labels:
                labels_by_id[ticket.ticket_id] = ticket.labels
            bodies_by_id[ticket.ticket_id] = ticket.body
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(raw):
                print(f"  processed {i}/{len(raw)}", flush=True)

    reconciliation = dlog.reconcile(results)
    metrics = compute_metrics(
        results,
        labels_by_id,
        input_count=len(raw),
        dev_bodies=_dev_bodies(),
        bodies_by_id=bodies_by_id,
        decisions_logged=reconciliation["decisions_logged"],
        extra_governance={
            "decision_log": reconciliation,
            "guardrail_activations_by_type": dict(
                Counter(c for r in results for c in ((r.get("guardrail") or {}).get("failed_checks") or []))
            ),
            "private_data_detections": sum(
                1 for r in results if "private_data" in ((r.get("guardrail") or {}).get("failed_checks") or [])
            ),
        },
    )
    run_info = {
        "input": str(input_path),
        "output": str(output_dir),
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_seconds": round(time.perf_counter() - t0, 2),
        "classifier_version": results[0]["classifier_version"] if results else "",
    }
    (out / "metrics.json").write_text(json.dumps({"run": run_info, "metrics": metrics}, indent=2, ensure_ascii=False))
    (out / "metrics.md").write_text(to_markdown(metrics, run_info))
    return {"run": run_info, "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the full pipeline over a ticket file and write a metrics report.")
    ap.add_argument("--input", required=True, help="path to a JSON ticket file (same schema as the dataset)")
    ap.add_argument("--output", required=True, help="directory for results.jsonl, metrics.json and metrics.md")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N tickets (testing)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        result = run(a.input, a.output, a.limit)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    v = result["metrics"]["volume"]
    print(
        f"Done: {v['tickets_processed']}/{v['tickets_in_input']} processed, "
        f"{v['answered_automatically']} answered, {v['escalated']} escalated, "
        f"{v['blocked_by_guardrails']} blocked. Report: {Path(a.output) / 'metrics.md'}"
    )
    log_ok = (result["metrics"]["governance"].get("decision_log") or {}).get("reconciles", False)
    return 0 if v["reconciles"] and log_ok else 1


if __name__ == "__main__":
    sys.exit(main())
