"""Persistent decision log (FR-09, NFR-05, Build Spec A8, Governance Framework 'minimum record').

One row per automated decision: classification, routing, and (when they run) generation and validation.
Every ticket gets at least a classification row and a routing row, including tickets that fail, so the
log reconciles exactly with tickets processed.

Logging must never take a run down. If the database cannot be written, the row goes to a JSON Lines
fallback file beside it, the failure is counted, and reconcile() reports the gap rather than hiding it.

Privacy: input_summary holds the channel, tier, length, a short excerpt and a hash of the text. It never holds
the customer's name or id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Session

log = logging.getLogger(__name__)
EXCERPT_CHARS = 120


class Base(DeclarativeBase):
    pass


class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    decision_id = Column(String(40), unique=True, nullable=False)
    timestamp = Column(String(30), nullable=False)
    run_id = Column(String(40), index=True)
    ticket_id = Column(String(80), index=True, nullable=False)
    stage = Column(String(20), index=True, nullable=False)  # classification | routing | generation | validation
    input_summary = Column(Text)
    model_name = Column(String(120))
    model_version = Column(String(60))
    prediction_value = Column(String(120))
    prediction_confidence = Column(Float)
    alternatives = Column(JSON)
    sources_used = Column(JSON)
    threshold_applied = Column(Float)
    action_taken = Column(String(20))  # auto_respond | escalate | block
    reason = Column(Text)
    rule = Column(String(10))
    guardrail_results = Column(JSON)
    prompt_version = Column(String(40))
    requirement_ids = Column(JSON)


def input_summary(ticket) -> str:
    text = (getattr(ticket, "text", "") or "").replace("\n", " ")
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    excerpt = text[:EXCERPT_CHARS] + ("..." if len(text) > EXCERPT_CHARS else "")
    return (
        f"channel={ticket.channel}; tier={ticket.customer_tier}; chars={len(text)}; "
        f"sha256={digest}; excerpt={excerpt!r}"
    )


def expected_stages(result: dict) -> list[str]:
    stages = ["classification", "routing"]
    st = result.get("stage_status", {})
    if st.get("generate") not in (None, "not_implemented", "skipped"):
        stages.append("generation")
    if st.get("validate") not in (None, "not_implemented", "skipped"):
        stages.append("validation")
    return stages


def rows_for(
    ticket, result, run_id: str, classifier_version: str = "", router_version: str = "router-rules-v1"
) -> list[dict]:
    """Build the decision rows for one processed ticket from its TicketResult."""
    summary = input_summary(ticket)
    res = result if isinstance(result, dict) else result.to_dict()
    sources = [{"doc_id": p["doc_id"], "score": p["score"]} for p in res["retrieved_passages"]]
    classify_status = res.get("stage_status", {}).get("classify", "ok")
    rows = [
        {
            "stage": "classification",
            "input_summary": summary,
            "model_name": "minilm-embeddings+logistic-regression",
            "model_version": classifier_version or res.get("classifier_version", ""),
            "prediction_value": f"{res['intent']} / urgency={res['urgency']}",
            "prediction_confidence": res["intent_confidence"],
            "alternatives": [{"value": v, "confidence": c} for v, c in res.get("intent_alternatives", [])],
            "sources_used": [],
            "threshold_applied": None,
            "action_taken": "escalate" if classify_status == "fallback" else "continue",
            "reason": (
                f"Intent '{res['intent']}' (confidence {res['intent_confidence']}); urgency '{res['urgency']}' "
                f"(confidence {res['urgency_confidence']}); chance of an always-escalate class "
                f"{res['p_must_escalate']}."
                + (" Classification fell back to the default." if classify_status == "fallback" else "")
            ),
            "rule": None,
            "guardrail_results": None,
            "prompt_version": None,
            "requirement_ids": ["FR-02"],
        },
        {
            "stage": "routing",
            "input_summary": summary,
            "model_name": "rule-based router",
            "model_version": router_version,
            "prediction_value": res["route"],
            "prediction_confidence": res.get("p_auto"),
            "alternatives": [],
            "sources_used": sources,
            "threshold_applied": res["threshold_applied"],
            "action_taken": res["route"],
            "reason": res["route_reason"],
            "rule": res.get("route_rule"),
            "guardrail_results": None,
            "prompt_version": None,
            "requirement_ids": ["FR-04", "FR-05"],
        },
    ]
    if "generation" in expected_stages(res):
        g = res.get("response") or {}
        rows.append(
            {
                "stage": "generation",
                "input_summary": summary,
                "model_name": g.get("model"),
                "model_version": g.get("model_version", ""),
                "prediction_value": "can_answer" if g.get("can_answer") else "cannot_answer",
                "prediction_confidence": None,
                "alternatives": [],
                "sources_used": [{"doc_id": d, "score": None} for d in g.get("citations", [])],
                "threshold_applied": None,
                "action_taken": "auto_respond" if g.get("can_answer") else "escalate",
                "reason": (g.get("unsure_about") or "") + (f" [{g['error']}]" if g.get("error") else "")
                or "Reply drafted from the retrieved passages.",
                "rule": None,
                "guardrail_results": None,
                "prompt_version": g.get("prompt_version"),
                "requirement_ids": ["FR-06"],
            }
        )
    if "validation" in expected_stages(res):
        v = res.get("guardrail") or {}
        rows.append(
            {
                "stage": "validation",
                "input_summary": summary,
                "model_name": v.get("model"),
                "model_version": v.get("model_version", ""),
                "prediction_value": "blocked" if v.get("blocked") else "passed",
                "prediction_confidence": None,
                "alternatives": [],
                "sources_used": [],
                "threshold_applied": None,
                "action_taken": "block" if v.get("blocked") else res["route"],
                "reason": v.get("reason") or "All guardrail checks passed.",
                "rule": None,
                "guardrail_results": v.get("checks"),
                "prompt_version": v.get("prompt_version"),
                "requirement_ids": ["FR-07"],
            }
        )
    return rows


class DecisionLog:
    def __init__(self, url: str | None = None, run_id: str | None = None, reset: bool = False):
        from .config import get_settings

        self.url = url or get_settings().database_url
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
        self.write_failures = 0
        self._lock = threading.Lock()
        self.path = self._sqlite_path(self.url)
        if self.path is not None:
            self.url = f"sqlite:///{self.path}"  # absolute, so the working directory never matters
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if reset and self.path.exists():
                self.path.unlink()
        self.fallback_path = self.path.with_suffix(".fallback.jsonl") if self.path else Path("decisions.fallback.jsonl")
        self.engine = create_engine(self.url, future=True)
        Base.metadata.create_all(self.engine)

    @staticmethod
    def _sqlite_path(url: str) -> Path | None:
        if not url.startswith("sqlite:///"):
            return None
        from .config import ROOT

        p = Path(url[len("sqlite:///") :])
        return p if p.is_absolute() else (ROOT / p).resolve()

    # ------------------------------------------------------------------ write
    def record(self, ticket_id: str, **fields: Any) -> str:
        decision_id = uuid.uuid4().hex
        row = {
            "decision_id": decision_id,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "run_id": self.run_id,
            "ticket_id": ticket_id,
            **fields,
        }
        try:
            with self._lock, Session(self.engine) as s:
                s.add(Decision(**row))
                s.commit()
        except Exception as exc:  # logging must not stop the run; keep the row elsewhere and count the failure
            self.write_failures += 1
            log.error("decision log write failed (%s); row kept in %s", exc, self.fallback_path)
            try:
                with open(self.fallback_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
            except Exception:  # pragma: no cover - nothing more can be done
                log.error("fallback write failed too")
        return decision_id

    def record_ticket(self, ticket, result, classifier_version: str = "") -> list[str]:
        return [self.record(ticket.ticket_id, **r) for r in rows_for(ticket, result, self.run_id, classifier_version)]

    # ------------------------------------------------------------------- read
    def count(self, run_id: str | None = None) -> int:
        with Session(self.engine) as s:
            q = select(func.count()).select_from(Decision)
            if run_id or self.run_id:
                q = q.where(Decision.run_id == (run_id or self.run_id))
            return int(s.execute(q).scalar_one())

    def stages_by_ticket(self, run_id: str | None = None) -> dict[str, list[str]]:
        with Session(self.engine) as s:
            q = select(Decision.ticket_id, Decision.stage).where(Decision.run_id == (run_id or self.run_id))
            out: dict[str, list[str]] = {}
            for tid, stage in s.execute(q):
                out.setdefault(tid, []).append(stage)
            return out

    def fetch(self, ticket_id: str) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.execute(select(Decision).where(Decision.ticket_id == ticket_id).order_by(Decision.id)).scalars()
            return [{c.name: getattr(r, c.name) for c in Decision.__table__.columns} for r in rows]

    def reconcile(self, results: list[dict]) -> dict:
        """Compare the log with the tickets processed. Gaps are listed, never hidden. Never raises."""
        try:
            return self._reconcile(results)
        except Exception as exc:  # an unreadable log is reported as a failed reconciliation, not a crash
            return {
                "tickets_processed": len(results),
                "decisions_expected": sum(len(expected_stages(r)) for r in results),
                "decisions_logged": 0,
                "missing": [],
                "missing_count": len(results),
                "write_failures": self.write_failures,
                "reconciles": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _reconcile(self, results: list[dict]) -> dict:
        from collections import Counter

        logged = Counter()
        for tid, stages in self.stages_by_ticket().items():
            for st in stages:
                logged[(tid, st)] += 1
        wanted = Counter((r["ticket_id"], st) for r in results for st in expected_stages(r))
        missing = [
            {"ticket_id": t, "stage": st, "short_by": n - logged[(t, st)]}
            for (t, st), n in wanted.items()
            if logged[(t, st)] < n
        ]
        total = self.count()
        return {
            "tickets_processed": len(results),
            "decisions_expected": sum(wanted.values()),
            "decisions_logged": total,
            "missing": missing[:50],
            "missing_count": len(missing),
            "write_failures": self.write_failures,
            "reconciles": not missing and total == sum(wanted.values()),
        }
