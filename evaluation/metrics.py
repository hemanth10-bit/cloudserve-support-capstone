"""Metrics calculated by code from a run's results (Build Spec A10; Evaluation Framework tiers).

Every figure in the report is computed here, not by hand. Label-based figures are only computed for
tickets that carry labels, and the report says how many that was.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any

import numpy as np

from src.classify import MUST_ESCALATE, expected_calibration_error


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def _pct(x: float) -> float:
    return round(100 * x, 2)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def _per_class(true: list[str], pred: list[str]) -> dict[str, dict]:
    out = {}
    for c in sorted(set(true) | set(pred)):
        tp = sum(1 for t, p in zip(true, pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(true, pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(true, pred) if t == c and p != c)
        out[c] = {
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "support": tp + fn,
        }
    return out


def compute_metrics(
    results: list[dict],
    labels_by_id: dict[str, dict],
    input_count: int,
    dev_bodies: set[str] | None = None,
    bodies_by_id: dict[str, str] | None = None,
    decisions_logged: int | None = None,
    extra_governance: dict | None = None,
) -> dict[str, Any]:
    n = len(results)
    answered = [r for r in results if r["outcome"] == "answered"]
    escalated = [r for r in results if r["outcome"] == "escalated"]
    blocked = [r for r in results if (r.get("guardrail") or {}).get("blocked")]
    lat = [r["latency_ms"] for r in results]
    m: dict[str, Any] = {}

    m["volume"] = {
        "tickets_in_input": input_count,
        "tickets_processed": n,
        "reconciles": n == input_count,
        "answered_automatically": len(answered),
        "escalated": len(escalated),
        "blocked_by_guardrails": len(blocked),
        "tickets_with_errors": sum(1 for r in results if r["errors"]),
    }
    m["business"] = {
        "first_contact_resolution_pct": _pct(len(answered) / n) if n else None,
        "escalation_rate_pct": _pct(len(escalated) / n) if n else None,
        "mean_processing_seconds": round(statistics.mean(lat) / 1000, 4) if lat else None,
        "median_processing_seconds": round(statistics.median(lat) / 1000, 4) if lat else None,
        "note": (
            "FCR here = share of tickets the system answered without a person (answered / total). "
            "Processing time is the system's own latency, not the historical resolution time, which the "
            "data records differently. Neither measures real customer satisfaction."
        ),
    }

    # ------------------------------------------------------------ technical
    lab = [(r, labels_by_id[r["ticket_id"]]) for r in results if labels_by_id.get(r["ticket_id"])]
    tech: dict[str, Any] = {"labelled_tickets": len(lab)}
    if lab:
        yi = [l["intent"] for _, l in lab]
        pi = [r["intent"] for r, _ in lab]
        correct = [t == p for t, p in zip(yi, pi)]
        tech["intent_accuracy_pct"] = _pct(sum(correct) / len(lab))
        pc = _per_class(yi, pi)
        precs = [v["precision"] for v in pc.values() if v["precision"] is not None and v["support"] > 0]
        recs = [v["recall"] for v in pc.values() if v["recall"] is not None and v["support"] > 0]
        tech["intent_macro_precision_pct"] = _pct(float(np.mean(precs))) if precs else None
        tech["intent_macro_recall_pct"] = _pct(float(np.mean(recs))) if recs else None
        tech["intent_per_class"] = pc
        conf = np.array([r["intent_confidence"] for r, _ in lab])
        probs = np.zeros((len(lab), 2))
        probs[:, 1] = conf
        probs[:, 0] = 1 - conf
        # calibration of the top-class confidence: stated confidence vs observed accuracy
        tech["intent_calibration_error_pct"] = _pct(_ece_top(conf, np.array(correct)))
        tech["intent_mean_confidence"] = round(float(conf.mean()), 4)
        yu = [l["urgency"] for _, l in lab]
        pu = [r["urgency"] for r, _ in lab]
        tech["urgency_accuracy_pct"] = _pct(sum(a == b for a, b in zip(yu, pu)) / len(lab))
        tech["urgency_majority_baseline_pct"] = _pct(Counter(yu).most_common(1)[0][1] / len(lab))
        tech["urgency_calibration_error_pct"] = _pct(
            _ece_top(np.array([r["urgency_confidence"] for r, _ in lab]), np.array([a == b for a, b in zip(yu, pu)]))
        )
        route_ok = [r["route"] == l["expected_route"] for r, l in lab if l.get("expected_route")]
        tech["route_agreement_with_labels_pct"] = _pct(sum(route_ok) / len(route_ok)) if route_ok else None
        ans = [(r, l) for r, l in lab if l.get("answerable_from_docs")]
        if ans:
            hit = sum(1 for r, l in ans if set(l.get("expected_doc_ids") or []) & set(r["retrieved_doc_ids"][:5]))
            tech["retrieval_hit_rate_top5_pct"] = _pct(hit / len(ans))
            tech["retrieval_answerable_tickets"] = len(ans)
        # safety: any must-not ticket that was answered automatically
        mn = [(r, l) for r, l in lab if l.get("must_not_auto_respond")]
        tech["must_not_auto_respond_tickets"] = len(mn)
        tech["must_not_answered_automatically"] = sum(1 for r, _ in mn if r["outcome"] != "escalated")
        # split by whether the text also appears in the development set (validation overlaps development)
        if dev_bodies is not None and bodies_by_id is not None:
            seen = [c for (r, _), c in zip(lab, correct) if _norm(bodies_by_id.get(r["ticket_id"], "")) in dev_bodies]
            unseen = [
                c for (r, _), c in zip(lab, correct) if _norm(bodies_by_id.get(r["ticket_id"], "")) not in dev_bodies
            ]
            tech["overlap_with_development"] = {
                "tickets_whose_text_is_in_development": len(seen),
                "tickets_with_unseen_text": len(unseen),
                "intent_accuracy_seen_text_pct": _pct(sum(seen) / len(seen)) if seen else None,
                "intent_accuracy_unseen_text_pct": _pct(sum(unseen) / len(unseen)) if unseen else None,
                "note": "Scores on seen text are optimistic: the classifier was trained on those texts.",
            }
    tech["latency_ms_median"] = round(statistics.median(lat), 2) if lat else None
    tech["latency_ms_p95"] = round(_quantile(lat, 95), 2) if lat else None
    gen = [r for r in results if r.get("response")]
    drafted = [r for r in gen if r["response"].get("can_answer")]
    tech["generation"] = {
        "tickets_sent_to_generation": len(gen),
        "drafted_a_reply": len(drafted),
        "declined_or_rejected": len(gen) - len(drafted),
        "model_errors": sum(1 for r in gen if str(r["response"].get("error") or "").startswith("llm_unavailable")),
        "rejected_for_citations": sum(
            1 for r in gen if str(r["response"].get("error") or "").startswith(("invalid_citation", "no_citation"))
        ),
        "unparseable_output": sum(1 for r in gen if r["response"].get("error") == "unparseable_output"),
        "replies_with_an_uncited_sentence": sum(1 for r in drafted if r["response"].get("uncited_sentences")),
        "cached_replies": sum(1 for r in gen if r["response"].get("cached")),
        "handover_notes_from_model": sum(1 for r in results if (r.get("handover") or {}).get("source") == "model"),
        "handover_notes_from_template": sum(
            1 for r in results if (r.get("handover") or {}).get("source") == "template"
        ),
        "note": "Every cited id is checked in code against the retrieved passages. Whether each claim is supported by "
        "its source is checked by the validator (B-09).",
    }
    if lab:
        by_id = {r["ticket_id"]: l for r, l in lab}
        tech["generation"]["drafted_but_labelled_not_answerable"] = sum(
            1 for r in drafted if not by_id.get(r["ticket_id"], {}).get("answerable_from_docs", True)
        )
    tech["hallucination_rate_pct"] = None  # added with the validator (B-09)
    tech["citation_accuracy_pct"] = None
    m["technical"] = tech

    # ------------------------------------------------------------ governance
    m["governance"] = {
        "decisions_logged": decisions_logged,
        "decision_log": (extra_governance or {}).get("decision_log"),
        "guardrail_activations_by_type": (extra_governance or {}).get("guardrail_activations_by_type", {}),
        "private_data_detections": (extra_governance or {}).get("private_data_detections", 0),
        "must_escalate_classes": sorted(MUST_ESCALATE),
        "pending_components": sorted(
            {k for r in results for k, v in r["stage_status"].items() if v == "not_implemented"}
        ),
    }
    return m


def _ece_top(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    probs = np.stack([1 - conf, conf], axis=1)
    return expected_calibration_error(probs, correct.astype(int), bins)


def to_markdown(m: dict, run_info: dict) -> str:
    v, b, t, g = m["volume"], m["business"], m["technical"], m["governance"]
    lines = [
        "# Evaluation report",
        "",
        f"- Input: `{run_info['input']}`  |  Run at: {run_info['started_at']}"
        f"  |  Duration: {run_info['duration_seconds']} s",
        f"- Classifier version: {run_info.get('classifier_version', '')}",
        "",
    ]
    if g["pending_components"]:
        lines += [
            f"> **PROVISIONAL RUN.** Not yet implemented: {', '.join(g['pending_components'])}. "
            "'Answered automatically' is only the router's decision, with no generated reply behind it, and "
            "the router is a placeholder. Do not quote the business figures from this run.",
            "",
        ]
    lines += [
        "## Volume",
        f"- Tickets in input: {v['tickets_in_input']}; processed: {v['tickets_processed']} "
        f"(reconciles: {v['reconciles']})",
        f"- Answered automatically: {v['answered_automatically']}; escalated: {v['escalated']}; "
        f"blocked by guardrails: {v['blocked_by_guardrails']}; tickets with errors: {v['tickets_with_errors']}",
        "",
        "## Business",
        f"- First contact resolution: {b['first_contact_resolution_pct']}%"
        f"  |  Escalation rate: {b['escalation_rate_pct']}%",
        f"- Processing time: mean {b['mean_processing_seconds']} s, median {b['median_processing_seconds']} s",
        f"- {b['note']}",
        "",
        "## Technical",
    ]
    for k in (
        "labelled_tickets",
        "intent_accuracy_pct",
        "intent_macro_precision_pct",
        "intent_macro_recall_pct",
        "intent_calibration_error_pct",
        "urgency_accuracy_pct",
        "urgency_majority_baseline_pct",
        "route_agreement_with_labels_pct",
        "retrieval_hit_rate_top5_pct",
        "must_not_auto_respond_tickets",
        "must_not_answered_automatically",
        "latency_ms_median",
        "latency_ms_p95",
        "hallucination_rate_pct",
        "citation_accuracy_pct",
    ):
        if k in t:
            lines.append(f"- {k}: {t[k] if t[k] is not None else 'not yet measured'}")
    o = t.get("overlap_with_development")
    if o:
        lines += [
            "",
            f"- Overlap with development texts: {o['tickets_whose_text_is_in_development']} seen, "
            f"{o['tickets_with_unseen_text']} unseen. Intent accuracy: seen {o['intent_accuracy_seen_text_pct']}%, "
            f"unseen {o['intent_accuracy_unseen_text_pct']}%. {o['note']}",
        ]
    lines += [
        "",
        "## Governance",
        f"- Decisions logged: {g['decisions_logged'] if g['decisions_logged'] is not None else 'not yet measured'}"
        + (
            f" of {g['decision_log']['decisions_expected']} expected; reconciles with tickets processed: "
            f"{g['decision_log']['reconciles']}; missing: {g['decision_log']['missing_count']}; "
            f"write failures: {g['decision_log']['write_failures']}"
            if g.get("decision_log")
            else ""
        ),
        f"- Guardrail activations: {g['guardrail_activations_by_type'] or 'none recorded'}; "
        f"private data detections: {g['private_data_detections']}",
        f"- Components not yet implemented in this run: {g['pending_components'] or 'none'}",
        "",
    ]
    return "\n".join(lines)
