"""TC-FR09: decision log (FR-09, NFR-05, A8)."""

import json
import sqlite3
from pathlib import Path

import pytest

from evaluation import harness
from src.decision_log import DecisionLog, expected_stages, input_summary
from src.ingest import normalise
from src.pipeline import Pipeline

DATA = Path(__file__).resolve().parent.parent / "data"
REQUIRED = {
    "decision_id",
    "timestamp",
    "ticket_id",
    "stage",
    "input_summary",
    "model_name",
    "model_version",
    "prediction_value",
    "prediction_confidence",
    "alternatives",
    "sources_used",
    "threshold_applied",
    "action_taken",
    "reason",
    "guardrail_results",
    "prompt_version",
    "requirement_ids",
}


@pytest.fixture(scope="module")
def pipe_and_log(tmp_path_factory):
    d = tmp_path_factory.mktemp("dlog")
    log = DecisionLog(f"sqlite:///{d / 'd.db'}", reset=True)
    return Pipeline(decision_log=log), log


def test_every_ticket_gets_classification_and_routing_rows(pipe_and_log):
    pipe, log = pipe_and_log
    raw = json.loads((DATA / "validation_tickets.json").read_text())[:10]
    results = [pipe.process(normalise(r)).to_dict() for r in raw]
    rec = log.reconcile(results)
    assert rec["reconciles"] and rec["missing_count"] == 0
    assert rec["decisions_logged"] == rec["decisions_expected"] == 20
    by = log.stages_by_ticket()
    assert all(sorted(by[r["ticket_id"]]) == ["classification", "routing"] for r in results)


def test_rows_carry_the_minimum_governance_fields(pipe_and_log):
    pipe, log = pipe_and_log
    raw = json.loads((DATA / "validation_tickets.json").read_text())[0]
    pipe.process(normalise(raw))
    rows = log.fetch(raw["ticket_id"])
    assert rows and all(REQUIRED <= set(r) for r in rows)
    routing = next(r for r in rows if r["stage"] == "routing")
    assert routing["action_taken"] in ("auto_respond", "escalate") and len(routing["reason"]) > 20
    assert (
        routing["rule"]
        and routing["threshold_applied"] is not None
        and routing["requirement_ids"] == ["FR-04", "FR-05"]
    )
    cls = next(r for r in rows if r["stage"] == "classification")
    assert len(cls["alternatives"]) == 3 and cls["requirement_ids"] == ["FR-02"]


def test_failed_and_malformed_tickets_are_logged_too(tmp_path):
    log = DecisionLog(f"sqlite:///{tmp_path / 'd.db'}", reset=True)
    pipe = Pipeline(decision_log=log)
    results = [
        pipe.process(r).to_dict() for r in [{"ticket_id": "A", "channel": "chat", "body": ""}, "garbage", None, {}]
    ]
    assert log.reconcile(results)["reconciles"]
    assert log.count() == 8


def test_log_does_not_store_customer_name_or_id(pipe_and_log):
    pipe, log = pipe_and_log
    raw = json.loads((DATA / "validation_tickets.json").read_text())[1]
    pipe.process(normalise(raw))
    blob = json.dumps(log.fetch(raw["ticket_id"]), default=str)
    assert raw["customer_name"] not in blob and raw["customer_id"] not in blob


def test_input_summary_is_short_and_hashed():
    t = normalise(
        {"ticket_id": "X", "channel": "email", "subject": "s", "body": "b" * 500, "customer_tier": "business"}
    )
    s = input_summary(t)
    assert "sha256=" in s and "tier=business" in s and len(s) < 300


def test_log_persists_across_restarts(tmp_path):
    url = f"sqlite:///{tmp_path / 'd.db'}"
    log = DecisionLog(url, run_id="r1", reset=True)
    Pipeline(decision_log=log).process(
        normalise({"ticket_id": "P-1", "channel": "chat", "body": "How do I export data?"})
    )
    reopened = DecisionLog(url, run_id="r1")
    assert reopened.count() == 2 and {r["stage"] for r in reopened.fetch("P-1")} == {"classification", "routing"}


def test_reconcile_reports_a_gap_instead_of_hiding_it(pipe_and_log, tmp_path):
    log = DecisionLog(f"sqlite:///{tmp_path / 'd.db'}", reset=True)
    pipe = Pipeline(decision_log=log)
    r = pipe.process(normalise({"ticket_id": "G-1", "channel": "chat", "body": "How do I export data?"})).to_dict()
    conn = sqlite3.connect(tmp_path / "d.db")
    conn.execute("delete from decisions where stage='routing'")
    conn.commit()
    conn.close()
    rec = log.reconcile([r])
    assert not rec["reconciles"] and rec["missing"][0]["stage"] == "routing"


def test_write_failure_does_not_stop_the_run_and_is_counted(tmp_path, monkeypatch):
    log = DecisionLog(f"sqlite:///{tmp_path / 'd.db'}", reset=True)
    pipe = Pipeline(decision_log=log)
    import src.decision_log as mod

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("disk full")

    monkeypatch.setattr(mod, "Session", Boom)
    res = pipe.process(normalise({"ticket_id": "W-1", "channel": "chat", "body": "How do I export data?"}))
    assert res.outcome in ("answered", "escalated")  # the ticket was still handled
    assert log.write_failures == 2
    lines = log.fallback_path.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["ticket_id"] == "W-1"


def test_expected_stages_follow_what_ran():
    assert expected_stages({"stage_status": {"generate": "not_implemented"}}) == ["classification", "routing"]
    assert expected_stages({"stage_status": {"generate": "ok", "validate": "ok"}}) == [
        "classification",
        "routing",
        "generation",
        "validation",
    ]


def test_harness_writes_a_log_that_reconciles(tmp_path):
    raw = json.loads((DATA / "validation_tickets.json").read_text())[:15] + ["junk", {"body": None}]
    p = tmp_path / "in.json"
    p.write_text(json.dumps(raw))
    assert harness.main(["--input", str(p), "--output", str(tmp_path / "out")]) == 0
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())["metrics"]["governance"]
    assert m["decisions_logged"] == 34 and m["decision_log"]["reconciles"] is True
    n = sqlite3.connect(tmp_path / "out" / "decisions.db").execute("select count(*) from decisions").fetchone()[0]
    assert n == 34


def test_running_the_harness_twice_does_not_double_count(tmp_path):
    raw = json.loads((DATA / "validation_tickets.json").read_text())[:5]
    p = tmp_path / "in.json"
    p.write_text(json.dumps(raw))
    for _ in range(2):
        harness.main(["--input", str(p), "--output", str(tmp_path / "out")])
    n = sqlite3.connect(tmp_path / "out" / "decisions.db").execute("select count(*) from decisions").fetchone()[0]
    assert n == 10


def test_classification_row_does_not_claim_an_action(pipe_and_log):
    pipe, log = pipe_and_log
    raw = json.loads((DATA / "validation_tickets.json").read_text())[2]
    pipe.process(normalise(raw))
    cls = next(r for r in log.fetch(raw["ticket_id"]) if r["stage"] == "classification")
    assert cls["action_taken"] == "continue"
