"""TC-FR04 and TC-FR05: routing and the always-escalate rules (FR-04, FR-05, A5)."""

import json
from pathlib import Path

import pytest

from src.classify import MUST_ESCALATE, Classification, Classifier
from src.ingest import normalise
from src.pipeline import Pipeline
from src.retrieval import Passage, RetrievalResult
from src.router import ROUTER_PATH, Router

DATA = Path(__file__).resolve().parent.parent / "data"
RATES = {"data_export": 0.9, "database_issue": 0.4, "rate_limit": 0.87, "security_incident": 0.0}


def cls(intent="data_export", conf=0.95, pm=0.01, fallback=False, probs=None):
    return Classification(
        intent=intent,
        intent_confidence=conf,
        urgency="medium",
        urgency_confidence=0.5,
        p_must_escalate=pm,
        intent_probs=probs or {intent: conf},
        fallback=fallback,
        error="boom" if fallback else None,
    )


def ret(n=1, error=None):
    ps = [Passage("DOC-DATA-001#symptoms", "DOC-DATA-001", "Export", "symptoms", "text", 0.7)] * n
    return RetrievalResult(ps, [("DOC-DATA-001", 0.7)] if n else [], 0.7 if n else 0.0, error)


@pytest.fixture
def router():
    return Router(rates=RATES, threshold=0.64, p_must_limit=0.2)


def test_auto_respond_when_everything_supports_it(router):
    d = router.decide(cls(), ret())
    assert d.route == "auto_respond" and d.rule == "AUTO" and "DOC-DATA-001" in d.reason


def test_R1_classification_unavailable_escalates(router):
    d = router.decide(cls(fallback=True), ret())
    assert d.route == "escalate" and d.rule == "R1"


@pytest.mark.parametrize("intent", sorted(MUST_ESCALATE))
def test_R2_always_escalate_classes_never_auto_respond(router, intent):
    d = router.decide(cls(intent=intent, conf=0.99), ret())
    assert d.route == "escalate" and d.rule == "R2" and "person" in d.reason


def test_R3_probability_mass_on_escalate_classes_escalates(router):
    d = router.decide(cls(intent="data_export", conf=0.7, pm=0.25), ret())
    assert d.route == "escalate" and d.rule == "R3"


def test_R4_retrieval_failure_or_nothing_found_escalates(router):
    assert router.decide(cls(), ret(0)).rule == "R4"
    assert router.decide(cls(), ret(0, error="index_unavailable")).rule == "R4"


def test_R5_low_answerability_escalates_and_says_why(router):
    d = router.decide(cls(intent="database_issue", conf=0.95), ret())
    assert d.route == "escalate" and d.rule == "R5" and "38%" in d.reason and "64%" in d.reason


def test_threshold_boundary_is_inclusive(router):
    c = cls(intent="data_export", conf=1.0, probs={"data_export": 0.64 / 0.9})
    assert abs(router.p_auto(c) - 0.64) < 1e-3
    assert (
        Router(rates={"data_export": 1.0}, threshold=0.5).decide(cls(probs={"data_export": 0.5}), ret()).route
        == "auto_respond"
    )


def test_p_auto_mixes_the_distribution(router):
    c = cls(intent="data_export", probs={"data_export": 0.5, "database_issue": 0.5})
    assert router.p_auto(c) == pytest.approx(0.65, abs=1e-3)


def test_same_input_same_decision(router):
    a = router.decide(cls(), ret())
    b = router.decide(cls(), ret())
    assert a == b


def test_every_decision_has_a_readable_reason(router):
    for c, r in [
        (cls(), ret()),
        (cls(fallback=True), ret()),
        (cls(intent="security_incident"), ret()),
        (cls(), ret(0)),
    ]:
        d = router.decide(c, r)
        assert len(d.reason) > 20 and d.reason[0].isupper() and d.reason.endswith(".")


def test_missing_router_file_falls_back_to_safe_defaults(monkeypatch, tmp_path):
    import src.router as mod

    monkeypatch.setattr(mod, "ROUTER_PATH", tmp_path / "none.json")
    r = Router()
    assert r.rates == {} and r.decide(cls(), ret()).route == "escalate"  # no rates known -> p_auto 0 -> escalate


def test_fitted_config_is_documented():
    cfg = json.loads(ROUTER_PATH.read_text())
    assert 0.5 <= cfg["threshold"] < 1 and "W" in cfg["rule"] and cfg["filter_pass_through_ASSUMED"]
    assert all(cfg["answerable_rates"][i] == 0.0 for i in MUST_ESCALATE)


# ---- end to end with the real classifier and retriever ----
@pytest.fixture(scope="module")
def pipe():
    return Pipeline()


@pytest.mark.parametrize("name", ["development_tickets.json", "validation_tickets.json"])
def test_no_always_escalate_ticket_is_ever_auto_answered(pipe, name):
    raw = json.loads((DATA / name).read_text())
    bad = []
    for r in raw:
        if r["labels"]["must_not_auto_respond"]:
            res = pipe.process(normalise(r))
            if res.route != "escalate":
                bad.append(res.ticket_id)
    assert bad == []


def test_pipeline_routing_is_deterministic(pipe):
    raw = json.loads((DATA / "validation_tickets.json").read_text())[:15]
    a = [(x.route, x.route_rule, x.route_reason) for x in (pipe.process(normalise(r)) for r in raw)]
    b = [(x.route, x.route_rule, x.route_reason) for x in (pipe.process(normalise(r)) for r in raw)]
    assert a == b


def test_billing_commitment_and_data_location_claims_are_not_auto_answered_blindly(pipe):
    # Open question 4 in the PRD: until decided, these are covered by intent rates, not a separate rule.
    res = pipe.process(
        normalise(
            {
                "ticket_id": "B-1",
                "channel": "email",
                "subject": "Refund",
                "body": "Please confirm you will refund last month's invoice in full.",
            }
        )
    )
    assert res.route in ("auto_respond", "escalate") and res.route_reason


def test_classifier_unavailable_escalates_everything(tmp_path):
    clf = Classifier.__new__(Classifier)
    clf.art, clf.error = None, "no model"
    res = Pipeline(classifier=clf).process(normalise({"ticket_id": "X", "channel": "chat", "body": "hello"}))
    assert res.route == "escalate" and res.route_rule == "R1"
