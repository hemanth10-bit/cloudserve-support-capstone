"""TC-FR02: classification (FR-02, A3, A5)."""

import json
from pathlib import Path

import pytest

from src.classify import MUST_ESCALATE, Classifier, expected_calibration_error, fit_temperature, text_group
from src.ingest import normalise

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def clf():
    return Classifier()


@pytest.fixture(scope="module")
def dev():
    return json.loads((DATA / "development_tickets.json").read_text())


def test_every_output_has_class_and_confidence_in_range(clf, dev):
    tickets = [normalise(r) for r in dev[:40]]
    for c in clf.predict_many(tickets):
        assert c.intent and c.urgency in ("low", "medium", "high")
        assert 0.0 <= c.intent_confidence <= 1.0 and 0.0 <= c.urgency_confidence <= 1.0
        assert 0.0 <= c.p_must_escalate <= 1.0
        assert not c.fallback


def test_alternatives_are_recorded_and_ordered(clf, dev):
    c = clf.predict(normalise(dev[0]))
    assert len(c.intent_alternatives) == 3
    assert c.intent_alternatives[0][0] == c.intent
    confs = [p for _, p in c.intent_alternatives]
    assert confs == sorted(confs, reverse=True)
    assert len(c.urgency_alternatives) == 3


def test_deterministic(clf, dev):
    t = normalise(dev[7])
    assert clf.predict(t).to_dict() == clf.predict(t).to_dict()


def test_labels_are_never_used_at_prediction_time(clf, dev):
    with_labels = normalise(dev[3])
    without = normalise({k: v for k, v in dev[3].items() if k != "labels"})
    assert clf.predict(with_labels).to_dict() == clf.predict(without).to_dict()


def test_security_text_puts_mass_on_always_escalate_classes(clf):
    t = normalise(
        {
            "ticket_id": "S-1",
            "channel": "email",
            "subject": "Possible account compromise",
            "body": "We think someone outside our team has accessed our admin account and changed settings.",
        }
    )
    c = clf.predict(t)
    assert c.intent in MUST_ESCALATE and c.p_must_escalate > 0.5


def test_empty_text_returns_defined_fallback_not_exception(clf):
    c = clf.predict(normalise({"ticket_id": "E-1", "channel": "chat", "body": ""}))
    assert c.fallback and c.intent == "unclear_request" and c.intent_confidence == 0.0 and c.p_must_escalate == 1.0


def test_model_failure_returns_fallback(clf, monkeypatch):
    import src.classify as mod

    monkeypatch.setattr(mod, "embed", lambda texts: (_ for _ in ()).throw(RuntimeError("embedding service down")))
    c = clf.predict(normalise({"ticket_id": "F-1", "channel": "chat", "body": "hello there"}))
    assert c.fallback and "embedding service down" in c.error


def test_missing_model_and_no_training_data_degrades(monkeypatch, tmp_path):
    import src.classify as mod

    monkeypatch.setattr(mod, "MODEL_PATH", tmp_path / "none.joblib")
    c = Classifier(dev_path=tmp_path / "does_not_exist.json")
    out = c.predict(normalise({"ticket_id": "M-1", "channel": "chat", "body": "help"}))
    assert out.fallback and "classifier_unavailable" in out.error


def test_group_key_is_same_for_same_body():
    assert text_group("Hello,  World!") == text_group("hello world")


def test_calibration_helpers():
    import numpy as np

    logits = np.array([[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]])
    y = np.array([0, 1, 1, 0])  # overconfident and half wrong -> temperature should flatten
    assert fit_temperature(logits, y) > 1.0
    perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(perfect, np.array([0, 1])) == 0.0


def test_training_report_meets_documented_targets():
    rep = json.loads((Path(__file__).resolve().parent.parent / "models" / "classifier_report.json").read_text())
    assert rep["intent"]["oof_macro_precision"] >= 0.85
    assert rep["intent"]["ece_after_calibration"] <= 0.05
    assert rep["must_escalate_detection_oof"]["missed"] == 0
