"""Intent and urgency classification with calibrated confidence (FR-02, A3).

Design (docs/decisions.md D-03, D-06)
- Features: all-MiniLM-L6-v2 sentence embeddings of the ticket text.
- Intent: multinomial logistic regression, probabilities calibrated by temperature scaling fitted on
  out-of-fold predictions. Folds are grouped by ticket text, because the development set repeats texts
  and an ordinary split would leak.
- Urgency: chosen among candidate feature sets by grouped cross-validation. Urgency is only weakly
  predictable from the text (see docs/data_findings.md), so its confidence is expected to be low.
- Deterministic: no randomness at prediction time; the same ticket always gives the same output.
- Never raises: any failure returns the defined fallback (intent "unclear_request", confidence 0.0).
- Labels on tickets are used only for training here, never at prediction time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .embeddings import embed
from .ingest import Ticket, load_tickets

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "classifier.joblib"
REPORT_PATH = ROOT / "models" / "classifier_report.json"

# Classes that must never be answered automatically (labels.must_not_auto_respond is True for exactly these).
MUST_ESCALATE = frozenset({"security_incident", "compliance_request", "feature_request", "unclear_request"})
FALLBACK_INTENT = "unclear_request"
FALLBACK_URGENCY = "medium"


@dataclass
class Classification:
    intent: str
    intent_confidence: float
    urgency: str
    urgency_confidence: float
    intent_alternatives: list[tuple[str, float]] = field(default_factory=list)
    urgency_alternatives: list[tuple[str, float]] = field(default_factory=list)
    p_must_escalate: float = 1.0  # probability mass on the always-escalate classes
    intent_probs: dict[str, float] = field(default_factory=dict)  # full calibrated distribution
    fallback: bool = False
    error: str | None = None
    model_version: str = ""

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "urgency": self.urgency,
            "urgency_confidence": self.urgency_confidence,
            "intent_alternatives": self.intent_alternatives,
            "urgency_alternatives": self.urgency_alternatives,
            "p_must_escalate": self.p_must_escalate,
            "intent_probs": self.intent_probs,
            "fallback": self.fallback,
            "error": self.error,
            "model_version": self.model_version,
        }


# ------------------------------------------------------------------ helpers
def text_group(text: str) -> str:
    return hashlib.md5(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _calibrated(model: LogisticRegression, X: np.ndarray, temperature: float) -> np.ndarray:
    return _softmax(model.decision_function(X) / temperature)


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar

    def nll(t: float) -> float:
        p = _softmax(logits / t)
        return float(-np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-12, None)).mean())

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins: int = 10) -> float:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _oof_logits(X, y_idx, groups, C, n_classes, folds=5):
    out = np.zeros((len(y_idx), n_classes))
    for tr, te in GroupKFold(folds).split(X, y_idx, groups):
        m = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y_idx[tr])
        d = m.decision_function(X[te])
        if d.ndim == 1:  # binary case
            d = np.stack([-d, d], axis=1)
        full = np.full((len(te), n_classes), -30.0)  # classes unseen in a fold get a very low score
        full[:, m.classes_] = d
        out[te] = full
    return out


def _similarity_groups(X: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the graph linking texts with cosine similarity >= threshold."""
    n = len(X)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sim = X @ X.T
    for i in range(n):
        for j in np.nonzero(sim[i, i + 1 :] >= threshold)[0] + i + 1:
            parent[find(i)] = find(int(j))
    return np.array([find(i) for i in range(n)])


def _per_class(y_true, y_pred, classes):
    rows = {}
    for i, c in enumerate(classes):
        tp = int(((y_true == i) & (y_pred == i)).sum())
        fp = int(((y_true != i) & (y_pred == i)).sum())
        fn = int(((y_true == i) & (y_pred != i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows[c] = {"precision": round(p, 3), "recall": round(r, 3), "support": int((y_true == i).sum())}
    return rows


# ----------------------------------------------------------------- training
def train(dev_path: str | Path, save: bool = True) -> tuple[dict, dict]:
    tickets = [t for t in load_tickets(dev_path) if t.labels and t.labels.get("intent") and t.labels.get("urgency")]
    X = embed([t.text for t in tickets])
    intents = sorted({t.labels["intent"] for t in tickets})
    urgencies = ["low", "medium", "high"]
    yi = np.array([intents.index(t.labels["intent"]) for t in tickets])
    yu = np.array([urgencies.index(t.labels["urgency"]) for t in tickets])
    groups = np.array(
        [text_group(t.body) for t in tickets]
    )  # body only: same body under another subject is the same text
    report: dict = {
        "n_train": len(tickets),
        "n_distinct_texts": int(len(set(groups))),
        "folds": "GroupKFold(5) by ticket text",
    }

    # ---- intent: pick C by grouped CV log-loss, then fit temperature on the out-of-fold logits
    best = None
    for C in (1, 3, 10, 30, 100):
        lg = _oof_logits(X, yi, groups, C, len(intents))
        T = fit_temperature(lg, yi)
        p = _softmax(lg / T)
        nll = float(-np.log(np.clip(p[np.arange(len(yi)), yi], 1e-12, None)).mean())
        acc = float((p.argmax(1) == yi).mean())
        report.setdefault("intent_C_search", {})[str(C)] = {"oof_accuracy": round(acc, 4), "oof_nll": round(nll, 4)}
        if best is None or nll < best[0]:
            best = (nll, C, T, lg)
    _, C_int, T_int, lg_int = best
    p_raw, p_cal = _softmax(lg_int), _softmax(lg_int / T_int)
    pred = p_cal.argmax(1)
    f1s = []
    pc = _per_class(yi, pred, intents)
    for v in pc.values():
        f1s.append(
            2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"]) if v["precision"] + v["recall"] else 0
        )
    report["intent"] = {
        "C": C_int,
        "temperature": round(T_int, 4),
        "oof_accuracy": round(float((pred == yi).mean()), 4),
        "oof_macro_precision": round(float(np.mean([v["precision"] for v in pc.values()])), 4),
        "oof_macro_recall": round(float(np.mean([v["recall"] for v in pc.values()])), 4),
        "oof_macro_f1": round(float(np.mean(f1s)), 4),
        "ece_before_calibration": round(expected_calibration_error(p_raw, yi), 4),
        "ece_after_calibration": round(expected_calibration_error(p_cal, yi), 4),
        "per_class": pc,
    }
    # stricter lower bound: also keep near-duplicate paraphrases (cosine >= 0.85) together
    strict = _similarity_groups(X, 0.85)
    lg_s = _oof_logits(X, yi, strict, C_int, len(intents))
    report["intent_strict_grouping"] = {
        "cosine_threshold": 0.85,
        "n_groups": int(len(set(strict))),
        "oof_accuracy": round(float((_softmax(lg_s / T_int).argmax(1) == yi).mean()), 4),
        "note": "lower bound for brand-new paraphrases; the hidden set likely repeats development texts",
    }
    # accuracy of must-escalate decision from the predicted class
    me_true = np.array([intents[i] in MUST_ESCALATE for i in yi])
    me_pred = np.array([intents[i] in MUST_ESCALATE for i in pred])
    report["must_escalate_detection_oof"] = {
        "recall": round(float((me_pred & me_true).sum() / me_true.sum()), 4),
        "missed": int((me_true & ~me_pred).sum()),
        "of": int(me_true.sum()),
    }

    # ---- urgency: candidate feature sets, judged by grouped CV on log-loss (second level uses OOF intent probs)
    P = p_cal
    cands = {"intent_probs": P, "embedding": X, "embedding+intent_probs": np.hstack([X, P])}
    baseline_acc = float(np.bincount(yu).max() / len(yu))
    report["urgency_candidates"] = {"majority_baseline_accuracy": round(baseline_acc, 4)}
    best_u = None
    for name, F in cands.items():
        for C in (0.3, 1, 3, 10):
            lg = _oof_logits(F, yu, groups, C, 3)
            T = fit_temperature(lg, yu)
            p = _softmax(lg / T)
            nll = float(-np.log(np.clip(p[np.arange(len(yu)), yu], 1e-12, None)).mean())
            acc = float((p.argmax(1) == yu).mean())
            report["urgency_candidates"][f"{name} C={C}"] = {"oof_accuracy": round(acc, 4), "oof_nll": round(nll, 4)}
            if best_u is None or nll < best_u[0]:
                best_u = (nll, name, C, T, p, acc)
    _, u_name, u_C, T_u, p_u, acc_u = best_u
    report["urgency"] = {
        "features": u_name,
        "C": u_C,
        "temperature": round(T_u, 4),
        "oof_accuracy": round(acc_u, 4),
        "majority_baseline_accuracy": round(baseline_acc, 4),
        "ece_after_calibration": round(expected_calibration_error(p_u, yu), 4),
        "mean_confidence": round(float(p_u.max(1).mean()), 4),
        "per_class": _per_class(yu, p_u.argmax(1), urgencies),
    }

    # ---- final models on all development data
    m_int = LogisticRegression(C=C_int, max_iter=5000).fit(X, yi)
    F_u = cands[u_name]
    m_urg = LogisticRegression(C=u_C, max_iter=5000).fit(F_u, yu)
    version = "clf-" + hashlib.md5(json.dumps(report["intent"]["C"]).encode() + X.tobytes()).hexdigest()[:8]
    artefact = {
        "version": version,
        "intents": intents,
        "urgencies": urgencies,
        "intent_model": m_int,
        "intent_T": T_int,
        "urgency_model": m_urg,
        "urgency_T": T_u,
        "urgency_features": u_name,
    }
    report["version"] = version
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artefact, MODEL_PATH)
        REPORT_PATH.write_text(json.dumps(report, indent=2))
    return artefact, report


# --------------------------------------------------------------- prediction
class Classifier:
    def __init__(self, artefact: dict | None = None, dev_path: str | Path | None = None):
        self.error: str | None = None
        self.art = artefact
        if self.art is None:
            try:
                self.art = joblib.load(MODEL_PATH)
            except Exception:
                try:  # no saved model (clean checkout without models/): train deterministically
                    self.art, _ = train(dev_path or ROOT / "data" / "development_tickets.json")
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    log.error("classifier unavailable: %s", self.error)

    def _fallback(self, why: str) -> Classification:
        return Classification(
            FALLBACK_INTENT,
            0.0,
            FALLBACK_URGENCY,
            0.0,
            fallback=True,
            error=why,
            model_version=(self.art or {}).get("version", ""),
        )

    def predict(self, ticket: Ticket) -> Classification:
        return self.predict_many([ticket])[0]

    def predict_many(self, tickets: list[Ticket]) -> list[Classification]:
        if self.art is None:
            return [self._fallback(f"classifier_unavailable: {self.error}") for _ in tickets]
        out: list[Classification | None] = [None] * len(tickets)
        usable = [i for i, t in enumerate(tickets) if t.text.strip()]
        for i, t in enumerate(tickets):
            if i not in usable:
                out[i] = self._fallback("empty_text")
        if usable:
            try:
                X = embed([tickets[i].text for i in usable])
                a = self.art
                P = _calibrated(a["intent_model"], X, a["intent_T"])
                F = {"intent_probs": P, "embedding": X, "embedding+intent_probs": np.hstack([X, P])}[
                    a["urgency_features"]
                ]
                U = _calibrated(a["urgency_model"], F, a["urgency_T"])
                for row, i in enumerate(usable):
                    out[i] = self._build(P[row], U[row])
            except Exception as exc:
                for i in usable:
                    out[i] = self._fallback(f"{type(exc).__name__}: {exc}")
        return out  # type: ignore[return-value]

    def _build(self, p: np.ndarray, u: np.ndarray) -> Classification:
        a = self.art
        order = np.lexsort((np.arange(len(p)), -p))  # ties broken by class index: deterministic
        uo = np.lexsort((np.arange(len(u)), -u))
        intents = a["intent_model"].classes_
        names = [a["intents"][c] for c in intents]
        pm = float(sum(p[k] for k, n in enumerate(names) if n in MUST_ESCALATE))
        return Classification(
            intent=names[order[0]],
            intent_confidence=round(float(p[order[0]]), 4),
            urgency=a["urgencies"][a["urgency_model"].classes_[uo[0]]],
            urgency_confidence=round(float(u[uo[0]]), 4),
            intent_alternatives=[(names[k], round(float(p[k]), 4)) for k in order[:3]],
            urgency_alternatives=[(a["urgencies"][a["urgency_model"].classes_[k]], round(float(u[k]), 4)) for k in uo],
            p_must_escalate=round(pm, 4),
            intent_probs={n: round(float(p[k]), 4) for k, n in enumerate(names)},
            model_version=a["version"],
        )


if __name__ == "__main__":
    import sys

    _, rep = train(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "development_tickets.json")
    print(
        json.dumps({k: v for k, v in rep.items() if k not in ("intent_C_search", "urgency_candidates")}, indent=2)[
            :6000
        ]
    )
