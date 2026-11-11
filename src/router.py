"""Routing: decide auto_respond or escalate, deterministically, with a plain-language reason (FR-04, FR-05, A5).

Order of rules (each decision records the rule that fired):
  R1  classification unavailable                      -> escalate
  R2  predicted intent is an always-escalate class    -> escalate   (FR-05; enforced here, outside any model)
  R3  probability on always-escalate classes too high -> escalate
  R4  retrieval failed or found nothing relevant      -> escalate
  R5  estimated answerability below the threshold     -> escalate
  otherwise                                           -> auto_respond

Estimated answerability p_auto = sum over intents of P(intent) * answerable_rate(intent), where the rate is the
share of development tickets of that intent that were answerable and not always-escalate. It is transparent
(a support manager can read it) and calibrated by construction. It is weakly discriminating (see
docs/decisions.md D-08): protection against wrong answers also comes from generation and the validator.

The threshold comes from a cost argument and a development-set sweep, `python -m src.router`; it is not a
default. All fitted values live in models/router.json, which `python -m src.router` rewrites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .classify import MUST_ESCALATE, Classification
from .retrieval import RetrievalResult

ROOT = Path(__file__).resolve().parent.parent
ROUTER_PATH = ROOT / "models" / "router.json"

# Used only if models/router.json is missing; replaced by the fitted values.
DEFAULT_THRESHOLD = 0.64
DEFAULT_P_MUST_LIMIT = 0.20


@dataclass(frozen=True)
class RouteDecision:
    route: str  # "auto_respond" | "escalate"
    reason: str  # plain language, readable by a support manager
    rule: str  # "R1".."R5" or "AUTO"
    p_auto: float
    threshold: float


class Router:
    def __init__(
        self, rates: dict[str, float] | None = None, threshold: float | None = None, p_must_limit: float | None = None
    ):
        cfg = {}
        if rates is None or threshold is None or p_must_limit is None:
            try:
                cfg = json.loads(ROUTER_PATH.read_text())
            except Exception:
                cfg = {}
        self.rates = rates if rates is not None else cfg.get("answerable_rates", {})
        self.threshold = float(threshold if threshold is not None else cfg.get("threshold", DEFAULT_THRESHOLD))
        self.p_must_limit = float(
            p_must_limit if p_must_limit is not None else cfg.get("p_must_limit", DEFAULT_P_MUST_LIMIT)
        )

    def p_auto(self, c: Classification) -> float:
        if not c.intent_probs or not self.rates:
            return 0.0
        return round(float(sum(p * self.rates.get(i, 0.0) for i, p in c.intent_probs.items())), 4)

    def decide(self, c: Classification, r: RetrievalResult) -> RouteDecision:
        pa = self.p_auto(c)
        th = self.threshold
        if c.fallback:
            return RouteDecision(
                "escalate", f"The ticket could not be classified ({c.error}), so a person should read it.", "R1", pa, th
            )
        label = c.intent.replace("_", " ")
        if c.intent in MUST_ESCALATE:
            return RouteDecision(
                "escalate",
                f"This looks like a {label}. Those are always handled by a person, " "never answered automatically.",
                "R2",
                pa,
                th,
            )
        if c.p_must_escalate >= self.p_must_limit:
            return RouteDecision(
                "escalate",
                f"There is a {c.p_must_escalate:.0%} chance this is a security, compliance, "
                f"feature or unclear request (best guess: {label}), which a person must handle.",
                "R3",
                pa,
                th,
            )
        if r.error:
            return RouteDecision(
                "escalate", f"Documentation search failed ({r.error}), so no sourced answer is possible.", "R4", pa, th
            )
        if not r.passages:
            return RouteDecision(
                "escalate", "No documentation passage was relevant enough to answer from.", "R4", pa, th
            )
        if pa < th:
            return RouteDecision(
                "escalate",
                f"Classified as {label} (confidence {c.intent_confidence:.2f}), but tickets "
                f"like this are answerable from the documentation only {pa:.0%} of the time, below the "
                f"{th:.0%} needed to answer without a person.",
                "R5",
                pa,
                th,
            )
        return RouteDecision(
            "auto_respond",
            f"Classified as {label} (confidence {c.intent_confidence:.2f}). Tickets like "
            f"this are answerable from the documentation {pa:.0%} of the time (needs {th:.0%}). Best "
            f"matching article: {r.doc_ids[0]} (score {r.top_score:.2f}).",
            "AUTO",
            pa,
            th,
        )


# ------------------------------------------------------------------ fitting
# Assumptions behind the threshold (docs/decisions.md D-08). The pass-through share is a PLACEHOLDER until generation
# and the validator exist (B-08, B-09); it is measured at B-11 and the threshold is re-fitted.
WRONG_ANSWER_COST = 9.0  # a wrong reply sent to a customer, in units of the saving from one correct automatic reply
FILTER_PASS_THROUGH = 0.20  # share of wrongly auto-routed tickets assumed to survive generation and validation


def fit(
    dev_path: str | Path,
    wrong_cost: float = WRONG_ANSWER_COST,
    pass_through: float = FILTER_PASS_THROUGH,
    save: bool = True,
) -> dict:
    """Fit answerable rates and choose the threshold on DEVELOPMENT data only, with leak-free folds."""
    from sklearn.model_selection import GroupKFold

    from .classify import _oof_logits, _softmax, text_group
    from .embeddings import embed
    from .ingest import load_tickets

    tickets = [t for t in load_tickets(dev_path) if t.labels]
    intents = sorted({t.labels["intent"] for t in tickets})
    yi = np.array([intents.index(t.labels["intent"]) for t in tickets])
    auto = np.array([t.labels["expected_route"] == "auto_respond" for t in tickets])
    groups = np.array([text_group(t.body) for t in tickets])
    X = embed([t.text for t in tickets])
    art = json.loads((ROOT / "models" / "classifier_report.json").read_text())
    P = _softmax(_oof_logits(X, yi, groups, art["intent"]["C"], len(intents)) / art["intent"]["temperature"])

    def rates_from(idx):
        # Laplace smoothing keeps a rate away from exactly 0 or 1 for small classes
        return np.array([(auto[idx][yi[idx] == k].sum() + 1) / ((yi[idx] == k).sum() + 2) for k in range(len(intents))])

    p_auto = np.zeros(len(tickets))
    for tr, te in GroupKFold(5).split(X, yi, groups):
        r = rates_from(tr)
        for k, name in enumerate(intents):
            if name in MUST_ESCALATE:
                r[k] = 0.0
        p_auto[te] = P[te] @ r
    full = rates_from(np.arange(len(tickets)))
    rates = {n: (0.0 if n in MUST_ESCALATE else round(float(full[k]), 4)) for k, n in enumerate(intents)}

    must = np.array([t.labels.get("must_not_auto_respond", False) for t in tickets])
    pm = P[:, [k for k, n in enumerate(intents) if n in MUST_ESCALATE]].sum(1)
    sweep = []
    for th in np.round(np.arange(0.50, 0.96, 0.05), 2):
        sent = (p_auto >= th) & (pm < DEFAULT_P_MUST_LIMIT)
        n = int(sent.sum())
        right = int((sent & auto).sum())
        sweep.append(
            {
                "threshold": float(th),
                "auto_responded": n,
                "coverage_pct": round(100 * n / len(tickets), 1),
                "labelled_auto_among_them_pct": round(100 * right / n, 1) if n else None,
                "wrongly_auto_responded": n - right,
                "must_not_auto_responded": int((sent & must).sum()),
                "implied_router_cost_ratio": round(float(th) / (1 - float(th)), 2),
            }
        )
    # cost argument: answer if p_auto * saved > (1 - p_auto) * wrong_router  =>  threshold = W / (W + 1),
    # where W is the cost that actually reaches the customer: the true cost times the share the later stages miss.
    w_router = wrong_cost * pass_through
    cost_threshold = w_router / (w_router + 1.0)
    limits = []
    for lim in (0.05, 0.10, 0.20, 0.30, 0.50):
        sent = (p_auto >= cost_threshold) & (pm < lim)
        limits.append(
            {
                "p_must_limit": lim,
                "auto_responded": int(sent.sum()),
                "must_not_auto_responded": int((sent & must).sum()),
            }
        )
    cfg = {
        "answerable_rates": rates,
        "threshold": round(cost_threshold, 2),
        "p_must_limit": DEFAULT_P_MUST_LIMIT,
        "rule": "threshold = W / (W + 1), W = wrong_cost x pass_through = %s x %s = %s (units of the saving from one "
        "correct automatic reply)" % (wrong_cost, pass_through, round(w_router, 2)),
        "wrong_cost": wrong_cost,
        "filter_pass_through_ASSUMED": pass_through,
        "n_train": len(tickets),
        "folds": "GroupKFold(5) by ticket body text",
        "sweep": sweep,
        "p_must_limit_sweep": limits,
    }
    if save:
        ROUTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        ROUTER_PATH.write_text(json.dumps(cfg, indent=2))
    return cfg


if __name__ == "__main__":
    out = fit(ROOT / "data" / "development_tickets.json")
    print("rates:", json.dumps(out["answerable_rates"]))
    print("chosen threshold:", out["threshold"], "|", out["rule"])
    print("threshold coverage% auto-correct% wrong must_not")
    for s in out["sweep"]:
        print(
            f"{s['threshold']:>9}  {s['coverage_pct']:>8}  {s['labelled_auto_among_them_pct']}  "
            f"{s['wrongly_auto_responded']:>5}  {s['must_not_auto_responded']:>3}"
        )
    print("p_must_limit sweep:", out["p_must_limit_sweep"])
