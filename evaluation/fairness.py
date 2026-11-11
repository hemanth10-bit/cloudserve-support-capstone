"""Fairness audit (NFR-06, FR-12): is service quality within 5 percentage points across customer groups?

    python -m evaluation.fairness --results DIR --tickets FILE [--output DIR]

Groups: language fluency, region, tier, channel. None of these fields is an input to classification, routing,
generation or validation (checked in tests/test_fairness.py), so a gap can only come from the ticket text.

"Quality" has no single number here, so several are reported, each a rate per group:
  answered_rate              share of tickets the system answered without a person
  answered_given_answerable  the same, among tickets the labels call answerable and not always-escalate
                             (controls for groups simply asking more unanswerable questions)
  route_agrees_with_label    router/outcome matches the expected route
  right_answer_rate          among answered tickets, share the labels call answerable
  blocked_rate               validator blocked the drafted reply
  intent_correct             intent matches the label (optimistic on development data: the classifier trained on it)
  retrieval_hit_top5         expected document in the top five (answerable tickets only)

For every metric and attribute: n and rate per group with a 95% Wilson interval, the largest gap between groups, a
chi-square test of "no difference", and a verdict that separates a real-looking gap from one the sample cannot
distinguish from chance. A group with fewer than MIN_N tickets is listed but never used to claim a gap or its absence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

LIMIT_PP = 5.0
MIN_N = 30
ATTRIBUTES = {
    "language_fluency": "fluency",
    "customer_region": "region",
    "customer_tier": "tier",
    "channel": "channel",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - w) / d), min(1.0, (c + w) / d))


def metrics_for(r: dict, lab: dict) -> dict[str, bool | None]:
    must = bool(lab.get("must_not_auto_respond"))
    answerable = bool(lab.get("answerable_from_docs"))
    answered = r["outcome"] == "answered"
    exp_route = lab.get("expected_route")
    exp_docs = set(lab.get("expected_doc_ids") or [])
    blocked = bool((r.get("guardrail") or {}).get("blocked"))
    return {
        "answered_rate": answered,
        "answered_given_answerable": answered if (answerable and not must) else None,
        "route_agrees_with_label": (
            ((r["outcome"] == "answered") == (exp_route == "auto_respond")) if exp_route else None
        ),
        "right_answer_rate": answerable and not must if answered else None,
        "blocked_rate": blocked if r.get("response") else None,
        "intent_correct": r["intent"] == lab.get("intent") if lab.get("intent") else None,
        "retrieval_hit_top5": (
            bool(exp_docs & set(r.get("retrieved_doc_ids", [])[:5])) if (answerable and exp_docs) else None
        ),
    }


def audit(results: list[dict], tickets: dict[str, dict]) -> dict:
    rows = []
    for r in results:
        t = tickets.get(r["ticket_id"])
        if not t or not t.get("labels"):
            continue
        rows.append(({k: t.get(k, "unknown") for k in ATTRIBUTES}, metrics_for(r, t["labels"])))
    names = list(rows[0][1]) if rows else []
    out = {"tickets_audited": len(rows), "limit_pp": LIMIT_PP, "min_group_n": MIN_N, "attributes": {}}
    for attr, label in ATTRIBUTES.items():
        groups = sorted({g[attr] for g, _ in rows})
        out["attributes"][label] = {}
        for m in names:
            cells = {}
            for g in groups:
                vals = [mm[m] for gg, mm in rows if gg[attr] == g and mm[m] is not None]
                k, n = sum(vals), len(vals)
                lo, hi = wilson(k, n)
                cells[g] = {
                    "n": n,
                    "rate_pct": round(100 * k / n, 1) if n else None,
                    "ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
                }
            usable = {g: c for g, c in cells.items() if c["n"] >= MIN_N}
            res: dict = {"groups": cells, "groups_too_small_to_judge": sorted(set(cells) - set(usable))}
            if len(usable) >= 2:
                rates = {g: c["rate_pct"] for g, c in usable.items()}
                hi_g, lo_g = max(rates, key=rates.get), min(rates, key=rates.get)
                gap = round(rates[hi_g] - rates[lo_g], 1)
                table = np.array(
                    [
                        [round(c["rate_pct"] * c["n"] / 100), c["n"] - round(c["rate_pct"] * c["n"] / 100)]
                        for c in usable.values()
                    ]
                )
                p = float(chi2_contingency(table)[1]) if table.min() >= 0 and (table.sum(axis=0) > 0).all() else None
                # interval for the gap between the two extreme groups (normal approximation on two proportions)
                a, b = usable[hi_g], usable[lo_g]
                se = (
                    math.sqrt(
                        a["rate_pct"] / 100 * (1 - a["rate_pct"] / 100) / a["n"]
                        + b["rate_pct"] / 100 * (1 - b["rate_pct"] / 100) / b["n"]
                    )
                    * 100
                )
                lo_gap, hi_gap = round(gap - 1.96 * se, 1), round(gap + 1.96 * se, 1)
                if gap <= LIMIT_PP and hi_gap <= LIMIT_PP:
                    verdict = "within the limit, and the interval is inside it"
                elif gap <= LIMIT_PP:
                    verdict = (
                        "gap within the limit, but the interval reaches past it: "
                        "sample too small to rule out a larger gap"
                    )
                elif lo_gap > LIMIT_PP and (p is not None and p < 0.05):
                    verdict = "GAP ABOVE THE LIMIT and not explained by chance"
                else:
                    verdict = "gap above the limit, but the sample cannot tell it from chance"
                res.update(
                    {
                        "highest": hi_g,
                        "lowest": lo_g,
                        "gap_pp": gap,
                        "gap_ci95_pp": [lo_gap, hi_gap],
                        "chi2_p": round(p, 4) if p is not None else None,
                        "verdict": verdict,
                    }
                )
            else:
                res["verdict"] = "fewer than two groups large enough to compare"
            out["attributes"][label][m] = res
    return out


def to_markdown(rep: dict, source: str) -> str:
    lines = [
        "# Fairness audit",
        "",
        f"- Source: `{source}`; tickets audited: {rep['tickets_audited']}; limit {rep['limit_pp']} percentage points; "
        f"groups under {rep['min_group_n']} tickets are shown but never used to claim a gap or its absence.",
        "- Group fields are not inputs to any decision (tests/test_fairness.py).",
        "- Intent accuracy on development data is optimistic (the classifier trained on it).",
        "- The largest-minus-smallest gap is chosen after looking, so it overstates; the chi-square p-value is the "
        "fair test. Many comparisons are made, so a few p-values under 0.05 would be expected by chance.",
        "",
    ]
    for attr, metrics in rep["attributes"].items():
        lines += [f"## By {attr}", ""]
        for m, r in metrics.items():
            cells = ", ".join(f"{g} {c['rate_pct']}% (n={c['n']})" for g, c in r["groups"].items())
            gap = (
                f"gap {r['gap_pp']} pp (95% interval {r['gap_ci95_pp'][0]} to {r['gap_ci95_pp'][1]}, p={r['chi2_p']})"
                if "gap_pp" in r
                else ""
            )
            lines.append(f"- **{m}**: {cells}. {gap} **{r['verdict']}.**")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tickets", required=True)
    ap.add_argument("--output")
    a = ap.parse_args(argv)
    res = Path(a.results)
    results = [json.loads(line) for line in (res / "results.jsonl").read_text().splitlines()]
    tickets = {t["ticket_id"]: t for t in json.loads(Path(a.tickets).read_text())}
    rep = audit(results, tickets)
    out = Path(a.output or a.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fairness.json").write_text(json.dumps(rep, indent=1))
    (out / "fairness.md").write_text(to_markdown(rep, str(a.results)))
    print(to_markdown(rep, str(a.results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
