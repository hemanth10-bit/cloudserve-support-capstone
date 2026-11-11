"""Fairness audit (NFR-06): the method is sound and group fields are not decision inputs."""

import re
from pathlib import Path

from evaluation.fairness import MIN_N, audit, wilson

SRC = Path(__file__).resolve().parent.parent / "src"


def test_group_attributes_are_not_inputs_to_any_decision():
    for name in ("classify.py", "router.py", "generation.py", "validator.py", "retrieval.py", "pipeline.py"):
        code = (SRC / name).read_text()
        for field in ("customer_tier", "customer_region", "language_fluency"):
            assert not re.search(rf"\b{field}\b", code), f"{name} reads {field}"


def test_customer_id_and_name_are_only_used_to_block_them_from_replies():
    """They are identifiers, not group attributes: the only use allowed is passing them to the privacy check."""
    for name in ("classify.py", "router.py", "generation.py", "retrieval.py", "validator.py"):
        code = (SRC / name).read_text()
        assert not re.search(r"\bcustomer_(id|name)\b", code), f"{name} reads a customer identifier"
    uses = [ln for ln in (SRC / "pipeline.py").read_text().splitlines() if re.search(r"\bcustomer_(id|name)\b", ln)]
    assert len(uses) == 1 and "ticket.customer_id, ticket.customer_name" in uses[0]


def test_wilson_interval_behaves():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(10, 10)[1] == 1.0 and wilson(10, 10)[0] > 0.6


def synth(rates, n=100):
    """rates: {group: answered share}. Tickets all labelled answerable; the region is the group."""
    results, tickets = [], {}
    for g, share in rates.items():
        for i in range(n):
            tid = f"{g}-{i}"
            answered = i < share * n
            results.append(
                {
                    "ticket_id": tid,
                    "outcome": "answered" if answered else "escalated",
                    "intent": "x",
                    "retrieved_doc_ids": [],
                    "response": None,
                    "guardrail": None,
                }
            )
            tickets[tid] = {
                "customer_region": g,
                "language_fluency": "fluent",
                "customer_tier": "standard",
                "channel": "chat",
                "labels": {
                    "answerable_from_docs": True,
                    "must_not_auto_respond": False,
                    "expected_route": "auto_respond",
                    "intent": "x",
                    "expected_doc_ids": [],
                },
            }
    return audit(results, tickets)


def verdict(rep):
    return rep["attributes"]["region"]["answered_rate"]


def test_a_large_real_gap_is_called_a_gap():
    v = verdict(synth({"a": 0.9, "b": 0.5}, n=200))
    assert v["gap_pp"] == 40.0 and v["verdict"].startswith("GAP ABOVE")


def test_equal_groups_are_within_the_limit_only_if_the_sample_can_show_it():
    v = verdict(synth({"a": 0.6, "b": 0.6}, n=2000))
    assert v["gap_pp"] == 0 and v["verdict"].startswith("within the limit, and")
    small = verdict(synth({"a": 0.6, "b": 0.6}, n=40))
    assert "too small" in small["verdict"]


def test_a_gap_a_small_sample_cannot_distinguish_from_chance_is_not_claimed():
    v = verdict(synth({"a": 0.65, "b": 0.50}, n=40))
    assert v["gap_pp"] > 5 and "cannot tell it from chance" in v["verdict"]


def test_groups_below_the_minimum_are_listed_but_never_judged():
    r = verdict(synth({"a": 0.6, "b": 0.6}, n=MIN_N))
    assert "tiny" not in r["groups"]
    rep = synth({"a": 0.6, "b": 0.6}, n=MIN_N - 1)
    assert "fewer than two" in verdict(rep)["verdict"]
