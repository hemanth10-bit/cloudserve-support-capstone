"""TC-FR12: the unattended harness (FR-12, A9, A10, A11)."""

import json
import shutil
from pathlib import Path

import pytest

from evaluation import harness

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def sample(tmp_path_factory):
    d = tmp_path_factory.mktemp("in")
    raw = json.loads((DATA / "validation_tickets.json").read_text())[:12]
    messy = [
        {"ticket_id": "BAD-1", "channel": "email", "subject": "x", "body": ""},  # empty body
        {"ticket_id": "BAD-2", "channel": "telegram", "body": "?"},  # unknown channel
        {"body": None, "channel": None},  # nothing usable
        "this is not even an object",  # malformed record
        {"ticket_id": "BAD-3", "channel": "chat", "body": "x" * 20000},  # huge
    ]
    p = d / "mixed.json"
    p.write_text(json.dumps(raw + messy))
    return p, len(raw) + len(messy)


def test_every_ticket_gets_an_outcome_and_counts_reconcile(sample, tmp_path):
    path, n = sample
    code = harness.main(["--input", str(path), "--output", str(tmp_path)])
    assert code == 0
    results = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(results) == n
    assert all(r["outcome"] in ("answered", "escalated", "blocked") for r in results)
    assert all(r["route_reason"] for r in results)
    metrics = json.loads((tmp_path / "metrics.json").read_text())["metrics"]
    assert metrics["volume"]["tickets_processed"] == metrics["volume"]["tickets_in_input"] == n
    assert metrics["volume"]["reconciles"] is True


def test_report_files_are_produced_without_manual_work(sample, tmp_path):
    path, _ = sample
    harness.main(["--input", str(path), "--output", str(tmp_path / "nested" / "dir")])
    out = tmp_path / "nested" / "dir"
    assert (out / "metrics.json").exists() and (out / "metrics.md").exists() and (out / "results.jsonl").exists()
    md = (out / "metrics.md").read_text()
    for section in ("## Volume", "## Business", "## Technical", "## Governance", "PROVISIONAL"):
        assert section in md


def test_runs_on_a_renamed_unseen_file(sample, tmp_path):
    path, n = sample
    renamed = tmp_path / "some_hidden_file_i_have_never_seen.json"
    shutil.copy(path, renamed)
    assert harness.main(["--input", str(renamed), "--output", str(tmp_path / "o")]) == 0


def test_unlabelled_tickets_are_processed(tmp_path):
    raw = [{"ticket_id": "U-1", "channel": "chat", "body": "How do I roll back a failed release?"}]
    p = tmp_path / "unlabelled.json"
    p.write_text(json.dumps(raw))
    assert harness.main(["--input", str(p), "--output", str(tmp_path / "o")]) == 0
    m = json.loads((tmp_path / "o" / "metrics.json").read_text())["metrics"]
    assert m["technical"]["labelled_tickets"] == 0 and m["volume"]["tickets_processed"] == 1


def test_unreadable_input_gives_clear_error_and_code_2(tmp_path, capsys):
    assert harness.main(["--input", str(tmp_path / "nope.json"), "--output", str(tmp_path / "o")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_same_input_gives_same_decisions(sample, tmp_path):
    path, _ = sample
    harness.main(["--input", str(path), "--output", str(tmp_path / "a")])
    harness.main(["--input", str(path), "--output", str(tmp_path / "b")])

    def decisions(d):
        return [
            (r["ticket_id"], r["intent"], r["route"], r["route_reason"])
            for r in map(json.loads, (d / "results.jsonl").read_text().splitlines())
        ]

    assert decisions(tmp_path / "a") == decisions(tmp_path / "b")
