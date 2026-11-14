"""The web UI backend: it must call the real pipeline, show every outcome, and keep the kill switch working."""

import json

import pytest
from fastapi.testclient import TestClient

from src import killswitch
from src.decision_log import DecisionLog
from src.pipeline import Pipeline
from tests.fakes import FakeLLM
from ui.app import SAMPLES, create_app

EASY = {
    "subject": "Export is stuck",
    "body": "My data export has been running for over an hour and shows no progress. How do I check the job status?",
}


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    log = DecisionLog(f"sqlite:///{tmp_path_factory.mktemp('ui') / 'd.db'}")
    return Pipeline(llm=FakeLLM(), decision_log=log)


@pytest.fixture()
def client(pipe, tmp_path):
    return TestClient(create_app(pipeline=pipe, results_root=tmp_path / "runs"))


def test_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Support desk" in r.text


def test_try_a_ticket_returns_the_full_result_and_logs_it(client):
    r = client.post("/api/process", json=EASY)
    assert r.status_code == 200
    d = r.json()
    assert d["outcome"] in ("answered", "escalated") and d["stage_status"]["ingest"] == "ok"
    assert d["decisions_logged"] >= 3 and "classification" in d["decision_stages"]


def test_empty_ticket_is_refused_with_a_plain_message(client):
    r = client.post("/api/process", json={"subject": " ", "body": ""})
    assert r.status_code == 422 and "Enter a subject" in r.json()["detail"]


def test_oversized_text_is_refused(client):
    assert client.post("/api/process", json={"subject": "x", "body": "a" * 5000}).status_code == 422


def test_kill_switch_from_the_page_escalates_the_next_ticket(client):
    assert client.get("/api/killswitch").json()["active"] is False
    s = client.post("/api/killswitch", json={"on": True, "reason": "test"}).json()
    assert s["active"] is True and "test" in s["reason"]
    d = client.post("/api/process", json=EASY).json()
    assert d["outcome"] == "escalated" and d["route_rule"] == "KS" and d["kill_switch"] is True
    assert client.post("/api/killswitch", json={"on": False}).json()["active"] is False
    assert killswitch.status().active is False


def test_runs_are_listed_and_read_from_harness_files(pipe, tmp_path):
    d = tmp_path / "runs" / "demo"
    d.mkdir(parents=True)
    res = pipe.process(dict(EASY, ticket_id="T1")).to_dict()
    (d / "results.jsonl").write_text(json.dumps(res) + "\nnot json\n", encoding="utf-8")
    (d / "metrics.json").write_text(
        json.dumps({"run": {"input": "x.json"}, "metrics": {"volume": {"tickets_processed": 1}}})
    )
    c = TestClient(create_app(pipeline=pipe, results_root=tmp_path / "runs"))
    assert c.get("/api/runs").json()[0]["name"] == "demo"
    body = c.get("/api/runs/demo").json()
    assert len(body["tickets"]) == 1 and body["tickets"][0]["ticket_id"] == "T1"  # the bad line is skipped


@pytest.mark.parametrize("name", ["../../etc/passwd", "..", "nope"])
def test_run_names_cannot_escape_the_results_folder(client, name):
    assert client.get(f"/api/runs/{name}").status_code in (404, 405)


def test_samples_come_from_the_validation_file(client):
    s = client.get("/api/samples").json()
    assert 1 <= len(s) <= len(SAMPLES) and all(x["body"] for x in s)
