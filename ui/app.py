"""Web UI backend (FastAPI). Three jobs: try one ticket, browse a harness run, switch the kill switch.

Nothing here makes a decision: tickets go through Pipeline.process, runs are read from the harness output files,
and the switch uses src.killswitch. The server is meant for this machine only (see __main__: 127.0.0.1).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import killswitch
from src.config import ROOT

STATIC = Path(__file__).resolve().parent / "static"
MAX_TEXT = 4000

# Tickets worth showing, with what the 2026-11-11 gate run did with them (see docs/gate_run_validation.md).
# The outcome of a live run can differ, e.g. if the model's answer changes, so the labels say "in the gate run".
SAMPLES = [
    ("VAL-0004", "Answered in the gate run"),
    ("VAL-0005", "Answered in the gate run"),
    ("VAL-0009", "Escalated in the gate run: compliance request"),
    ("VAL-0002", "Escalated in the gate run: unclear request"),
    ("VAL-0028", "Reply blocked in the gate run (the checking model is erratic, so a live run may pass it)"),
    ("VAL-0047", "Reply blocked in the gate run (the checking model is erratic, so a live run may pass it)"),
]


class TicketIn(BaseModel):
    subject: str = Field("", max_length=MAX_TEXT)
    body: str = Field("", max_length=MAX_TEXT)
    channel: str = "email"


class SwitchIn(BaseModel):
    on: bool
    reason: str = Field("", max_length=200)


def _load_tickets(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("tickets", [])


def create_app(pipeline=None, results_root: Path | None = None, tickets_file: Path | None = None) -> FastAPI:
    app = FastAPI(title="CloudServe support desk", docs_url=None, redoc_url=None, openapi_url=None)
    root = Path(results_root) if results_root else ROOT / "evaluation" / "results"
    tickets_path = Path(tickets_file) if tickets_file else ROOT / "data" / "validation_tickets.json"
    state: dict = {"pipeline": pipeline, "n": 0}
    lock = threading.Lock()  # one ticket at a time: the demo is single-user and the pipeline is not built for races

    def get_pipeline():
        if state["pipeline"] is None:
            from src.decision_log import DecisionLog
            from src.pipeline import Pipeline

            log = DecisionLog(f"sqlite:///{(ROOT / 'storage' / 'ui_demo.db').resolve()}")
            state["pipeline"] = Pipeline(decision_log=log)
        return state["pipeline"]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/samples")
    def samples():
        try:
            by_id = {t.get("ticket_id"): t for t in _load_tickets(tickets_path)}
        except Exception:
            return []
        return [
            {
                "id": i,
                "note": note,
                "subject": by_id[i].get("subject", ""),
                "body": by_id[i].get("body", ""),
                "channel": by_id[i].get("channel", "email"),
            }
            for i, note in SAMPLES
            if i in by_id
        ]

    @app.post("/api/process")
    def process(t: TicketIn):
        if not (t.subject.strip() or t.body.strip()):
            raise HTTPException(422, "Enter a subject or a message first.")
        with lock:
            state["n"] += 1
            raw = {"ticket_id": f"UI-{state['n']:04d}", "channel": t.channel, "subject": t.subject, "body": t.body}
            p = get_pipeline()
            res = p.process(raw).to_dict()
            dl = getattr(p, "decision_log", None)  # ids restart each server run, so keep only this run's rows
            rows = [r for r in dl.fetch(res["ticket_id"]) if r.get("run_id") == dl.run_id] if dl else []
        res["decisions_logged"] = len(rows)
        res["decision_stages"] = [r.get("stage") for r in rows]
        res["kill_switch"] = killswitch.status().active
        return res

    # ---- harness runs
    def run_dirs() -> dict[str, Path]:
        if not root.is_dir():
            return {}
        return {d.name: d for d in sorted(root.iterdir()) if (d / "results.jsonl").is_file()}

    @app.get("/api/runs")
    def runs():
        out = []
        for name, d in run_dirs().items():
            m = d / "metrics.json"
            info = {}
            if m.is_file():
                try:
                    info = json.loads(m.read_text(encoding="utf-8")).get("run", {})
                except Exception:
                    pass
            out.append({"name": name, "started_at": info.get("started_at", ""), "input": info.get("input", "")})
        return out

    @app.get("/api/runs/{name}")
    def run(name: str):
        d = run_dirs().get(name)  # only names that exist in the listing; no path is built from user input
        if d is None:
            raise HTTPException(404, "No such run.")
        metrics = {}
        try:
            metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        rows = []
        for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            g = r.get("guardrail") or {}
            rows.append(
                {
                    "ticket_id": r.get("ticket_id"),
                    "channel": r.get("channel"),
                    "intent": r.get("intent"),
                    "urgency": r.get("urgency"),
                    "route": r.get("route"),
                    "rule": r.get("route_rule"),
                    "outcome": r.get("outcome"),
                    "blocked": bool(g.get("blocked")),
                    "failed_checks": g.get("failed_checks", []),
                    "latency_ms": r.get("latency_ms"),
                    "reason": r.get("route_reason"),
                    "reply": (r.get("response") or {}).get("reply", ""),
                    "handover": r.get("handover"),
                    "citations": (r.get("response") or {}).get("citations", []),
                }
            )
        return {"name": name, "metrics": metrics, "tickets": rows}

    # ---- kill switch
    @app.get("/api/killswitch")
    def ks_get():
        s = killswitch.status()
        return {"active": s.active, "reason": s.reason}

    @app.post("/api/killswitch")
    def ks_set(body: SwitchIn):
        if body.on:
            killswitch.turn_on(body.reason.strip() or "switched on from the web page")
        else:
            killswitch.turn_off()
        s = killswitch.status()
        return {"active": s.active, "reason": s.reason}

    return app
