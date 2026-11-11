"""Kill switch (Governance Framework section 5): stops automatic answering at once, without a deployment."""

import json
import time
from pathlib import Path

from src import killswitch
from src.ingest import normalise
from src.pipeline import Pipeline
from tests.fakes import FakeLLM

DATA = Path(__file__).resolve().parent.parent / "data"
EASY = {"ticket_id": "E-1", "channel": "email", "subject": "Export", "body": "How do I export all our account data?"}


def test_off_by_default_and_the_easy_ticket_is_answered():
    assert not killswitch.status().active
    assert Pipeline(llm=FakeLLM()).process(normalise(EASY)).outcome == "answered"


def test_flag_file_escalates_everything_and_calls_no_model():
    killswitch.turn_on("provider returning bad answers")
    llm = FakeLLM()
    p = Pipeline(llm=llm)
    raw = json.loads((DATA / "validation_tickets.json").read_text())[:15]
    out = [p.process(normalise(r)) for r in raw]
    assert all(r.outcome == "escalated" and r.route_rule == "KS" for r in out)
    assert "switched off" in out[0].route_reason and "bad answers" in out[0].route_reason
    assert llm.calls == 0  # no drafting, no validation, no hand-over model call
    assert all(r.handover and r.handover["source"] == "template" for r in out)


def test_environment_variable_also_works(monkeypatch):
    monkeypatch.setenv("AUTO_RESPOND_DISABLED", "1")
    assert killswitch.status().active
    assert Pipeline(llm=FakeLLM()).process(normalise(EASY)).route_rule == "KS"


def test_it_takes_effect_on_the_very_next_ticket_and_can_be_lifted():
    p = Pipeline(llm=FakeLLM())
    assert p.process(normalise(EASY)).outcome == "answered"
    killswitch.turn_on("test")
    assert p.process(normalise({**EASY, "ticket_id": "E-2"})).outcome == "escalated"
    assert killswitch.turn_off() is True
    assert p.process(normalise({**EASY, "ticket_id": "E-3"})).outcome == "answered"


def test_a_ticket_already_in_flight_is_held_back_before_sending():
    class Throws(FakeLLM):
        """Throws the switch while the reply is being checked, i.e. after routing and drafting."""

        def complete(self, system, user, max_tokens=700):
            if "check whether a support reply" in system:
                killswitch.turn_on("thrown mid-flight")
            return super().complete(system, user, max_tokens)

    res = Pipeline(llm=Throws()).process(normalise(EASY))
    assert res.outcome == "escalated" and res.response["withheld"] is True
    assert "Held back before sending" in res.route_reason and res.handover


def test_the_switch_is_recorded_in_the_decision_log(tmp_path):
    from src.decision_log import DecisionLog

    killswitch.turn_on("test")
    log = DecisionLog(f"sqlite:///{tmp_path / 'd.db'}", reset=True)
    Pipeline(llm=FakeLLM(), decision_log=log).process(normalise(EASY))
    routing = next(r for r in log.fetch("E-1") if r["stage"] == "routing")
    assert routing["rule"] == "KS" and routing["action_taken"] == "escalate" and "switched off" in routing["reason"]


def test_if_the_check_itself_fails_the_system_fails_closed(monkeypatch):
    monkeypatch.setattr(killswitch, "flag_path", lambda: (_ for _ in ()).throw(OSError("disk gone")))
    s = killswitch.status()
    assert s.active and "failing closed" in s.reason


def test_checking_is_fast_enough_to_run_per_ticket():
    t = time.perf_counter()
    for _ in range(1000):
        killswitch.status()
    assert (time.perf_counter() - t) / 1000 < 0.001  # well under a millisecond each


def test_command_line_on_off_status(capsys):
    assert killswitch.main(["on", "incident", "42"]) == 0
    assert killswitch.main(["status"]) == 0
    assert "ON" in capsys.readouterr().out and killswitch.status().active
    assert killswitch.main(["off"]) == 0 and not killswitch.status().active
    assert killswitch.main(["nonsense"]) == 2
