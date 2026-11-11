"""The ticket pipeline: ingest -> classify -> retrieve -> route -> (generate -> validate).

process() never raises and always returns a TicketResult, so a run cannot drop a ticket (A9, A11).

STATUS (sprint plan): ingest, classify, retrieve, route (B-02..B-07), generation (B-08), validation (B-09) built.
The validator (B-09) checks every drafted reply; any failed check blocks it and the ticket is escalated.

`route` is the router's decision. `outcome` is the final result: a ticket the router sent to auto_respond is
still escalated if generation says it cannot answer, fails, or cites sources it was not given.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .classify import Classifier
from .config import Settings, get_settings
from .ingest import Ticket, normalise
from .retrieval import RetrievalResult, Retriever
from .decision_log import DecisionLog
from .generation import draft_answer, handover
from .llm import LLMClient
from .router import RouteDecision, Router
from . import killswitch
from .validator import validate


@dataclass
class TicketResult:
    ticket_id: str
    channel: str
    intent: str
    intent_confidence: float
    urgency: str
    urgency_confidence: float
    p_must_escalate: float
    intent_alternatives: list = field(default_factory=list)
    retrieved_doc_ids: list = field(default_factory=list)
    retrieved_passages: list = field(default_factory=list)  # [{chunk_id, doc_id, score}]
    top_score: float = 0.0
    route: str = "escalate"  # "auto_respond" | "escalate"
    route_reason: str = ""
    route_rule: str = ""  # which router rule fired: R1..R5 or AUTO
    p_auto: float = 0.0  # estimated chance a ticket like this is answerable from the documentation
    threshold_applied: float = 0.0
    outcome: str = "escalated"  # "answered" | "escalated" | "blocked"
    response: dict | None = None  # the drafted reply (PR-01), when generation ran; 'withheld' if blocked
    handover: dict | None = None  # note for the human agent (PR-02), for every escalated ticket
    guardrail: dict | None = None  # validator verdict (FR-07) for every drafted reply
    stage_status: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    flags: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    classifier_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        classifier: Classifier | None = None,
        retriever: Retriever | None = None,
        router: Router | None = None,
        decision_log: DecisionLog | None = None,
        llm: LLMClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.classifier = classifier or Classifier()
        self.retriever = retriever or Retriever(self.settings)
        self.router = router or Router()
        self.decision_log = decision_log
        self.llm = llm or LLMClient(self.settings)

    # ------------------------------------------------------------ generate
    def _generate(self, ticket: Ticket, res: TicketResult, ret: RetrievalResult) -> None:
        d = draft_answer(self.llm, ticket.subject, ticket.body, ret.passages)
        res.response = d.to_dict()
        failed = bool(d.error) and not d.can_answer and d.error != "no_sources"
        res.stage_status["generate"] = "error" if failed else "ok"
        if d.error:
            res.errors.append(f"generate: {d.error}")
        if not d.can_answer:
            why = d.unsure_about or "the documentation does not cover it"
            res.outcome = "escalated"
            res.route_reason = f"{res.route_reason} Then the drafting step declined to answer: {why}"
            return
        v = validate(
            self.llm,
            ticket.subject,
            ticket.body,
            d.reply,
            ret.passages,
            d.citations,
            [ticket.customer_id, ticket.customer_name],
        )
        res.guardrail = v.to_dict()
        res.stage_status["validate"] = "error" if v.error else "ok"
        if v.error:
            res.errors.append(f"validate: {v.error}")
        if v.blocked:
            res.outcome = "escalated"
            res.response["withheld"] = True
            res.route_reason = (
                f"{res.route_reason} The drafted reply was blocked by the checks and not sent: {v.reason}"
            )

    # ------------------------------------------------------------- process
    def process(self, raw: Ticket | Any) -> TicketResult:
        start = time.perf_counter()
        errors: list[str] = []
        ticket = raw if isinstance(raw, Ticket) else normalise(raw)
        try:
            cls = self.classifier.predict(ticket)
            if cls.error:
                errors.append(f"classify: {cls.error}")
            ret = self.retriever.search(ticket.text) if ticket.text.strip() else RetrievalResult(error="empty_query")
            if ret.error and ret.error != "empty_query":
                errors.append(f"retrieve: {ret.error}")
            decision = self.router.decide(cls, ret)
            ks = killswitch.status()
            if ks.active:
                decision = RouteDecision(
                    "escalate",
                    f"Automatic answering is switched off ({ks.reason}), so a person must handle every ticket.",
                    "KS",
                    decision.p_auto,
                    decision.threshold,
                )
            res = TicketResult(
                ticket_id=ticket.ticket_id,
                channel=ticket.channel,
                intent=cls.intent,
                intent_confidence=cls.intent_confidence,
                urgency=cls.urgency,
                urgency_confidence=cls.urgency_confidence,
                p_must_escalate=cls.p_must_escalate,
                intent_alternatives=cls.intent_alternatives,
                retrieved_doc_ids=ret.doc_ids,
                retrieved_passages=[
                    {"chunk_id": p.chunk_id, "doc_id": p.doc_id, "score": p.score} for p in ret.passages
                ],
                top_score=ret.top_score,
                route=decision.route,
                route_reason=decision.reason,
                route_rule=decision.rule,
                p_auto=decision.p_auto,
                threshold_applied=decision.threshold,
                outcome="answered" if decision.route == "auto_respond" else "escalated",
                stage_status={
                    "ingest": "ok",
                    "classify": "fallback" if cls.fallback else "ok",
                    "retrieve": "error" if ret.error else "ok",
                    "generate": "skipped",
                    "validate": "skipped",
                },
                flags=list(ticket.flags),
                errors=errors,
                classifier_version=cls.model_version,
            )
            if decision.route == "auto_respond":
                self._generate(ticket, res, ret)
            if res.outcome == "answered":  # last gate before release: the switch may have been thrown meanwhile
                ks = killswitch.status()
                if ks.active:
                    res.outcome = "escalated"
                    if res.response:
                        res.response["withheld"] = True
                    res.route_reason = (
                        f"{res.route_reason} Held back before sending: "
                        f"automatic answering was switched off ({ks.reason})."
                    )
            if res.outcome == "escalated":
                res.handover = handover(
                    self.llm,
                    ticket,
                    {
                        "intent": res.intent,
                        "confidence": res.intent_confidence,
                        "urgency": res.urgency,
                        "reason": res.route_reason,
                    },
                    ret.passages,
                    use_model=res.route_rule != "KS",
                )
        except Exception as exc:  # absolute last resort: a ticket is never dropped
            res = TicketResult(
                ticket_id=ticket.ticket_id,
                channel=ticket.channel,
                intent="unclear_request",
                intent_confidence=0.0,
                urgency="medium",
                urgency_confidence=0.0,
                p_must_escalate=1.0,
                route="escalate",
                route_reason=f"pipeline error ({type(exc).__name__}); escalated for a human to read",
                outcome="escalated",
                flags=list(ticket.flags),
                errors=errors + [f"pipeline: {type(exc).__name__}: {exc}"],
            )
        res.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        if self.decision_log is not None:
            try:
                self.decision_log.record_ticket(ticket, res, res.classifier_version)
            except Exception as exc:  # logging must never take the run down; reconcile() will show any gap
                res.errors.append(f"decision_log: {type(exc).__name__}: {exc}")
        return res
