"""Grounded answer drafting (PR-01, FR-06) and escalation hand-over notes (PR-02, FR-08).

Both functions never raise. If the model is unavailable or returns something unusable:
  - draft_answer returns can_answer=False with the reason, so the ticket is escalated rather than answered;
  - handover returns a template note built from facts the pipeline already has, so an escalation is never empty.

Citations are checked in code, not trusted: every cited id must be one of the passages that were retrieved for this
ticket, and a reply with no valid citation is rejected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient, LLMResult

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
MAX_TICKET_CHARS = 3000
CITE = re.compile(r"\[(DOC-[A-Z0-9-]+)\]")
TAGS = re.compile(r"</?\s*ticket\s*>", re.I)


@dataclass
class Prompt:
    version: str
    system: str
    user: str


def load_prompt(prefix: str, version: str) -> Prompt:
    """Read prompts/<prefix>.v<version>.txt: comment header, then SYSTEM ... USER ... sections."""
    path = next(PROMPT_DIR.glob(f"{prefix}_*.v{version}.txt"))
    body = path.read_text(encoding="utf-8").split("# ---- prompt text starts below this line ----\n", 1)[1]
    system, user = body.split("\nUSER\n", 1)
    return Prompt(f"{prefix}.v{version}", system.removeprefix("SYSTEM\n").strip(), user.strip())


PR01 = ("PR-01", "0.2")
PR02 = ("PR-02", "0.2")


def clean_ticket_text(subject: str, body: str) -> str:
    """Remove our delimiter tags from customer text so it cannot close the <ticket> block, and bound the length."""
    text = TAGS.sub(" ", f"{subject}\n{body}".strip())
    return text[:MAX_TICKET_CHARS]


def format_sources(passages) -> str:
    return "\n".join(f"[{p.doc_id}] ({p.title}, {p.section}) {p.text}" for p in passages) or "(none retrieved)"


def parse_json(text: str) -> dict | None:
    """Accept a bare JSON object, or one wrapped in a code fence or prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    for candidate in (t, t[t.find("{") : t.rfind("}") + 1] if "{" in t else ""):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


@dataclass
class Draft:
    can_answer: bool = False
    reply: str = ""
    citations: list = field(default_factory=list)
    unsure_about: str = ""
    model: str = ""
    model_version: str = ""
    prompt_version: str = ""
    cached: bool = False
    attempts: int = 0
    error: str | None = None  # why the model's output was not usable, if it was not
    uncited_sentences: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _uncited(reply: str) -> int:
    """Count sentences that state something but carry no [DOC-...] id (reported, not enforced here)."""
    n = 0
    for s in re.split(r"(?<=[.!?])\s+|\n+", reply):
        s = s.strip()
        if len(s.split()) >= 6 and not CITE.search(s) and not s.endswith(":"):
            n += 1
    return n


def draft_answer(llm: LLMClient, subject: str, body: str, passages) -> Draft:
    prompt = load_prompt(*PR01)
    d = Draft(prompt_version=prompt.version)
    if not passages:
        d.unsure_about, d.error = "No documentation passages were retrieved.", "no_sources"
        return d
    user = prompt.user.replace("{sources}", format_sources(passages)).replace(
        "{ticket}", clean_ticket_text(subject, body)
    )
    res: LLMResult = llm.complete(prompt.system, user)
    d.model, d.model_version, d.cached, d.attempts = res.model, res.model_version, res.cached, res.attempts
    if res.error:
        d.error, d.unsure_about = f"llm_unavailable: {res.error}", "The model could not be reached."
        return d
    obj = parse_json(res.text)
    if obj is None or "can_answer" not in obj:
        d.error, d.unsure_about = "unparseable_output", "The model did not return valid JSON."
        return d
    reply = str(obj.get("reply") or "").strip()
    allowed = {p.doc_id for p in passages}
    cited = [str(c) for c in (obj.get("citations") or []) if isinstance(c, (str, int))]
    inline = CITE.findall(reply)
    d.unsure_about = str(obj.get("unsure_about") or "")
    d.reply = reply
    d.citations = sorted(set(cited) | set(inline))
    if not bool(obj.get("can_answer")) or not reply:
        d.can_answer = False
        d.error = None if not bool(obj.get("can_answer")) else "empty_reply"
        d.reply = ""
        return d
    bad = [c for c in d.citations if c not in allowed]
    if bad or not d.citations:
        d.can_answer, d.reply = False, ""
        d.error = f"invalid_citation: {bad}" if bad else "no_citation"
        d.unsure_about = d.unsure_about or "The reply could not be tied to the retrieved documentation."
        return d
    d.can_answer = True
    d.uncited_sentences = _uncited(reply)
    return d


def handover(llm: LLMClient, ticket, result_fields: dict, passages) -> dict:
    """A short note for the human agent. Falls back to a template so the escalation is never empty."""
    template = {
        "summary": (
            f"Customer ticket classified as {result_fields['intent'].replace('_', ' ')} "
            f"(urgency {result_fields['urgency']}). Escalated: {result_fields['reason']}"
        ),
        "customer_wants": "Not summarised automatically; read the ticket.",
        "relevant_docs": sorted({p.doc_id for p in passages}),
        "unsure_about": "Model summary unavailable; this note was built from the routing facts only.",
        "suggested_next_step": "Read the ticket and answer or route it.",
        "source": "template",
    }
    prompt = load_prompt(*PR02)
    user = (
        prompt.user.replace("{intent}", result_fields["intent"])
        .replace("{confidence}", str(result_fields["confidence"]))
        .replace("{urgency}", result_fields["urgency"])
        .replace("{reason}", result_fields["reason"])
        .replace("{sources}", format_sources(passages))
        .replace("{ticket}", clean_ticket_text(ticket.subject, ticket.body))
    )
    try:
        res = llm.complete(prompt.system, user, max_tokens=400)
        obj = None if res.error else parse_json(res.text)
        if not obj or "summary" not in obj:
            return {**template, "error": res.error or "unparseable_output", "prompt_version": prompt.version}
        allowed = {p.doc_id for p in passages}
        docs = [d for d in (obj.get("relevant_docs") or []) if d in allowed]
        words = str(obj.get("summary", "")).split()
        return {
            "summary": " ".join(words[:60]),
            "customer_wants": str(obj.get("customer_wants", "")),
            "relevant_docs": docs,
            "unsure_about": str(obj.get("unsure_about", "")),
            "suggested_next_step": str(obj.get("suggested_next_step", "")),
            "source": "model",
            "model_version": res.model_version,
            "prompt_version": prompt.version,
        }
    except Exception as exc:
        return {**template, "error": f"{type(exc).__name__}: {exc}", "prompt_version": prompt.version}
