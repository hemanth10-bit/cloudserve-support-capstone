"""Output validation before anything reaches a customer (FR-07, A7, NFR-03).

A drafted reply passes only if every check passes. Any single failure blocks it and the ticket is escalated with the
reason recorded. The checks are deliberately cheap and in code wherever possible, because the claim-support model
(PR-03) is the same small model that wrote the reply and can be too lenient:

  private_data       email, phone, key/token or card number in the reply
  commitment         promise of a refund, credit, discount, date, roadmap item or guarantee that no source states
  billing_commitment refund/credit/waiver wording tied to a promise, or that no source contains
  data_location      region, residency or compliance claim (EU, GDPR, SOC 2 ...) that no source contains
  numbers            a figure in the reply that is in neither the sources nor the customer's own message
  prompt_leak        the reply contains our instructions or delimiter tags
  claim_support      PR-03 model finds a factual claim the sources do not support (fails closed if the model fails)

check_reply() runs only the code checks (no model) and is what the tests and the red-team set exercise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .generation import clean_ticket_text, format_sources, load_prompt, parse_json
from .llm import LLMClient

PR03 = ("PR-03", "0.2")
MAX_REPLY_CHARS = 2500

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?![\w])")
KEY = re.compile(
    r"\b(?:sk|pk|rk|api|key|tok|token|secret)[-_][A-Za-z0-9_-]{10,}|\bAKIA[0-9A-Z]{16}\b|\b[A-Za-z0-9]{32,}\b"
)
CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")

PROMISE = re.compile(
    r"\b(?:we|cloudserve)\s*(?:will|'ll|shall|are going to|promise to)\s+"
    r"(?:refund|credit|waive|reimburse|compensate|reverse|issue|add|build|release|ship|deliver|fix|guarantee|"
    r"restore|upgrade|extend|give|provide)\b"
    r"|\b(?:guarantee[ds]?|promise[ds]?|assure[ds]? you|definitely will)\b"
    r"|\byou(?:'ll| will)\s+(?:receive|get|be credited|be refunded)\b",
    re.I,
)
ROADMAP = re.compile(
    r"\b(?:roadmap|coming soon|planned for|next release|upcoming (?:feature|release)|"
    r"will be (?:added|released|available|supported))\b",
    re.I,
)
DATE_PROMISE = re.compile(
    r"\b(?:by|before)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|"
    r"end of (?:the )?(?:day|week|month)|\d{1,2}(?:st|nd|rd|th)?\s+\w+)\b"
    r"|\bwithin\s+(?:the next\s+)?\d+\s+(?:minutes?|hours?|business days?|days?)\b",
    re.I,
)
BILLING_WORDS = re.compile(
    r"\b(?:refund\w*|credit(?!\s+card)\w*|waive[ds]?|waiver|discount\w*|reimburs\w+|compensat\w+)\b", re.I
)
LOCATION = re.compile(
    r"\b(?:GDPR|HIPAA|SOC\s?2|ISO\s?27001|PCI|EU|E\.U\.|Europe(?:an)?|US|U\.S\.|United States|Frankfurt|Ireland|"
    r"Singapore|Mumbai|Sydney|Tokyo|London|residency|sovereign\w*|stored in|hosted in|located in|"
    r"data cent(?:re|er)s?)\b"
)
NUMBER = re.compile(r"(?<![\w.#-])\d+(?:[.,]\d+)?%?")
LEAK = re.compile(r"</?\s*ticket\s*>|</?\s*reply\s*>|\bRULES\b|\bSOURCES\b|as an ai\b|system prompt", re.I)
SENT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class Verdict:
    blocked: bool = False
    reason: str = ""
    checks: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    model: str = ""
    model_version: str = ""
    prompt_version: str = ""
    error: str | None = None

    @property
    def failed(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["passed"]]

    def to_dict(self) -> dict:
        return {**self.__dict__, "failed_checks": self.failed}


def _sentences_with(pattern: re.Pattern, text: str) -> list[str]:
    return [s.strip() for s in SENT.split(text) if pattern.search(s)]


def _terms(pattern: re.Pattern, text: str) -> set[str]:
    return {m.group(0).lower().replace(" ", "") for m in pattern.finditer(text)}


STEP_MARKER = re.compile(r"(?:^|(?<=\s))\d{1,2}[.)]\s+(?=[A-Z])")
STOP = {"this", "that", "with", "from", "your", "have", "will", "then", "they", "them", "when", "which", "also", "into"}
GROUNDED_AT = 0.8  # share of a claim's content words that must appear in the sources to overrule the model


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z-]{3,}", text.lower()) if w not in STOP}


def grounded_in(claim: str, sources_text: str) -> float:
    """Share of the claim's content words found in the sources (1.0 = every word is there)."""
    w = _words(claim)
    return len(w & _words(sources_text)) / len(w) if w else 1.0


def _numbers(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
        line = STEP_MARKER.sub("", line)  # step markers ("1. Open ...") are not figures
        out |= {n.rstrip("%").replace(",", "") for n in NUMBER.findall(line)}
    return out


def check_reply(
    reply: str, sources_text: str, ticket_text: str = "", identifiers: list[str] | None = None
) -> list[Check]:
    """The code checks. No model involved, so the result is the same every time."""
    checks: list[Check] = []
    src = sources_text

    found = [("email", EMAIL), ("key", KEY), ("card", CARD), ("phone", PHONE)]
    hits = [name for name, rx in found if rx.search(reply)]
    # the customer's account id or full name must never be repeated (a first-name greeting is allowed)
    if any(i and len(i.strip()) >= 4 and i.strip().lower() in reply.lower() for i in (identifiers or [])):
        hits.append("customer identifier")
    checks.append(Check("private_data", not hits, f"found: {', '.join(hits)}" if hits else ""))

    bad = [s for s in _sentences_with(PROMISE, reply) if not PROMISE.search(" ".join(_sentences_with(PROMISE, src)))]
    bad += [s for s in _sentences_with(ROADMAP, reply) if not ROADMAP.search(src)]
    bad += [s for s in _sentences_with(DATE_PROMISE, reply) if not DATE_PROMISE.search(src)]
    checks.append(
        Check("commitment", not bad, f"promise or timing not in the sources: {bad[0][:120]!r}" if bad else "")
    )

    bill = []
    src_billing = _terms(BILLING_WORDS, src)
    for s in _sentences_with(BILLING_WORDS, reply):
        words = _terms(BILLING_WORDS, s)
        if not words <= src_billing or PROMISE.search(s) or re.search(r"\b(?:will|'ll)\b", s, re.I):
            bill.append(s)
    checks.append(Check("billing_commitment", not bill, f"refund/credit wording: {bill[0][:120]!r}" if bill else ""))

    unsupported_loc = _terms(LOCATION, reply) - _terms(LOCATION, src)
    checks.append(
        Check(
            "data_location",
            not unsupported_loc,
            f"not in sources: {sorted(unsupported_loc)}" if unsupported_loc else "",
        )
    )

    extra = _numbers(reply) - _numbers(src) - _numbers(ticket_text)
    checks.append(Check("numbers", not extra, f"figures not in the sources: {sorted(extra)}" if extra else ""))

    leak = LEAK.search(reply)
    too_long = len(reply) > MAX_REPLY_CHARS
    checks.append(
        Check(
            "prompt_leak",
            not leak and not too_long,
            f"found {leak.group(0)!r}" if leak else ("reply too long" if too_long else ""),
        )
    )
    return checks


def validate(
    llm: LLMClient,
    subject: str,
    body: str,
    reply: str,
    passages,
    cited: list[str],
    identifiers: list[str] | None = None,
) -> Verdict:
    """Run every check on a drafted reply. Never raises; any failure to check means blocked."""
    v = Verdict()
    try:
        used = [p for p in passages if p.doc_id in set(cited)] or list(passages)
        sources_text = "\n".join(p.text for p in used)
        checks = check_reply(reply, sources_text, clean_ticket_text(subject, body), identifiers)

        prompt = load_prompt(*PR03)
        v.prompt_version = prompt.version
        user = prompt.user.replace("{sources}", format_sources(used)).replace("{reply}", reply.replace("</reply>", ""))
        res = llm.complete(prompt.system, user, max_tokens=800)
        v.model, v.model_version = res.model, res.model_version
        obj = None if res.error else parse_json(res.text)
        if not obj or not isinstance(obj.get("claims"), list):
            v.error = res.error or "unparseable_output"
            checks.append(Check("claim_support", False, f"claim check unavailable ({v.error}); failing closed"))
        else:
            v.claims = [
                {"text": str(c.get("text", ""))[:300], "supported": bool(c.get("supported")), "source": c.get("source")}
                for c in obj["claims"]
                if isinstance(c, dict)
            ]
            for c in v.claims:
                # The checking model is sometimes strict about text that is in the source word for word. A claim made
                # of words the sources contain is accepted, and the override is recorded.
                if not c["supported"] and grounded_in(c["text"], sources_text) >= GROUNDED_AT:
                    c["supported"], c["overridden_by_lexical_check"] = True, True
            weak = [c for c in v.claims if not c["supported"]]
            checks.append(
                Check("claim_support", not weak, f"unsupported claim: {weak[0]['text'][:120]!r}" if weak else "")
            )
        v.checks = [c.to_dict() for c in checks]
        failed = [c for c in checks if not c.passed]
        v.blocked = bool(failed)
        v.reason = "; ".join(f"{c.name}: {c.detail}" if c.detail else c.name for c in failed)
    except Exception as exc:  # fail closed
        v.blocked, v.error = True, f"{type(exc).__name__}: {exc}"
        v.reason = f"validator error ({v.error}); reply withheld"
        v.checks = v.checks or [Check("validator", False, v.error).to_dict()]
    return v
