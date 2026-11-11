"""Ingest: normalise tickets from the four channels into one internal record.

Requirements served: FR-01 (see the Stage 2 PRD). Build Spec criterion A2.

Design rules
- normalise() never raises. Anything it cannot use is recorded in Ticket.flags
  so that downstream components and the decision log can see what happened.
- The original subject, body and channel are always preserved (original_*).
- Labels, if present in the input, are kept for evaluation only. No pipeline
  component may read Ticket.labels when making a decision.
- Output is deterministic: the same raw input always gives the same Ticket,
  including the generated id for tickets that arrive without one.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CHANNELS = ("email", "chat", "docs_comment", "forum")
TIERS = ("enterprise", "business", "standard")
REGIONS = ("north_america", "europe", "asia_pacific", "latin_america")

# Spellings other systems might use for the four channels.
_CHANNEL_ALIASES = {
    "email": "email",
    "e-mail": "email",
    "mail": "email",
    "chat": "chat",
    "live_chat": "chat",
    "livechat": "chat",
    "live chat": "chat",
    "docs_comment": "docs_comment",
    "docs comment": "docs_comment",
    "doc_comment": "docs_comment",
    "docs-comment": "docs_comment",
    "documentation_comment": "docs_comment",
    "docs": "docs_comment",
    "forum": "forum",
    "community": "forum",
    "community_forum": "forum",
}

MAX_TEXT_CHARS = 4000  # cap what goes downstream; the original is kept untouched
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"[ \t  -​  　]+")
_BLANKLINES = re.compile(r"\n{3,}")


@dataclass
class Ticket:
    ticket_id: str
    channel: str  # one of CHANNELS, or "unknown"
    original_channel: str  # exactly what arrived
    original_subject: str
    original_body: str
    subject: str  # cleaned
    body: str  # cleaned
    text: str  # subject + body, cleaned, used downstream
    received_at: str | None  # ISO 8601 UTC, or None if missing/invalid
    customer_id: str
    customer_name: str
    customer_tier: str  # one of TIERS, or "unknown"
    customer_region: str  # one of REGIONS, or "unknown"
    language_fluency: str  # "fluent", "non_fluent" or "unknown"
    flags: list[str] = field(default_factory=list)
    labels: dict[str, Any] | None = None  # evaluation only, never used to decide

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_str(value: Any) -> tuple[str, bool]:
    """Return (string, was_usable). None and non-scalars become ''."""
    if value is None:
        return "", False
    if isinstance(value, str):
        return value, True
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace"), True
    if isinstance(value, (int, float, bool)):
        return str(value), True
    return "", False


def _clean(text: str, flags: list[str], where: str) -> str:
    if _CONTROL.search(text):
        flags.append(f"control_chars_removed:{where}")
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANKLINES.sub("\n\n", text)
    return text.strip()


def _parse_time(value: Any) -> str | None:
    s, ok = _as_str(value)
    if not ok or not s.strip():
        return None
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _choice(value: Any, allowed: Iterable[str], flags: list[str], name: str) -> str:
    s, _ = _as_str(value)
    s = s.strip().lower()
    if s in allowed:
        return s
    flags.append(f"unknown_{name}")
    return "unknown"


def _content_id(*parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8", errors="replace")).hexdigest()
    return "UNKNOWN-" + digest[:10]


def normalise(raw: Any) -> Ticket:
    """Convert one raw ticket (any channel) into a Ticket. Never raises."""
    flags: list[str] = []
    try:
        if not isinstance(raw, dict):
            flags.append("malformed_input")
            text, _ = _as_str(raw)
            raw = {"body": text}

        orig_channel, _ = _as_str(raw.get("channel"))
        channel_key = orig_channel.strip().lower().replace("-", "_") if orig_channel else ""
        channel = _CHANNEL_ALIASES.get(channel_key) or _CHANNEL_ALIASES.get(channel_key.replace("_", " "))
        if channel is None:
            flags.append("unknown_channel")
            channel = "unknown"
        elif channel != channel_key:
            flags.append("channel_alias_mapped")

        orig_subject, subj_ok = _as_str(raw.get("subject"))
        orig_body, body_ok = _as_str(raw.get("body"))
        if not subj_ok and raw.get("subject") is not None:
            flags.append("subject_not_text")
        if not body_ok and raw.get("body") is not None:
            flags.append("body_not_text")

        subject = _clean(orig_subject, flags, "subject")
        body = _clean(orig_body, flags, "body")
        if not body:
            flags.append("empty_body")
        if not subject and not body:
            flags.append("empty_ticket")

        text = (subject + "\n\n" + body).strip() if subject else body
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
            flags.append("text_truncated")

        ticket_id, id_ok = _as_str(raw.get("ticket_id"))
        ticket_id = ticket_id.strip()
        if not ticket_id:
            ticket_id = _content_id(str(orig_channel), orig_subject, orig_body)
            flags.append("missing_ticket_id")

        labels = raw.get("labels")
        return Ticket(
            ticket_id=ticket_id,
            channel=channel,
            original_channel=orig_channel,
            original_subject=orig_subject,
            original_body=orig_body,
            subject=subject,
            body=body,
            text=text,
            received_at=_parse_time(raw.get("received_at")) or _flag_time(flags),
            customer_id=_as_str(raw.get("customer_id"))[0].strip(),
            customer_name=_as_str(raw.get("customer_name"))[0].strip(),
            customer_tier=_choice(raw.get("customer_tier"), TIERS, flags, "tier"),
            customer_region=_choice(raw.get("customer_region"), REGIONS, flags, "region"),
            language_fluency=_choice(raw.get("language_fluency"), ("fluent", "non_fluent"), flags, "fluency"),
            flags=flags,
            labels=labels if isinstance(labels, dict) else None,
        )
    except Exception as exc:  # last resort: ingest must not take the run down
        return Ticket(
            ticket_id=_content_id(repr(raw)[:500]),
            channel="unknown",
            original_channel="",
            original_subject="",
            original_body="",
            subject="",
            body="",
            text="",
            received_at=None,
            customer_id="",
            customer_name="",
            customer_tier="unknown",
            customer_region="unknown",
            language_fluency="unknown",
            flags=["ingest_error:" + type(exc).__name__],
        )


def _flag_time(flags: list[str]) -> None:
    flags.append("missing_or_invalid_timestamp")
    return None


def load_raw_tickets(path: str | Path) -> list[Any]:
    """Read a ticket file. Accepts a JSON list, {"tickets": [...]}, or JSON Lines.

    A file that cannot be read at all raises ValueError with a clear message;
    individual bad records are returned as-is so normalise() can flag them.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ValueError(f"cannot read ticket file {p}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        items: list[Any] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                items.append(line)  # kept so it shows up as malformed, not silently dropped
        if not items:
            raise ValueError(f"{p} is neither valid JSON nor JSON Lines")
        return items
    if isinstance(data, dict):
        for key in ("tickets", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"{p}: expected a list of tickets, got {type(data).__name__}")


def load_tickets(path: str | Path) -> list[Ticket]:
    return [normalise(r) for r in load_raw_tickets(path)]


def _demo(path: str) -> None:
    """Print one normalised ticket per channel (Build Spec day-one check)."""
    seen: dict[str, Ticket] = {}
    for t in load_tickets(path):
        seen.setdefault(t.channel, t)
    for channel in CHANNELS:
        t = seen.get(channel)
        print(f"--- {channel} ---")
        if t is None:
            print("(no ticket of this channel in the file)")
            continue
        d = t.to_dict()
        d.pop("labels")
        print(json.dumps(d, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import sys

    _demo(sys.argv[1] if len(sys.argv) > 1 else "data/development_tickets.json")
