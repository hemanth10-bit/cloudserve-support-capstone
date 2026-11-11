"""TC-FR01: ingest behaviour (FR-01, A2)."""

import json
from pathlib import Path

import pytest

from src.ingest import CHANNELS, Ticket, load_raw_tickets, load_tickets, normalise

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def dev_raw():
    return json.loads((DATA / "development_tickets.json").read_text())


def first_of(raw, channel):
    return next(t for t in raw if t["channel"] == channel)


@pytest.mark.parametrize("channel", CHANNELS)
def test_one_ticket_per_channel_gives_same_shape(dev_raw, channel):
    t = normalise(first_of(dev_raw, channel))
    assert isinstance(t, Ticket)
    assert t.channel == channel
    assert t.original_channel == channel
    assert t.ticket_id and t.text
    assert t.customer_tier in ("enterprise", "business", "standard")
    assert t.flags == [] or all(isinstance(f, str) for f in t.flags)


def test_all_channels_share_identical_field_set(dev_raw):
    keysets = {tuple(sorted(normalise(first_of(dev_raw, c)).to_dict())) for c in CHANNELS}
    assert len(keysets) == 1


def test_original_text_preserved(dev_raw):
    raw = first_of(dev_raw, "email")
    t = normalise(raw)
    assert t.original_subject == raw["subject"]
    assert t.original_body == raw["body"]


def test_chat_ticket_with_empty_subject_is_ok(dev_raw):
    t = normalise(first_of(dev_raw, "chat"))
    assert t.subject == ""
    assert t.text == t.body and t.body


def test_every_development_and_validation_ticket_normalises_cleanly():
    for name in ("development_tickets.json", "validation_tickets.json"):
        raw = load_raw_tickets(DATA / name)
        tickets = [normalise(r) for r in raw]
        assert len(tickets) == len(raw)
        assert all(t.flags == [] for t in tickets), name


def test_empty_body_does_not_raise():
    t = normalise({"ticket_id": "X-1", "channel": "email", "subject": "Help", "body": ""})
    assert "empty_body" in t.flags
    assert t.text == "Help"


def test_missing_and_null_fields_do_not_raise():
    t = normalise({"channel": None, "body": None, "customer_tier": None, "received_at": None})
    assert t.channel == "unknown"
    assert "empty_ticket" in t.flags and "missing_ticket_id" in t.flags
    assert t.customer_tier == "unknown"


@pytest.mark.parametrize("bad", [None, 42, "just a string", [], ["a"], {"x": 1}, b"\xff\xfe"])
def test_non_dict_or_garbage_input_does_not_raise(bad):
    t = normalise(bad)
    assert isinstance(t, Ticket)
    assert t.ticket_id


def test_unicode_emoji_and_non_latin_text_survive():
    body = "ログインできません 🙏 мой пароль не работает"
    t = normalise({"ticket_id": "U-1", "channel": "chat", "body": body})
    assert "ログイン" in t.text and "🙏" in t.text and "пароль" in t.text
    assert t.original_body == body


def test_control_characters_removed_but_original_kept():
    t = normalise({"ticket_id": "C-1", "channel": "email", "body": "hello\x00 wor\x07ld"})
    assert t.body == "hello world"
    assert t.original_body == "hello\x00 wor\x07ld"
    assert "control_chars_removed:body" in t.flags


def test_very_long_body_is_capped_downstream_but_kept_in_original():
    t = normalise({"ticket_id": "L-1", "channel": "email", "body": "a" * 50000})
    assert len(t.text) <= 4000 and "text_truncated" in t.flags
    assert len(t.original_body) == 50000


def test_channel_aliases_are_mapped():
    assert normalise({"channel": "Live Chat", "body": "x"}).channel == "chat"
    assert normalise({"channel": "community-forum", "body": "x"}).channel == "forum"
    t = normalise({"channel": "carrier pigeon", "body": "x"})
    assert t.channel == "unknown" and t.original_channel == "carrier pigeon"


def test_bad_timestamp_is_flagged_not_fatal():
    t = normalise({"ticket_id": "T-1", "channel": "chat", "body": "x", "received_at": "yesterday-ish"})
    assert t.received_at is None and "missing_or_invalid_timestamp" in t.flags


def test_timestamp_normalised_to_utc():
    t = normalise({"ticket_id": "T-2", "channel": "chat", "body": "x", "received_at": "2026-03-14T10:22:00+01:00"})
    assert t.received_at == "2026-03-14T09:22:00Z"


def test_deterministic_including_generated_ids():
    raw = {"channel": "email", "subject": "s", "body": "b"}
    a, b = normalise(dict(raw)), normalise(dict(raw))
    assert a.to_dict() == b.to_dict()
    assert a.ticket_id.startswith("UNKNOWN-")


def test_labels_are_kept_for_evaluation_only(dev_raw):
    t = normalise(dev_raw[0])
    assert t.labels is not None and "intent" in t.labels
    assert normalise({"ticket_id": "N-1", "channel": "chat", "body": "x"}).labels is None


def test_load_tickets_accepts_list_wrapper_and_jsonl(tmp_path):
    rec = {"ticket_id": "J-1", "channel": "chat", "body": "hi"}
    (tmp_path / "list.json").write_text(json.dumps([rec, rec]))
    (tmp_path / "wrap.json").write_text(json.dumps({"tickets": [rec]}))
    (tmp_path / "lines.jsonl").write_text(json.dumps(rec) + "\n\nnot json at all\n" + json.dumps(rec))
    assert len(load_tickets(tmp_path / "list.json")) == 2
    assert len(load_tickets(tmp_path / "wrap.json")) == 1
    lines = load_tickets(tmp_path / "lines.jsonl")
    assert len(lines) == 3  # the bad line is kept and flagged, not dropped
    assert any("malformed_input" in t.flags for t in lines)


def test_unreadable_file_gives_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="cannot read"):
        load_raw_tickets(tmp_path / "missing.json")
    (tmp_path / "empty.json").write_text("")
    with pytest.raises(ValueError):
        load_raw_tickets(tmp_path / "empty.json")
