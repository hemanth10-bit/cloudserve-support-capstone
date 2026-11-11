"""Kill switch: stop all automatic answering immediately, without a deployment (Governance Framework section 5).

While it is on, every ticket is escalated to a person (route rule KS) and no model is called. It is checked for every
ticket, and again just before a drafted reply would be released, so a ticket already in flight is held back too.

Two ways to turn it on (either is enough):
  - create the flag file (default storage/KILL_SWITCH):   python -m src.killswitch on "reason"
  - set the environment variable AUTO_RESPOND_DISABLED=1  (needs a restart of the process, so the file is preferred)

    python -m src.killswitch on "reason" | off | status

Checking is a file-existence test, so it takes effect on the very next ticket. If the check itself fails, the system
fails closed: it behaves as if the switch were on.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT


def flag_path() -> Path:
    p = Path(os.environ.get("KILL_SWITCH_FILE", "storage/KILL_SWITCH"))
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass(frozen=True)
class Status:
    active: bool
    reason: str = ""


def status() -> Status:
    try:
        if os.environ.get("AUTO_RESPOND_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
            return Status(True, "environment variable AUTO_RESPOND_DISABLED is set")
        p = flag_path()
        if p.exists():
            note = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return Status(True, f"flag file present ({note[0] if note else 'no reason given'})")
        return Status(False)
    except Exception as exc:  # fail closed
        return Status(True, f"kill switch could not be checked ({type(exc).__name__}); failing closed")


def turn_on(reason: str = "no reason given") -> Path:
    p = flag_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{reason} (set {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})\n", encoding="utf-8")
    return p


def turn_off() -> bool:
    p = flag_path()
    if p.exists():
        p.unlink()
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a[0] if a else "status"
    if cmd == "on":
        print(f"Kill switch ON ({turn_on(' '.join(a[1:]) or 'no reason given')}). All tickets will be escalated.")
    elif cmd == "off":
        print("Kill switch OFF." if turn_off() else "Kill switch was already off.")
        s = status()
        if s.active:
            print(f"Still active: {s.reason}")
            return 1
    elif cmd == "status":
        s = status()
        print(f"Kill switch {'ON: ' + s.reason if s.active else 'OFF'}")
    else:
        print("usage: python -m src.killswitch on [reason] | off | status")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
