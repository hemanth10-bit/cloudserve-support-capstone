# Demo runbook (the 7+ minute live part, minutes 7 to 14 of the video)

Everything runs on your laptop. Use a large terminal font (18+) and browser zoom 125%.

## The day before: rehearse and warm the cache
1. `cd cloudserve-support && source .venv/bin/activate`
2. `python -m src.killswitch status` must say OFF.
3. Warm the cache and make a fresh finished run (about 95 s the first time, about 10 s after):
   `python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/demo_val`
   Expect: `80/80 processed`, exit 0. Counts are about 51 answered, 29 escalated, 7 blocked (D-22). If yours differ slightly, say so on camera and quote the screen, not these notes.
4. `python -m ui` in a second terminal, open http://127.0.0.1:8000, click each example once so the model calls are cached.
5. Keep `evaluation/results/gate_validation` as the recorded run (49 answered, 31 escalated, 9 blocked).

## On camera (times add up to about 9 minutes; trim to fit)
| Min | Do | Say |
|---|---|---|
| 1.5 | Browser, Try a ticket, example 1 (VAL-0004, MFA code). Click Process ticket. Watch the track light up. | Each step is a decision the system makes and logs. Route cleared it for a drafted reply (68%, needs 64%). The reply cites DOC-AUTH-002. Seven checks passed. |
| 1.5 | Example 3 (VAL-0009, retention period / compliance). | A person handles compliance by rule R2, in code, not by a model. No reply was drafted. The agent gets a hand-over note: summary, what the customer wants, the article, what it is unsure about. |
| 2 | Run results tab, gate_validation. Filter Outcome to Reply blocked. Click VAL-0028 and VAL-0047. | The checks blocked these. "Usually due to unpinned dependencies" and "search endpoints may have different limits" are not in the article. It blocked 9 here. Some blocks are over-cautious (the checker is the same small model): cost is automation, not safety. |
| 1.5 | Terminal: run the harness live on the validation file into a new folder (`--output evaluation/results/live_take`). While it runs, show the command line only: one command, two paths. | Unattended: no clicks. It finishes with 80 of 80 processed, exit code 0, and the decision log reconciles (276 of 276 in the gate run). Rerun is fast because replies are cached; say so. |
| 1 | `python -m evaluation.harness ...` output folder: open `metrics.md`, or the Run results tab for live_take. | Compare with the gate run: different counts at the checking step (D-22). Say it plainly: the checker is not repeatable; the route and the always-escalate result are. |
| 1 | Kill switch. Click the pill, type "demo", Stop. Back to Try a ticket, example 1 again. | Now the same ticket goes to a person with rule KS and no model call. Resume. Also available as `python -m src.killswitch on "reason"`. |
| 0.5 | Terminal: `python -m pytest tests/ -q` (about 20 s). | 204 tests, one command, fake model. |

## If something goes wrong
- Page will not load: `python -m ui` was not started or port 8000 is busy. Fallback: show `evaluation/results/gate_validation/metrics.md` and `results.jsonl` in the terminal.
- Model is slow or rate limited: the cached examples answer in about a second. If a live harness run stalls, stop it and show the finished `demo_val` or `gate_validation` folder; do not wait on camera.
- A sample gives a different outcome than the label next to it: say so ("a live run can differ"); that is the finding in D-22.
- Kill switch left on after a take: `python -m src.killswitch off` and check `status`.

## Things not to claim
- Do not say the hallucination rate is under 5%: it is 0 of 60 with an upper bound of 6.0%, one reviewer.
- Do not quote 100% intent accuracy: it is optimistic; quote 95.8% out of fold.
- Do not say the latency target was met (p95 17.9 s against 3 s).
