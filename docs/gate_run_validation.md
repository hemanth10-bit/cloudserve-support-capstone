# Gate run on the validation file (B-11, 2026-11-11)

Run once, unattended, no restarts, nothing tuned afterwards on this file. Command:
`python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/gate_validation`
(`evaluation/results/` is git-ignored; the figures below are copied from `metrics.json`). Started 11:35 UTC, the first run took 616 s
(about half the drafts were not cached). Validation run count: **1**. A second run on a renamed copy with the labels removed
(`hidden_like_tickets.json`, 80 tickets) exited 0, gave **80 of 80 identical decisions and replies**, and reconciled (7 s, cached).

| Measure | Result | Target / note |
|---|---|---|
| Tickets processed / outcomes | 80 / 80; 49 answered, 31 escalated (9 of those blocked by the validator) | every ticket has an outcome (A9): met |
| Decision log | 276 decisions expected, 276 logged, 0 missing, reconciles | NFR-05: met |
| Always-escalate tickets answered automatically | **0 of 14** | FR-05: met |
| Intent accuracy / macro precision | 100% / 100% on 80 tickets (62 have text seen in development, 18 unseen; both subsets 100%) | **Optimistic**: trained on overlapping text. The honest estimate is the out-of-fold figure, 95.8% (strict paraphrase lower bound 89.6%). The 18 unseen tickets are too few to separate these. |
| Intent calibration error | 0.13% (out-of-fold figure: 3.5%) | NFR-03 within 5 points: met, same caveat |
| Urgency accuracy | 46.3% vs 43.8% majority baseline; calibration error 8.7% | labels are not a function of the text (ceiling ~73%); weak |
| Retrieval hit rate, top 5 | 96.2% of 53 answerable tickets | A4: met |
| Route agreement with labels | 75.0% | labels put the escalation floor at 37.8%; we escalate 38.8% |
| First-contact resolution (answered / total) | 61.3% | Brief target not stated against this definition |
| Escalation rate | 38.8% | Brief target ≤30% conflicts with the labels' own floor (data_findings.md) |
| Answered tickets labelled answerable | 38 of 49 (**77.6%**); 11 labelled not answerable | not a hallucination rate: those replies were grounded in cited text; the labels are noisy |
| Missed automation (labelled answerable, not always-escalate, but escalated) | 11 | the cost of blocking and of the router threshold |
| Private-data detections in replies | 0 | NFR-04 |
| Validator blocks | 9, all `claim_support`; 4 of the 9 were labelled not answerable | |
| Latency | median 5.9 s, p95 17.9 s (uncached drafted tickets: median 11.6 s) | **NFR-01 (p95 under 3 s) NOT met** |
| Hallucination rate, citation accuracy | not yet measured independently (PR-04 judge, ground-truth comparison still to do) | NFR-03 |
| Replies with a sentence lacking an inline citation | 25 of 49 | the cited-id list is always valid (0 rejected) |

## What this changes
1. **The pass-through assumption was wrong.** The router threshold (0.64) assumed 20% of wrongly auto-routed tickets would survive
   generation and validation. Measured here: of 15 auto-routed tickets that the labels say were not answerable, **11 survived (73%)**;
   the validator stopped 4. On the 60-ticket development sample it was 5 of 6. The validator cannot see what the labels see, because
   those replies are grounded in the documentation. Setting the threshold from this validation figure would be tuning on the held-out
   set, so the re-fit must use development data (see D-13).
2. **NFR-01 latency fails** with a free 8B model over the network: drafting plus claim checking is two or three calls per ticket.
   Caching only helps on repeat input. To be recorded in the PRD revision (Stage 5) with the measured numbers; not hidden.
3. **Intent accuracy on validation is not the number to quote.** Quote the out-of-fold 95.8% and say how the validation overlap inflates this.
