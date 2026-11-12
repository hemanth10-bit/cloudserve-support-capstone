# Notes for the Stage 5 PRD revision (v1.0 -> v2.0)

Collected as the build goes so the revision log records real triggers, not tidying.

| Req / assumption | What v1.0 said | What the build found | Date |
|---|---|---|---|
| FR-03 acceptance criterion | "Feature_request and unclear_request tickets return no passages." | Retrieval score cannot do this: feature requests are on-topic and score high (median 0.72). Replace with intent gating (FR-05) keeping those classes off the auto-answer path. | 2026-11-10 |
| FR-03 target | Retrieval hit rate provisional 80% | Measured on dev: expected article in top 5 for 98.9% (98.0% at threshold 0.40); first for 88.8%. Raise target to 95% at top 5. | 2026-11-10 |
| Assumption 1 (answerable label reflects what a reply could cite) | Held as true | Partly wrong: identical ticket texts carry conflicting `answerable_from_docs` labels (41 texts); best accuracy from text alone is 86.4%. | 2026-11-10 |
| Assumption 3 (dev/validation/hidden are independent samples) | Held as true | Wrong for validation: 62 of 80 validation bodies appear verbatim in development; dev has only 215 distinct texts. Validation figures are optimistic. | 2026-11-10 |
| NFR-03 accuracy targets | Precision 85% etc. judged on validation | Also report grouped cross-validation by text on dev as the honest generalisation estimate. Urgency ceiling from text is 73%, so an urgency target must reflect that. | 2026-11-10 |
| Tooling assumption (course requirements.txt) | Pinned versions install | Do not install on Python 3.12; re-resolved (decisions.md D-01). | 2026-11-10 |
| NFR-03 urgency | Urgency measured like intent | Urgency is only 49.8% predictable out of fold (baseline 45.2%); text ceiling about 73%. Change the requirement from an accuracy target to "report urgency accuracy against the majority baseline and do not rely on it for prioritisation"; FR-10 (urgency-based ordering) is weak evidence-wise and is the first to cut. | 2026-11-10 |
| FR-02 target wording | "Intent precision 85% or better on validation" | Validation overlaps development, so use grouped cross-validation on development (95.8% accuracy, 96.0% macro precision, ECE 0.035) as the headline and validation as a labelled optimistic figure. | 2026-11-10 |
| Process | Build Spec's cross-check assumed validation is held out | Found a leakage bug in my own first cross-validation (grouping by subject + body gave 99.0% against the honest 95.8%). Lesson recorded for the report's reflection. | 2026-11-10 |
| FR-04 / success measure "Escalation rate" | Target 30% or lower; threshold "determined from data" | The labels put the floor at 37.8% and the pre-generation router leaves 15 of 58 auto-routed validation tickets labelled unanswerable. The threshold is a cost argument with a measured trade-off (D-08), not an accuracy target. State the achievable range in v2.0 once the end-to-end run (B-11) exists. | 2026-11-10 |
| FR-05 | Separate wording checks for billing commitments and data-residency | Implemented in the validator (B-09, D-11): billing and data-location wording checks block the reply and escalate. | 2026-11-10 |
| Assumption (new) | none | Router threshold assumes later stages let 20% of wrong auto-routes through. Unmeasured. Must be measured and the threshold re-fitted. | 2026-11-10 |
| NFR-01 | p95 latency under 3 s for the automated path | Not met: p95 17.9 s, median 5.9 s on the validation gate run (free 8B model, two to three calls per ticket). Revise the target or the architecture, with the numbers. | 2026-11-11 |
| FR-04 | Threshold from cost argument with 20% pass-through | Measured pass-through on validation was 73% (11 of 15); re-measure on development and re-fit (D-13). | 2026-11-11 |
| FR-04 | Threshold decision | Author decided 2026-11-12 to keep 0.64 despite the measured pass-through; recorded as a judgement call, not a fitted value (D-17). | 2026-11-12 |
