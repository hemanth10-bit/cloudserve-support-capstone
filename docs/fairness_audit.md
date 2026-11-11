# Fairness audit (B-14, NFR-06) - 2026-11-11

**Verdict in one paragraph.** On 216 development tickets (one per distinct text) no gap between customer groups is
statistically supported, and equally the sample cannot show that gaps are under 5 points. Several raw gaps exceed 5 points
(for example answered share by tier, 47% standard vs 65% enterprise; route agreement by region, 55% Europe vs 71-77% elsewhere,
p = 0.08), none reaches p < 0.05, and 28 comparisons were made, so one or two would be expected by chance. Group fields
(fluency, region, tier, channel) are not inputs to classification, routing, generation or validation (enforced by a test),
so any real gap would come from the wording of tickets. The validation set (80 tickets) is too small to judge any
group split except tier, which is also not significant. **NFR-06 is therefore "no evidence of a gap, not shown to be absent".**

Method: `python -m evaluation.fairness --results DIR --tickets FILE`. Metrics per group with 95% Wilson intervals, chi-square test
across groups, gap between the highest and lowest group with an interval. Groups under 30 tickets are listed but never used to
claim a gap or its absence. "Quality" is not one number: answered share, answered share among answerable tickets (controls for a
group asking more unanswerable questions), route agreement with labels, right-answer share, validator block rate, intent accuracy
(optimistic on development data) and retrieval hit rate.

Not covered: reply *quality* as judged by people (the human review sheet carries no group labels; if the reviewers also
mark the sheet, group-level quality can be added); the validation set's Latin America group (7 tickets).

---
## Development results (n=216)

- Source: `evaluation/results/dev_distinct_out`; tickets audited: 216; limit 5.0 percentage points; groups under 30 tickets are shown but never used to claim a gap or its absence.
- Group fields are not inputs to any decision (tests/test_fairness.py). Intent accuracy on development data is optimistic.

## By fluency

- **answered_rate**: fluent 55.3% (n=114), non_fluent 50.0% (n=102). gap 5.3 pp (95% interval -8.0 to 18.6, p=0.5241) **gap above the limit, but the sample cannot tell it from chance.**
- **answered_given_answerable**: fluent 67.7% (n=65), non_fluent 63.1% (n=65). gap 4.6 pp (95% interval -11.7 to 20.9, p=0.7123) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
- **route_agrees_with_label**: fluent 68.4% (n=114), non_fluent 68.6% (n=102). gap 0.2 pp (95% interval -12.2 to 12.6, p=1.0) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
- **right_answer_rate**: fluent 69.8% (n=63), non_fluent 80.4% (n=51). gap 10.6 pp (95% interval -5.1 to 26.3, p=0.2847) **gap above the limit, but the sample cannot tell it from chance.**
- **blocked_rate**: fluent 19.2% (n=78), non_fluent 19.0% (n=63). gap 0.2 pp (95% interval -12.8 to 13.2, p=1.0) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
- **intent_correct**: fluent 100.0% (n=114), non_fluent 100.0% (n=102). gap 0.0 pp (95% interval 0.0 to 0.0, p=None) **within the limit, and the interval is inside it.**
- **retrieval_hit_top5**: fluent 92.0% (n=75), non_fluent 88.6% (n=70). gap 3.4 pp (95% interval -6.3 to 13.1, p=0.6766) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**

## By region

- **answered_rate**: asia_pacific 53.1% (n=49), europe 51.7% (n=58), latin_america 61.8% (n=34), north_america 49.3% (n=75). gap 12.5 pp (95% interval -7.4 to 32.4, p=0.6854) **gap above the limit, but the sample cannot tell it from chance.**
- **answered_given_answerable**: asia_pacific 69.2% (n=26), europe 52.5% (n=40), latin_america 81.0% (n=21), north_america 67.4% (n=43). gap 14.9 pp (95% interval -6.0 to 35.8, p=0.2439) **gap above the limit, but the sample cannot tell it from chance.**
- **route_agrees_with_label**: asia_pacific 71.4% (n=49), europe 55.2% (n=58), latin_america 76.5% (n=34), north_america 73.3% (n=75). gap 21.3 pp (95% interval 2.1 to 40.5, p=0.0791) **gap above the limit, but the sample cannot tell it from chance.**
- **right_answer_rate**: asia_pacific 69.2% (n=26), europe 70.0% (n=30), latin_america 81.0% (n=21), north_america 78.4% (n=37). gap 8.4 pp (95% interval -12.7 to 29.5, p=0.6161) **gap above the limit, but the sample cannot tell it from chance.**
- **blocked_rate**: asia_pacific 10.3% (n=29), europe 25.0% (n=40), latin_america 19.2% (n=26), north_america 19.6% (n=46). gap 5.4 pp (95% interval -12.3 to 23.1, p=0.7298) **gap above the limit, but the sample cannot tell it from chance.**
- **intent_correct**: asia_pacific 100.0% (n=49), europe 100.0% (n=58), latin_america 100.0% (n=34), north_america 100.0% (n=75). gap 0.0 pp (95% interval 0.0 to 0.0, p=None) **within the limit, and the interval is inside it.**
- **retrieval_hit_top5**: asia_pacific 83.3% (n=30), europe 90.7% (n=43), latin_america 91.3% (n=23), north_america 93.9% (n=49). gap 10.6 pp (95% interval -4.3 to 25.5, p=0.3082) **gap above the limit, but the sample cannot tell it from chance.**

## By tier

- **answered_rate**: business 55.6% (n=72), enterprise 64.7% (n=34), standard 47.3% (n=110). gap 17.4 pp (95% interval -1.2 to 36.0, p=0.1736) **gap above the limit, but the sample cannot tell it from chance.**
- **answered_given_answerable**: business 65.3% (n=49), enterprise 77.3% (n=22), standard 61.0% (n=59). gap 4.3 pp (95% interval -13.9 to 22.5, p=0.7953) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
- **route_agrees_with_label**: business 69.4% (n=72), enterprise 73.5% (n=34), standard 66.4% (n=110). gap 7.1 pp (95% interval -10.2 to 24.4, p=0.7185) **gap above the limit, but the sample cannot tell it from chance.**
- **right_answer_rate**: business 80.0% (n=40), enterprise 77.3% (n=22), standard 69.2% (n=52). gap 10.8 pp (95% interval -6.8 to 28.4, p=0.3541) **gap above the limit, but the sample cannot tell it from chance.**
- **blocked_rate**: business 16.7% (n=48), enterprise 18.5% (n=27), standard 21.2% (n=66). gap 4.5 pp (95% interval -9.9 to 18.9, p=0.7137) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
- **intent_correct**: business 100.0% (n=72), enterprise 100.0% (n=34), standard 100.0% (n=110). gap 0.0 pp (95% interval 0.0 to 0.0, p=None) **within the limit, and the interval is inside it.**
- **retrieval_hit_top5**: business 86.8% (n=53), enterprise 92.0% (n=25), standard 92.5% (n=67). gap 5.7 pp (95% interval -5.4 to 16.8, p=0.4621) **gap above the limit, but the sample cannot tell it from chance.**

## By channel

- **answered_rate**: chat 55.6% (n=72), docs_comment 51.5% (n=33), email 51.8% (n=83), forum 50.0% (n=28). gap 4.1 pp (95% interval -16.5 to 24.7, p=0.8769) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
- **answered_given_answerable**: chat 65.4% (n=52), docs_comment 56.2% (n=16), email 66.0% (n=47), forum 73.3% (n=15). gap 0.6 pp (95% interval -18.1 to 19.3, p=1.0) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
- **route_agrees_with_label**: chat 70.8% (n=72), docs_comment 57.6% (n=33), email 68.7% (n=83), forum 75.0% (n=28). gap 13.2 pp (95% interval -6.7 to 33.1, p=0.3868) **gap above the limit, but the sample cannot tell it from chance.**
- **right_answer_rate**: chat 85.0% (n=40), docs_comment 52.9% (n=17), email 72.1% (n=43), forum 78.6% (n=14). gap 12.9 pp (95% interval -4.5 to 30.3, p=0.2464) **gap above the limit, but the sample cannot tell it from chance.**
- **blocked_rate**: chat 14.9% (n=47), docs_comment 26.1% (n=23), email 15.7% (n=51), forum 30.0% (n=20). gap 0.8 pp (95% interval -13.5 to 15.1, p=1.0) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
- **intent_correct**: chat 100.0% (n=72), docs_comment 100.0% (n=33), email 100.0% (n=83), forum 100.0% (n=28). gap 0.0 pp (95% interval 0.0 to 0.0, p=None) **within the limit, and the interval is inside it.**
- **retrieval_hit_top5**: chat 89.1% (n=55), docs_comment 94.7% (n=19), email 88.9% (n=54), forum 94.1% (n=17). gap 0.2 pp (95% interval -11.5 to 11.9, p=1.0) **gap within the limit, but the interval reaches past it: sample too small to rule out a larger gap.**
