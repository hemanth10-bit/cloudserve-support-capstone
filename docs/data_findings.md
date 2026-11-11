# Data findings that affect evaluation (found 2026-11-10 during B-04)

Measured on the supplied tickets. Reproduce with the snippet in `evaluation/data_checks.py`.

1. **Few distinct ticket texts.** The 500 development tickets contain only 215 distinct bodies (78 texts are used more than once).
2. **Validation overlaps development.** 62 of the 80 validation bodies also appear verbatim in the development set. A model that memorises development tickets will score well on validation without generalising, so validation figures are optimistic. The hidden set is drawn from the same population and is likely to overlap too.
   - Honest generalisation estimate: cross-validate on development with groups defined by ticket text, so no text is in both train and test. Report this alongside validation.
3. **Labels are not all functions of the text.** For identical text: intent and must_not_auto_respond never conflict (ceiling 100%); `answerable_from_docs` conflicts for 41 texts (best possible accuracy from text alone: 86.4%); `expected_route` 88.8%; `expected_doc_ids` 78.0%; **urgency only 73.0%**.
   - Consequence: no system can reach 100% on urgency, answerability or route from text alone. Targets for these must be judged against the ceiling; urgency may depend on fields other than text (tier, channel), to be tested in B-06.
4. **Retrieval score does not predict answerability** (see decisions.md D-05).
