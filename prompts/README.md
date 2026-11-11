# Prompt library

One file per prompt, named `PR-NN_<name>.v<version>.txt`. The header at the top of each file records the purpose, the
requirement it serves (see the PRD), the required output format, known weaknesses and the change history. The same
entries are in the Stage 3 workbook.

Rules:
- A prompt's version changes whenever its text changes; the old file is kept, never overwritten.
- The version used is written to the decision log (`prompt_version`) for every generated or validated reply.
- Customer text is always inside delimiter tags and every prompt says it is data, not instructions.

| Prompt | Version | Status |
|---|---|---|
| PR-01 grounded answer | 0.1 | draft, not yet run (B-08) |
| PR-02 escalation hand-over | 0.1 | draft, not yet run (B-08) |
| PR-03 claim-support check | 0.1 | draft, not yet run (B-09) |
| PR-04 answer-quality judge | 0.1 | draft, not yet run (evaluation) |

Examples inside the prompt text use `{placeholders}` that the code fills in.
