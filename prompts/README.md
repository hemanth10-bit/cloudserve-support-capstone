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
| PR-01 grounded answer | 0.2 | run on 60 development tickets (D-10); v0.1 kept |
| PR-02 escalation hand-over | 0.2 | run on 20 escalations, all notes from the model (D-10); v0.1 kept |
| PR-03 claim-support check | 0.2 | run on 40 drafts, 121 claims, fails closed (D-11); v0.1 kept |
| PR-04 answer-quality judge | 0.2 | run on 68 replies; unusable (echoed the forbidden list), discarded (D-14) |

Examples inside the prompt text use `{placeholders}` that the code fills in.
