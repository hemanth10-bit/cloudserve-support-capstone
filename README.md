# CloudServe support automation

Automatic first-line handling of support tickets for CloudServe Solutions. For each ticket it classifies the request,
searches the documentation, decides whether to answer or escalate to a person, drafts a cited reply, checks the reply,
and records every decision. Anything risky or uncertain goes to a person with a short hand-over note.

**Status:** build complete for the acceptance criteria below. A one-assessor human review found no unsupported claims in
60 replies; a second assessor and a held-out check are still advisable before automatic sending (see *Known limits*).

```
ticket -> ingest -> classify (intent, urgency) -> retrieve passages -> route (rules + answerability)
       -> draft reply with citations -> validate (7 checks) -> answer, or escalate with a hand-over note
every step writes to the decision log; a kill switch stops all automatic answering at once
```

## Setup (tested on macOS, Python 3.12)

```bash
python3.12 -m venv .venv            # Python 3.10+ is the minimum; 3.12 is what this was tested on
source .venv/bin/activate
pip install -r requirements.txt     # pinned lock; requirements.in lists the direct dependencies
cp .env.example .env                # then put your own OpenRouter key in .env (never commit it)
```

Needs internet on first use (downloads the all-MiniLM-L6-v2 embedding model, about 90 MB) and an OpenRouter key
(free tier is enough) for the model calls. The tests need the embedding model but no key: they use a fake language model.
The supplied course `requirements.txt` does not install on Python 3.12 (see `docs/decisions.md`, D-01).

## Run the tests (one command)

```bash
python -m pytest tests/ -v
```

The suite uses a fake language model, so it makes no paid or networked model calls and takes about 20 seconds. It covers each requirement (FR-01 to FR-12): ingest,
classification, retrieval, routing and always-escalate rules, generation, the validator and its red-team set, the decision
log, failure handling (provider outage, timeouts, rate limits, empty retrieval, malformed input), the kill switch and the
fairness audit method.

## Run the whole system on a ticket file (one unattended command)

```bash
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/val_run
```

Any JSON file of tickets with the dataset schema works (a list, `{"tickets": [...]}`, or JSON Lines). Labels are optional;
without them the accuracy measures are skipped and the rest is reported. Writes to the output folder:
`results.jsonl` (one result per ticket), `decisions.db` (the decision log), `metrics.json` and `metrics.md` (the report).
Exit code 0 = finished and the log reconciles; 1 = finished but the log has a gap; 2 = the input could not be read.
Every ticket gets an outcome: answered with a cited reply, or escalated with a reason. About 6 to 10 seconds per ticket on the
free model, much faster on a rerun because responses are cached in `storage/`.

## Kill switch

To stop all automatic answering immediately (no deployment, no restart):

```bash
python -m src.killswitch on "reason"     # every ticket is escalated from the next one; no model is called
python -m src.killswitch status
python -m src.killswitch off
```

Details, who may use it and how it is tested: `docs/governance.md`.

## Other commands

```bash
python -m src.ingest data/development_tickets.json        # one normalised ticket per channel
python -m evaluation.retrieval_eval --input data/development_tickets.json --output evaluation/results/
python -m evaluation.data_checks                          # label consistency and train/validation overlap
python -m src.classify                                    # retrain; models/ already holds the trained files
python -m src.router                                      # print the threshold sweep (writes models/router.json)
python -m evaluation.fairness --results DIR --tickets FILE   # fairness audit of a harness run
python -m evaluation.quality_eval --results DIR --tickets FILE --references data/ground_truth_responses.json
python -m evaluation.build_review_page --sheet SHEET.csv --out review_tool.html   # blinded human review page
```

## Where things are

| Path | What |
|---|---|
| `src/` | the system: `ingest`, `classify`, `retrieval`, `router`, `generation`, `validator`, `decision_log`, `pipeline`, `llm`, `killswitch` |
| `prompts/` | every prompt, one file per version, with purpose, known weaknesses and change history |
| `evaluation/` | the harness and the analyses that feed the report |
| `tests/` | the test suite |
| `docs/decisions.md` | every design decision with its evidence (D-01 onward) |
| `docs/governance.md` | risk register, guardrails, incident procedure, kill switch, declaration |
| `docs/gate_run_validation.md` | results of the unattended validation run |
| `docs/fairness_audit.md` | fairness audit and its limits |
| `docs/data_findings.md` | what is odd about the data (label noise, train/validation overlap) |
| `docs/ATTRIBUTION.md` | AI-use declaration |

## Known limits (read before relying on the numbers)

- **Hallucination rate: 0 of 60 reviewed replies (95% upper bound 6%), one assessor, development tickets, replies that
  had already passed the validator** (D-19). The framework asks for two assessors; a second is still wanted. The review also
  found replies missing their citation in the text; fixed.
- **Latency target not met:** p95 17.9 s against 3 s with a free 8B model (D-13).
- **The routing threshold (0.64) is a judgement call,** not a fitted optimum: measured pass-through of wrongly routed
  tickets was far higher than assumed (D-13, D-17).
- Validation intent accuracy is optimistic: 62 of 80 validation tickets share text with development. The honest
  figure is the out-of-fold 95.8% (strict lower bound 89.6%).
- No evidence of unfairness between customer groups, and the sample cannot show its absence (`docs/fairness_audit.md`).
