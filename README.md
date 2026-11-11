# CloudServe support automation

Intelligent ticket handling for CloudServe Solutions: ingest, classify, retrieve, route, generate and validate,
with a decision log and an evaluation harness. Status: **work in progress** (sprint plan steps B-01 to B-07 and B-10 done: environment, ingest, retrieval, evaluation harness, classifier, router, decision log; generation and guardrails still to build).

## Setup (tested on macOS, Python 3.12)

```bash
python3.12 -m venv .venv            # Python 3.10+ is the minimum; 3.12 is what this was tested on
source .venv/bin/activate
pip install -r requirements.txt     # pinned lock; requirements.in lists the direct dependencies
cp .env.example .env                # then put your own OpenRouter key in .env (never commit it)
```

The supplied course `requirements.txt` does not install on Python 3.12 (see `docs/decisions.md`, D-01).

## Run the tests

```bash
python -m pytest tests/ -v
```

## Ingest demo (one normalised ticket per channel)

```bash
python -m src.ingest data/development_tickets.json
```

## Retrieval evaluation (development set only; also picks the relevance threshold)

```bash
python -m evaluation.retrieval_eval --input data/development_tickets.json --output evaluation/results/
python -m evaluation.data_checks     # label-consistency and train/validation overlap numbers
```

The first run downloads the all-MiniLM-L6-v2 embedding model (needs internet once) and builds the Chroma index in `storage/`.

## Train the classifier (optional: the trained model is committed in `models/`)

```bash
python -m src.classify    # writes models/classifier.joblib and models/classifier_report.json
```

## Fit the router (optional: fitted values are committed in `models/router.json`)

```bash
python -m src.router     # prints the threshold sweep and writes models/router.json
```

## Run the evaluation harness (one unattended command; any ticket file with the dataset schema)

```bash
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/val_run
```

Writes `results.jsonl`, `decisions.db` (the decision log), `metrics.json` and `metrics.md`. Reports carry a PROVISIONAL banner until routing, generation and
guardrails are in place.

Further commands (API, evaluation harness) are added here as each component lands.

## Kill switch
To stop all automatic answering immediately (no deployment, no restart):

    python -m src.killswitch on "reason"     # every ticket is escalated from the next one; no model is called
    python -m src.killswitch status
    python -m src.killswitch off

Details, who may use it and how it is tested: `docs/governance.md`.
