"""Reply quality against the senior-agent reference answers, independent of the router and the validator.

    python -m evaluation.quality_eval --results DIR --tickets FILE --references FILE [--judge] [--review-sheet N]

For every drafted reply that has a reference answer (sent or withheld by the validator) it computes:
  doc_match       the reply cites one of the reference's expected documents
  must_mention    share of the reference's must-mention points present in the reply (text check)
  violations      must-not-claim statements the reply appears to make (pattern check)
  similarity      cosine similarity of reply and reference (MiniLM)
  judge           optional PR-04 score and violations from the model (same small model: indicative only)
and groups them by what happened (sent / withheld) and by what the labels say (answerable / not).

--review-sheet N also writes a blinded sheet of N sent replies for the two human assessors the Evaluation Framework
requires for the hallucination rate. That measure cannot be produced by this script; it only prepares the material.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


from src.embeddings import embed
from src.generation import load_prompt, parse_json
from src.llm import LLMClient
from src.validator import DATE_PROMISE

REFUND_DONE = re.compile(r"\b(?:refund(?:ed)?|credit(?:ed)?)\b.{0,40}\b(?:issued|processed|applied|been)\b", re.I)
FIXED = re.compile(
    r"\b(?:we(?:'ve| have)|has been|have been)\s+(?:now\s+)?(?:fixed|resolved|patched|corrected)\b", re.I
)


def violations(reply: str) -> list[str]:
    out = []
    if REFUND_DONE.search(reply) or re.search(r"\byou(?:'ll| will) be (?:refunded|credited)\b", reply, re.I):
        out.append("a refund has been issued")
    if FIXED.search(reply):
        out.append("the issue has been fixed on our side")
    if DATE_PROMISE.search(reply):
        out.append("a specific delivery date for a fix")
    return out


def must_mention_share(reply: str, points: list[str]) -> float | None:
    if not points:
        return None
    words = set(re.findall(r"[a-z0-9]{4,}", reply.lower()))
    hit = 0
    for p in points:
        pw = set(re.findall(r"[a-z0-9]{4,}", p.lower()))
        hit += bool(pw) and len(pw & words) / len(pw) >= 0.6
    return hit / len(points)


def summarise(rows: list[dict]) -> dict:
    def avg(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(v), 3) if v else None

    return {
        "n": len(rows),
        "cites_expected_doc_pct": round(100 * sum(r["doc_match"] for r in rows) / len(rows), 1) if rows else None,
        "must_mention_share": avg("must_mention"),
        "reply_makes_a_forbidden_claim_pct": (
            round(100 * sum(bool(r["violations"]) for r in rows) / len(rows), 1) if rows else None
        ),
        "similarity_to_reference": avg("similarity"),
        "judge_score_mean_0_to_5": avg("judge_score"),
        "judge_flags_a_violation_pct": round(
            100
            * sum(bool(r.get("judge_violated")) for r in rows if r.get("judge_score") is not None)
            / max(1, sum(r.get("judge_score") is not None for r in rows)),
            1,
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tickets", required=True)
    ap.add_argument("--references", required=True)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--review-sheet", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.results)

    results = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    tickets = {t["ticket_id"]: t for t in json.loads(Path(a.tickets).read_text())}
    refs = {r["ticket_id"]: r for r in json.loads(Path(a.references).read_text())}
    drafted = [r for r in results if r.get("response") and r["response"].get("reply")]

    judge_prompt = load_prompt("PR-04", "0.2") if a.judge else None
    llm = LLMClient() if a.judge else None
    rows = []
    sims = {}
    todo = [r for r in drafted if r["ticket_id"] in refs]
    if todo:
        vec = embed([r["response"]["reply"] for r in todo] + [refs[r["ticket_id"]]["reference_response"] for r in todo])
        n = len(todo)
        sims = {r["ticket_id"]: float(vec[i] @ vec[n + i]) for i, r in enumerate(todo)}
    for r in todo:
        ref, t, reply = refs[r["ticket_id"]], tickets[r["ticket_id"]], r["response"]["reply"]
        lab = t.get("labels") or {}
        row = {
            "ticket_id": r["ticket_id"],
            "sent": r["outcome"] == "answered",
            "label_answerable": bool(lab.get("answerable_from_docs")),
            "doc_match": bool(set(r["response"].get("citations") or []) & set(ref["expected_doc_ids"])),
            "must_mention": must_mention_share(reply, ref.get("must_mention") or []),
            "violations": violations(reply),
            "similarity": round(sims[r["ticket_id"]], 3),
        }
        if a.judge:
            user = (
                judge_prompt.user.replace("{ticket}", f"{t.get('subject', '')}\n{t.get('body', '')}"[:3000])
                .replace("{reference_response}", ref["reference_response"])
                .replace("{must_mention}", json.dumps(ref.get("must_mention") or []))
                .replace("{must_not_claim}", json.dumps(ref.get("must_not_claim") or []))
                .replace("{reply}", reply.replace("</reply>", ""))
            )
            res = llm.complete(judge_prompt.system, user, max_tokens=500)
            obj = None if res.error else parse_json(res.text)
            if obj and isinstance(obj.get("score"), (int, float)):
                row["judge_score"], row["judge_violated"] = float(obj["score"]), obj.get("violated") or []
        rows.append(row)

    groups = defaultdict(list)
    for row in rows:
        groups[
            ("sent" if row["sent"] else "withheld")
            + ("_labelled_answerable" if row["label_answerable"] else "_labelled_not")
        ].append(row)
    report = {
        "replies_with_a_reference": len(rows),
        "all": summarise(rows),
        "by_group": {k: summarise(v) for k, v in sorted(groups.items())},
        "note": "Same-model judge is indicative only; the hallucination rate needs two human assessors.",
    }
    (out / "quality.json").write_text(json.dumps({"report": report, "rows": rows}, indent=1))
    print(json.dumps(report, indent=1))

    if a.review_sheet:
        sent = [r for r in drafted if r["outcome"] == "answered"]
        rng = random.Random(7)
        pick = rng.sample(sent, min(a.review_sheet, len(sent)))
        docs = {
            d["doc_id"]: d["content"]
            for d in json.loads((Path(__file__).resolve().parent.parent / "data" / "documentation.json").read_text())
        }
        with (out / "human_review_sheet.csv").open("w", newline="", encoding="utf-8") as fh, (
            out / "human_review_key.json"
        ).open("w") as key:
            w = csv.writer(fh)
            w.writerow(
                [
                    "review_id",
                    "ticket",
                    "reply_to_check",
                    "cited_documents",
                    "source_article_text",
                    "unsupported_claims_found(count)",
                    "claim_text(s)",
                    "citations_correct(yes/no)",
                    "assessor_initials",
                ]
            )
            mapping = {}
            for i, r in enumerate(pick, 1):
                t = tickets[r["ticket_id"]]
                w.writerow(
                    [
                        f"R{i:02d}",
                        f"{t.get('subject', '')}\n{t.get('body', '')}"[:1500],
                        r["response"]["reply"],
                        ", ".join(r["response"].get("citations") or []),
                        "\n\n".join(f"[{d}] {docs.get(d, '')}" for d in (r["response"].get("citations") or [])),
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                mapping[f"R{i:02d}"] = r["ticket_id"]
            json.dump(mapping, key, indent=1)
        print(f"wrote blinded review sheet for {len(pick)} sent replies")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
