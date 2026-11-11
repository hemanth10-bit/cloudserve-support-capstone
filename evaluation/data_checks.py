"""Reproduces the label-consistency and overlap numbers in docs/data_findings.md."""

import collections as C
import json
import re
import sys

norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()  # noqa: E731


def main(dev="data/development_tickets.json", val="data/validation_tickets.json"):
    d, v = json.load(open(dev)), json.load(open(val))
    print(f"dev: {len({norm(t['body']) for t in d})} distinct bodies / {len(d)} tickets")
    dev_texts = {norm(t["body"]) for t in d}
    print(f"validation bodies also in dev: {sum(norm(t['body']) in dev_texts for t in v)} of {len(v)}")
    keys = {
        "intent": lambda t: t["labels"]["intent"],
        "urgency": lambda t: t["labels"]["urgency"],
        "answerable_from_docs": lambda t: t["labels"]["answerable_from_docs"],
        "expected_route": lambda t: t["labels"]["expected_route"],
        "must_not_auto_respond": lambda t: t["labels"]["must_not_auto_respond"],
        "expected_doc_ids": lambda t: tuple(t["labels"]["expected_doc_ids"]),
    }
    for name, key in keys.items():
        g = C.defaultdict(list)
        for t in d:
            g[norm(t["body"])].append(key(t))
        conflicts = sum(1 for x in g.values() if len(set(x)) > 1)
        ceiling = sum(max(C.Counter(x).values()) for x in g.values()) / len(d)
        print(f"{name:24s} texts with conflicting labels: {conflicts:3d}  ceiling accuracy from text: {ceiling:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
