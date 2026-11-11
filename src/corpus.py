"""Load the knowledge base and split each article into section chunks.

Chunking decision (B-03, FR-03): every article has the same structure
(title, applies-to, Symptoms, Common causes, Resolution, Notes) and is short
(about 1,100 characters). One chunk per section, with the title prefixed, because
customers describe symptoms and the symptoms section matches their wording.
Sections are never split further, so a numbered resolution sequence is never cut.
Citations stay at the article level (doc_id), matching labels.expected_doc_ids.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>#<section-slug>", resolves back to the corpus
    doc_id: str
    title: str
    category: str
    section: str  # "symptoms", "common_causes", "resolution", "notes", "overview"
    text: str  # the section text as written
    embed_text: str  # what is embedded: title + section label + text


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_documents(path: str | Path) -> list[dict]:
    docs = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(docs, list):
        raise ValueError(f"{path}: expected a list of articles")
    seen: set[str] = set()
    for d in docs:
        if d["doc_id"] in seen:
            raise ValueError(f"duplicate doc_id {d['doc_id']}")
        seen.add(d["doc_id"])
    return docs


def chunk_document(doc: dict) -> list[Chunk]:
    content = doc.get("content", "") or ""
    title, category, doc_id = doc.get("title", ""), doc.get("category", ""), doc["doc_id"]
    applies = doc.get("applies_to", "")
    matches = list(SECTION_RE.finditer(content))
    chunks: list[Chunk] = []

    head_end = matches[0].start() if matches else len(content)
    head = re.sub(r"^#\s+.*$", "", content[:head_end], flags=re.M).strip()
    if applies and "Applies to" not in head:
        head = f"Applies to: {applies}\n{head}".strip()
    if head:
        chunks.append(
            Chunk(f"{doc_id}#overview", doc_id, title, category, "overview", head, f"{title}. Overview: {head}")
        )

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[m.end() : end].strip()
        if not body:
            continue
        slug = _slug(m.group(1))
        chunks.append(Chunk(f"{doc_id}#{slug}", doc_id, title, category, slug, body, f"{title}. {m.group(1)}: {body}"))
    if not chunks and content.strip():  # unstructured article: keep it whole rather than lose it
        chunks.append(
            Chunk(f"{doc_id}#full", doc_id, title, category, "full", content.strip(), f"{title}. {content.strip()}")
        )
    return chunks


def build_chunks(docs: list[dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out
