"""Build a deterministic 200-question, PDF-grounded RAG-only benchmark.

Each case is generated from an extracted sentence-sized evidence unit.  The
source page and exact reference text are retained so the benchmark can be
audited without relying on a second language model to invent ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "data" / "raw" / "pdfs"
DEFAULT_OUTPUT = ROOT / "data" / "benchmarks" / "pdf_rag" / "pdf_rag_200.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "benchmarks" / "pdf_rag" / "pdf_rag_200_manifest.json"

# The quotas deliberately cover every PDF in the active corpus while keeping
# enough cases in the larger STI documents to exercise long-document retrieval.
QUOTAS = {
    "assistance-brochure.pdf": 10,
    "assistance-sti.pdf": 20,
    "brochure-household-contents-and-private-liability.pdf": 10,
    "brochure-services.pdf": 8,
    "buildings-insurance-sti.pdf": 25,
    "household-contents-private-liability-sti.pdf": 35,
    "legal-protection-sti.pdf": 15,
    "motor-vehicle-insurance-product-sheet.pdf": 10,
    "motor-vehicle-insurance-sti.pdf": 40,
    "mutual-provisions-pkv.pdf": 20,
    "rental-guarantee-insurance-sti.pdf": 7,
}

_BOILERPLATE = re.compile(
    r"(?:^|\s)(?:\d{1,2}/\d{1,2}\s*\|\s*)?Helvetia(?: Swiss Insurance Company Ltd)?(?:\s+STI)?[^.]{0,160}(?:edition|product information sheet)[^.]*",
    re.IGNORECASE,
)
_FOOTER = re.compile(r"\b\d{1,2}/\d{1,2}\s*\|[^.]{0,180}", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"^\s*\d{1,2}/\d{1,2}\s*$")
_SPACES = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

_STOPWORDS = {
    "a", "about", "after", "all", "also", "an", "and", "are", "as", "at", "be",
    "by", "can", "for", "from", "if", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "under", "we", "what", "when", "which", "with",
}


def _clean_text(value: str) -> str:
    value = value.replace("\u00ad", "")
    value = re.sub(r"([A-Za-z])[-]\s+([A-Za-z])", r"\1\2", value)
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    value = value.replace("\u2022", " • ")
    value = _FOOTER.sub(" ", value)
    value = _BOILERPLATE.sub(" ", value)
    return _SPACES.sub(" ", value).strip(" -•")


def _sentence_units(text: str) -> Iterable[str]:
    # Preserve bullets as separate evidence units, while retaining normal
    # sentence boundaries for prose and numbered clauses.
    text = _clean_text(text)
    if not text:
        return
    pieces = re.split(r"\s+•\s+|(?<=[.!?])\s+", text)
    for piece in pieces:
        candidate = _SPACES.sub(" ", piece).strip(" -•")
        words = _WORD.findall(candidate)
        if len(words) < 12 or len(words) > 90:
            continue
        if len(set(words)) < 6 or sum(char.isalpha() for char in candidate) < 45:
            continue
        if _PAGE_NUMBER.match(candidate) or candidate.lower().startswith("helvetia.ch/"):
            continue
        # Avoid pure page headers/footers and contact blocks.
        if sum(char.isdigit() for char in candidate) > max(12, len(candidate) // 3):
            continue
        yield candidate


def _candidates(pdf_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        for sentence_index, sentence in enumerate(_sentence_units(page.extract_text() or "")):
            key = re.sub(r"[^a-z0-9]+", " ", sentence.casefold()).strip()
            if key in seen:
                continue
            seen.add(key)
            # Prefer units that contain policy language or a section marker;
            # retaining the rest still gives broad corpus coverage.
            score = 0
            lower = sentence.casefold()
            score += 3 * sum(term in lower for term in ("insurance", "cover", "insured", "benefit", "exclusion"))
            score += 2 * bool(re.search(r"\b[A-Z]{1,5}\d+(?:\.\d+)+\b", sentence))
            score += min(len(sentence.split()) // 20, 3)
            rows.append({"page": page_number, "sentence_index": sentence_index, "text": sentence, "score": score})
    rows.sort(key=lambda row: (-int(row["score"]), int(row["page"]), int(row["sentence_index"])))
    return rows


def _subject(text: str) -> str:
    words = text.split()
    # Strip generic lead-ins so the question focuses on the actual subject.
    lead = re.compile(
        r"^(?:the|this) (?:insurance|policy|coverage) (?:covers|does not cover|provides|includes|is|states|offers)\b",
        re.IGNORECASE,
    )
    trimmed = lead.sub("", text).strip()
    words = trimmed.split() or text.split()
    # Keep function words inside the phrase; removing them makes extracted
    # questions read unnaturally and can also change the meaning of a clause.
    selected = words[:10]
    value = " ".join(selected).strip(".,:;()")
    return value or "the stated insurance conditions"


def _question(display_name: str, evidence: str, variant: int) -> str:
    subject = _subject(evidence)
    templates = (
        f'According to {display_name}, what does it state about "{subject}"?',
        f'What coverage or rule is described for "{subject}" in {display_name}?',
        f'How does {display_name} describe "{subject}"?',
        f'What does {display_name} say concerning "{subject}"?',
    )
    return templates[variant % len(templates)]


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    used_questions: set[str] = set()
    for filename, quota in QUOTAS.items():
        path = PDF_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        candidates = _candidates(path)
        if len(candidates) < quota:
            raise RuntimeError(f"{filename}: found {len(candidates)} candidates, need {quota}")
        # Select across the document rather than taking only the highest-scored
        # page, which keeps the benchmark representative of the whole PDF.
        stride = max(1, len(candidates) // quota)
        selected: list[dict[str, object]] = []
        used: set[str] = set()
        for index in range(quota):
            candidate = candidates[min(index * stride, len(candidates) - 1)]
            key = str(candidate["text"])
            if key in used:
                candidate = next(row for row in candidates if str(row["text"]) not in used)
            used.add(str(candidate["text"]))
            selected.append(candidate)
        display = filename.removesuffix(".pdf").replace("-", " ")
        for candidate in selected:
            number = len(cases) + 1
            evidence = str(candidate["text"])
            page = int(candidate["page"])
            digest = hashlib.sha256(f"{filename}|{page}|{evidence}".encode("utf-8")).hexdigest()[:16]
            question = _question(display, evidence, number)
            if question in used_questions:
                tail = " ".join(evidence.rstrip(".").split()[-6:])
                question = question.rstrip("?") + f" (specifically, the point about {tail})?"
            used_questions.add(question)
            cases.append(
                {
                    "id": f"pdf-rag-{number:03d}",
                    "question": question,
                    "expected_route": "retrieval_only",
                    "category": "pdf_rag_only",
                    "subcategory": filename.removesuffix(".pdf"),
                    "source_references": [f"data/raw/pdfs/{filename}"],
                    "source_pages": [page],
                    "expected_source": filename,
                    "reference_answer": evidence,
                    "reference_evidence": [
                        {"source": filename, "page": page, "quote": evidence}
                    ],
                    "evidence_id": digest,
                    "generation_method": "deterministic_pdf_sentence_extraction_v1",
                }
            )
    if len(cases) != 200:
        raise AssertionError(f"Expected 200 cases, got {len(cases)}")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    cases = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    counts = {filename: quota for filename, quota in QUOTAS.items()}
    manifest = {
        "dataset": str(args.output),
        "version": "pdf_rag_200_v1",
        "method": "deterministic sentence-sized evidence extraction from active PDF corpus",
        "human_validated": False,
        "total_cases": len(cases),
        "expected_route": "retrieval_only",
        "pdf_case_counts": counts,
        "source_directory": str(PDF_DIR),
        "reference_fields": ["reference_answer", "reference_evidence", "source_references", "source_pages"],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {args.output}")
    print(f"Wrote manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
