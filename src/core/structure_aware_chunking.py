"""Structure-aware, small-to-big chunk construction for policy documents.

The retrieval representation stays deliberately small, while every child chunk
keeps a bounded parent context in metadata.  Retrieval can therefore match a
specific clause and generation can consume the complete governing section
without guessing which neighbouring character chunk belongs to it.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from langchain_core.documents import Document


CHUNK_SCHEMA_VERSION = "structure_aware_parent_child_v1"

_LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*\u2022]\s+|(?:[a-zA-Z]|\d{1,3})[.)]\s+)",
    re.UNICODE,
)
_CLAUSE_HEADING_RE = re.compile(
    r"^\s*(?:[A-Z]{1,5})?\d+(?:\.\d+)+(?:\s+|$)",
    re.UNICODE,
)
_NUMBERED_HEADING_RE = re.compile(r"^\s*\d{1,3}\s+[A-ZÄÖÜ]", re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:])\s+")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class _Element:
    text: str
    role: str
    start_index: int


@dataclass(frozen=True)
class _Section:
    heading: str
    elements: tuple[_Element, ...]


def _portable_filename(value: object) -> str:
    normalized = str(value or "unknown").replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] or "unknown"


def _default_token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text or ""))


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _LIST_ITEM_RE.match(stripped):
        return False
    if _CLAUSE_HEADING_RE.match(stripped) or _NUMBERED_HEADING_RE.match(stripped):
        return True
    words = stripped.split()
    if len(words) > 14 or stripped.endswith((".", ",", ";", ":")):
        return False
    letters = [char for char in stripped if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        return True
    return False


def _line_role(line: str) -> str:
    if _LIST_ITEM_RE.match(line):
        return "list_item"
    if _is_heading(line):
        return "heading"
    return "paragraph"


def _extract_elements(text: str) -> list[_Element]:
    """Recover headings, list items, and paragraphs without rewriting source text."""

    elements: list[_Element] = []
    current_lines: list[str] = []
    current_role = "paragraph"
    current_start = 0
    search_from = 0

    def flush() -> None:
        nonlocal current_lines, current_role, current_start
        content = " ".join(part.strip() for part in current_lines if part.strip()).strip()
        if content:
            elements.append(_Element(content, current_role, current_start))
        current_lines = []

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        located = (text or "").find(raw_line, search_from)
        if located < 0:
            located = search_from
        search_from = located + len(raw_line)
        role = _line_role(line)
        starts_element = role in {"heading", "list_item"} or current_role == "heading"
        if current_lines and starts_element:
            flush()
        if not current_lines:
            current_role = role
            current_start = located
        current_lines.append(line)
    flush()
    return elements


def _sections(elements: Sequence[_Element]) -> list[_Section]:
    sections: list[_Section] = []
    heading = ""
    members: list[_Element] = []

    def flush() -> None:
        nonlocal members
        if members:
            sections.append(_Section(heading=heading, elements=tuple(members)))
            members = []

    for element in elements:
        if element.role == "heading":
            flush()
            heading = element.text
        members.append(element)
    flush()
    return sections


def _split_words(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    token_count: Callable[[str], int],
) -> list[str]:
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    start = 0
    while start < len(words):
        end = start
        best_end = min(len(words), start + 1)
        while end < len(words):
            candidate = " ".join(words[start : end + 1])
            if token_count(candidate) > max_tokens and end > start:
                break
            best_end = end + 1
            end += 1
            if token_count(candidate) >= max_tokens:
                break
        pieces.append(" ".join(words[start:best_end]))
        if best_end >= len(words):
            break
        retained = 0
        next_start = best_end
        while next_start > start and retained < overlap_tokens:
            next_start -= 1
            retained = token_count(" ".join(words[next_start:best_end]))
        start = max(start + 1, next_start)
    return pieces


def _split_oversized_element(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
    token_count: Callable[[str], int],
) -> list[str]:
    if token_count(text) <= max_tokens:
        return [text]
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        return _split_words(
            text,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            token_count=token_count,
        )

    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if token_count(sentence) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
            pieces.extend(
                _split_words(
                    sentence,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    token_count=token_count,
                )
            )
            continue
        candidate = " ".join((*current, sentence))
        if current and token_count(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _bounded_parent_groups(
    heading: str,
    elements: Sequence[_Element],
    *,
    max_tokens: int,
    token_count: Callable[[str], int],
) -> list[list[_Element]]:
    groups: list[list[_Element]] = []
    current: list[_Element] = []
    for element in elements:
        if element.role == "heading" and element.text == heading:
            continue
        candidate_parts = ([heading] if heading else []) + [item.text for item in (*current, element)]
        candidate = "\n".join(part for part in candidate_parts if part)
        if current and token_count(candidate) > max_tokens:
            groups.append(current)
            current = [element]
        else:
            current.append(element)
    if current:
        groups.append(current)
    if not groups and heading:
        heading_element = next((item for item in elements if item.role == "heading"), None)
        if heading_element is not None:
            groups.append([heading_element])
    return groups


def _merge_small_peer_elements(
    elements: Sequence[_Element],
    *,
    max_tokens: int,
    token_count: Callable[[str], int],
) -> list[_Element]:
    """Merge undersized peers inside one section, never across a heading boundary."""

    merged: list[_Element] = []
    current: list[_Element] = []
    for element in elements:
        candidate = "\n".join(item.text for item in (*current, element))
        if current and token_count(candidate) > max_tokens:
            roles = {item.role for item in current}
            merged.append(
                _Element(
                    text="\n".join(item.text for item in current),
                    role=next(iter(roles)) if len(roles) == 1 else "section_fragment",
                    start_index=current[0].start_index,
                )
            )
            current = [element]
        else:
            current.append(element)
    if current:
        roles = {item.role for item in current}
        merged.append(
            _Element(
                text="\n".join(item.text for item in current),
                role=next(iter(roles)) if len(roles) == 1 else "section_fragment",
                start_index=current[0].start_index,
            )
        )
    return merged


def _stable_id(*values: object) -> str:
    payload = "\x1f".join(str(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _contextualized_child_text(filename: str, heading: str, text: str) -> str:
    prefix = [f"Document: {filename}"]
    if heading and heading.casefold() not in text.casefold():
        prefix.append(f"Section: {heading}")
    prefix.append(text)
    return "\n".join(prefix)


def _table_chunks(
    document: Document,
    *,
    child_max_tokens: int,
    parent_max_tokens: int,
    token_count: Callable[[str], int],
) -> list[Document]:
    lines = [line.strip() for line in (document.page_content or "").splitlines() if line.strip()]
    if not lines:
        return []
    filename = _portable_filename(document.metadata.get("source"))
    header_lines = lines[:2] if len(lines) >= 2 and set(lines[1]) <= {"|", ":", "-", " "} else lines[:1]
    rows = lines[len(header_lines) :] or header_lines
    header = "\n".join(header_lines)
    parent_groups: list[list[str]] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join((header, *current, row))
        if current and token_count(candidate) > parent_max_tokens:
            parent_groups.append(current)
            current = [row]
        else:
            current.append(row)
    if current:
        parent_groups.append(current)

    chunks: list[Document] = []
    for group_index, group in enumerate(parent_groups):
        parent_content = "\n".join((header, *group))
        parent_id = _stable_id(filename, document.metadata.get("page"), "table", group_index, parent_content)
        child_groups: list[list[str]] = []
        current_rows: list[str] = []
        for row in group:
            candidate = "\n".join((header, *current_rows, row))
            if current_rows and token_count(candidate) > child_max_tokens:
                child_groups.append(current_rows)
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_rows:
            child_groups.append(current_rows)

        row_offset = 0
        for child_index, child_rows in enumerate(child_groups):
            row_text = "\n".join((header, *child_rows))
            chunk_id = _stable_id(parent_id, child_index, row_text)
            metadata = {
                **document.metadata,
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "parent_content": parent_content,
                "parent_start_index": group_index,
                "section_path": f"table {document.metadata.get('table_index', 1)}",
                "structural_role": "table_rows",
                "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                "retrieval_text_contextualized": True,
                "start_index": group_index * 1000 + row_offset,
            }
            chunks.append(Document(page_content=f"Document: {filename}\n{row_text}", metadata=metadata))
            row_offset += len(child_rows)
    return chunks


def build_structure_aware_chunks(
    documents: Iterable[Document],
    *,
    child_max_tokens: int,
    parent_max_tokens: int,
    overlap_tokens: int = 32,
    token_count: Callable[[str], int] | None = None,
) -> list[Document]:
    """Build retrieval children with bounded, structure-preserving parent context."""

    if child_max_tokens <= 0:
        raise ValueError("child_max_tokens must be positive")
    if parent_max_tokens < child_max_tokens:
        raise ValueError("parent_max_tokens must be at least child_max_tokens")
    counter = token_count or _default_token_count
    chunks: list[Document] = []

    for document in documents:
        if not (document.page_content or "").strip():
            continue
        source_type = str(document.metadata.get("source_type") or "")
        if source_type in {"table", "pdf_table"}:
            chunks.extend(
                _table_chunks(
                    document,
                    child_max_tokens=child_max_tokens,
                    parent_max_tokens=parent_max_tokens,
                    token_count=counter,
                )
            )
            continue

        filename = _portable_filename(document.metadata.get("source"))
        for section_index, section in enumerate(_sections(_extract_elements(document.page_content))):
            expanded_elements: list[_Element] = []
            for element in section.elements:
                if element.role == "heading":
                    continue
                pieces = _split_oversized_element(
                    element.text,
                    max_tokens=child_max_tokens,
                    overlap_tokens=overlap_tokens,
                    token_count=counter,
                )
                expanded_elements.extend(
                    _Element(piece, element.role, element.start_index + piece_index)
                    for piece_index, piece in enumerate(pieces)
                )
            if not expanded_elements and section.heading:
                heading_element = section.elements[0]
                expanded_elements = [heading_element]

            expanded_elements = _merge_small_peer_elements(
                expanded_elements,
                max_tokens=child_max_tokens,
                token_count=counter,
            )

            groups = _bounded_parent_groups(
                section.heading,
                expanded_elements,
                max_tokens=parent_max_tokens,
                token_count=counter,
            )
            for parent_index, group in enumerate(groups):
                parent_parts = ([section.heading] if section.heading else []) + [item.text for item in group]
                parent_content = "\n".join(
                    dict.fromkeys(part for part in parent_parts if part)
                )
                parent_id = _stable_id(
                    filename,
                    document.metadata.get("page"),
                    section_index,
                    parent_index,
                    parent_content,
                )
                for element_index, element in enumerate(group):
                    chunk_id = _stable_id(parent_id, element_index, element.text)
                    metadata = {
                        **document.metadata,
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "parent_content": parent_content,
                        "parent_start_index": min(item.start_index for item in group),
                        "section_path": section.heading,
                        "structural_role": element.role,
                        "chunk_schema_version": CHUNK_SCHEMA_VERSION,
                        "retrieval_text_contextualized": True,
                        "start_index": element.start_index,
                    }
                    chunks.append(
                        Document(
                            page_content=_contextualized_child_text(
                                filename,
                                section.heading,
                                element.text,
                            ),
                            metadata=metadata,
                        )
                    )

    for index, chunk in enumerate(chunks):
        if index > 0 and chunk.metadata.get("source") == chunks[index - 1].metadata.get("source"):
            chunk.metadata["previous_chunk_id"] = chunks[index - 1].metadata["chunk_id"]
        if index + 1 < len(chunks) and chunk.metadata.get("source") == chunks[index + 1].metadata.get("source"):
            chunk.metadata["next_chunk_id"] = chunks[index + 1].metadata["chunk_id"]
        chunk.metadata["token_count"] = int(counter(chunk.page_content))
        chunk.metadata["parent_token_count"] = int(counter(str(chunk.metadata.get("parent_content") or "")))
    return chunks


def estimated_tokens(text: str) -> int:
    """Public fallback counter used in diagnostics and tests."""

    return _default_token_count(text)
