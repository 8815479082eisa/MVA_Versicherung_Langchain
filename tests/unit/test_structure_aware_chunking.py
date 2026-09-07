from langchain_core.documents import Document

from src.core.structure_aware_chunking import (
    CHUNK_SCHEMA_VERSION,
    build_structure_aware_chunks,
)


def test_list_children_keep_complete_governing_parent_section() -> None:
    page = Document(
        page_content=(
            "G10.2 You will not have to bear a deductible:\n"
            "g) if the damaged front windscreen is repaired and not replaced.\n"
            "h) if glass repair is organised by Helvetia and a partner company.\n"
            "G10.3 One deductible applies to towing vehicles."
        ),
        metadata={"source": "motor-policy.pdf", "source_type": "pdf", "page": 8},
    )

    chunks = build_structure_aware_chunks(
        [page],
        child_max_tokens=80,
        parent_max_tokens=200,
        overlap_tokens=8,
    )

    windscreen = next(chunk for chunk in chunks if "windscreen" in chunk.page_content)
    assert windscreen.metadata["chunk_schema_version"] == CHUNK_SCHEMA_VERSION
    assert windscreen.metadata["structural_role"] == "list_item"
    assert "G10.2 You will not have to bear a deductible" in windscreen.page_content
    assert "g) if the damaged front windscreen" in windscreen.metadata["parent_content"]
    assert "h) if glass repair is organised" in windscreen.metadata["parent_content"]
    assert "G10.3" not in windscreen.metadata["parent_content"]


def test_table_rows_repeat_header_and_retain_parent_context() -> None:
    table = Document(
        page_content=(
            "| Coverage | Limit |\n"
            "|:--|--:|\n"
            "| Windscreen | CHF 500 |\n"
            "| Theft | CHF 1000 |"
        ),
        metadata={
            "source": "benefits.pdf",
            "source_type": "pdf_table",
            "page": 2,
            "table_index": 1,
        },
    )

    chunks = build_structure_aware_chunks(
        [table],
        child_max_tokens=60,
        parent_max_tokens=120,
    )

    windscreen = next(chunk for chunk in chunks if "Windscreen" in chunk.page_content)
    assert "| Coverage | Limit |" in windscreen.page_content
    assert "| Windscreen | CHF 500 |" in windscreen.page_content
    assert "| Theft | CHF 1000 |" in windscreen.metadata["parent_content"]
    assert windscreen.metadata["structural_role"] == "table_rows"
