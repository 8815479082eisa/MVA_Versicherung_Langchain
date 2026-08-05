# Helvetia RAG reindex summary

Date: 2026-08-01

## Result

- Status: successful
- Indexed corpus: 11 unique Helvetia PDF files
- Chroma collection: `insurance_rag_collection`
- Final chunk count: 1,585
- Chunk size: 1,000 characters
- Chunk overlap: 200 characters
- Embedding model: `BAAI/bge-m3` on CPU
- Reranker model: `BAAI/bge-reranker-base`
- PDF hashes and model configuration cache: updated
- `pdfs_have_changed`: false after completion
- `embedding_model_has_changed`: false after completion

## Indexed sources

| Source | Chunks |
| --- | ---: |
| assistance-brochure.pdf | 13 |
| assistance-sti.pdf | 213 |
| brochure-household-contents-and-private-liability.pdf | 11 |
| brochure-services.pdf | 6 |
| buildings-insurance-sti.pdf | 345 |
| household-contents-private-liability-sti.pdf | 559 |
| legal-protection-sti.pdf | 145 |
| motor-vehicle-insurance-product-sheet.pdf | 13 |
| motor-vehicle-insurance-sti.pdf | 169 |
| mutual-provisions-pkv.pdf | 87 |
| rental-guarantee-insurance-sti.pdf | 24 |

## Corpus cleanup

The following files were moved outside the indexed PDF directory to
`data/raw/excluded_pdfs`:

- `140_1261_e.pdf` (Baloise)
- `240_1184_e.pdf` (Baloise)
- `240_1217_e.pdf` (Baloise)
- `brochure-services (1).pdf` (byte-identical duplicate)

All 11 indexed PDFs opened successfully, contained extractable text on every
page, contained Helvetia references, and had no Baloise text matches.

## Retrieval validation

Hybrid retrieval was checked for motor vehicle glass damage, household
contents cover, and cancellation cost cover. The relevant Helvetia product and
STI documents appeared in the returned top five results. A reranked motor
vehicle query prioritized `motor-vehicle-insurance-sti.pdf` and
`motor-vehicle-insurance-product-sheet.pdf`.

## Operational note

Both Chroma batches completed successfully: 1,000 chunks followed by 585
chunks. The original terminal wait expired after 30 minutes while the worker
continued. After the final batch had been persisted, the disconnected stdout
handle caused a reporting error. The persisted collection was then verified,
PDF hashes and model status were updated, and retrieval-only initialization was
validated without another reindex.

## Security hardening

The runtime configuration snapshot now redacts sensitive dotenv conflict
values before they are persisted. The generated model configuration cache was
rewritten with redacted values. The targeted metadata test suite passed with
8 tests.
