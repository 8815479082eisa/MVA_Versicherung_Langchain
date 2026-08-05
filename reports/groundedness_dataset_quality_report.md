# Extended Groundedness Dataset - Technical Quality Report

## Result

- Candidates checked: **498**
- Cases with at least one non-destructive quality flag: **15**
- Duplicate case IDs: **0**
- Exact duplicate triplets: **7**
- Identical answers across technical PASS/FAIL expectations: **0**
- Near-duplicate pairs at cosine similarity >= 0.94: **319**
- Invalid source references: **0**
- Contexts longer than 6000 characters: **0**
- Cases excluded automatically: **0**

No candidate was silently deleted. Flagged cases remain in the review dataset
with `annotation_status=REVIEW_REQUIRED`.

## Quality flags

| Flag | Count |
|---|---:|
| `exact_duplicate_triplet` | 14 |
| `possibly_trivial_fail_low_lexical_overlap` | 1 |

## Methods

- Exact duplicates: normalized `(question, context, candidate_answer)` equality.
- Near duplicates: word unigram/bigram TF-IDF cosine similarity, threshold 0.94.
- English check: conservative Latin-character heuristic; uncertain cases are flagged, not removed.
- Source validation: every primary and mutation-source path must exist under the repository.
- Mutation trace: every technical FAIL must include a documented mutation detail.
- Trivial FAIL screen: lexical answer/context overlap below 0.08 is flagged for human review.
