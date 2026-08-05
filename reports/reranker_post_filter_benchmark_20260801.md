# Reranker benchmark after deterministic current-policy filtering

## Scope and method

The same 64 static regression queries and the same eight candidate chunks were used for every model. No `/api/ask`, Chroma, BM25, embeddings, reindex, LLM, self-check or CRM call was executed. The metadata gate reads only the fixture copy derived from the project's CRM CSV files.

- Reference date: 2026-08-01
- CPU only; FP16 disabled; threads=8
- max_length=512; batch_size=8
- Warm-up: 1 per loaded model; measured runs: 3 per query and mode
- Each model was loaded once in its own process; downloads completed before load timing.
- Raw scores are not compared; all quality metrics are ranking-based.
- Peak RAM was not sampled because reliable process-tree measurement would add instrumentation.
- Host: Windows-11-10.0.26100-SP0

## Compact comparison (post-filter ranking)

| Model | Params | Top-1 | MRR@5 | nDCG@5 | EN Lara raw→filtered | Lara all languages | Median rerank | Mean rerank | Load | Stable |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7M | 0.969 | 0.982 | 0.986 | 2→1 PASS | PASS | 467.8ms | 464.3ms | 0.10s | True |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 117.6M | 1.000 | 1.000 | 1.000 | 2→1 PASS | PASS | 843.5ms | 865.0ms | 1.34s | True |
| `BAAI/bge-reranker-base` | 278.0M | 1.000 | 1.000 | 1.000 | 2→1 PASS | PASS | 4889.3ms | 5624.1ms | 1.67s | True |

## Raw versus post-filter quality

| Model | Raw Top-1 | Filtered Top-1 | Raw MRR@5 | Filtered MRR@5 | Raw nDCG@5 | Filtered nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0.922 | 0.969 | 0.958 | 0.982 | 0.969 | 0.986 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | 0.969 | 1.000 | 0.984 | 1.000 | 0.988 | 1.000 |
| `BAAI/bge-reranker-base` | 0.953 | 1.000 | 0.977 | 1.000 | 0.983 | 1.000 |

## Language breakdown after filtering

| Model | Language | Top-1 | MRR@5 | nDCG@5 |
| --- | --- | ---: | ---: | ---: |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | de | 0.958 | 0.979 | 0.985 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | en | 1.000 | 1.000 | 1.000 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | mixed | 0.938 | 0.958 | 0.969 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | de | 1.000 | 1.000 | 1.000 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | en | 1.000 | 1.000 | 1.000 |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | mixed | 1.000 | 1.000 | 1.000 |
| `BAAI/bge-reranker-base` | de | 1.000 | 1.000 | 1.000 |
| `BAAI/bge-reranker-base` | en | 1.000 | 1.000 | 1.000 |
| `BAAI/bge-reranker-base` | mixed | 1.000 | 1.000 | 1.000 |

## Decision

**Recommendation: `cross-encoder/ms-marco-MiniLM-L-6-v2`.** Passed the English Lara gate and current-intent safety gate.
This is a benchmark recommendation only; productive configuration was not changed.

The Lara correction comes entirely from deterministic metadata selection: every raw model ranked `TEST-KFZ-2026-1001` first and `TEST-KFZ-2026-1003` second. After the selector retained the sole latest effective active motor policy, all models ranked `1003` first in all three measured runs.

MiniLM-L6 already satisfies the primary English gate with English Top-1/MRR@5/nDCG@5 of 1.000 and stable rankings. mMARCO has no measurable English-quality advantage in this fixture. Its diagnostic German/mixed results improve the overall Top-1 from MiniLM's 0.969 to 1.000, but its median CPU latency is 1.80x higher (843.5ms versus 467.8ms). Under the required selection order, that does not justify preferring mMARCO for the English production workload. A larger-model benchmark is therefore not required now.

### Quality gates

- English Lara current policy: PASS for all three models after filtering; raw rank was 2 for all three.
- Explicit old policy: PASS; the `1001_en` case remains rank 1 and the explicit-ID unit/integration selection is unchanged.
- Historical/comparison/list protection: PASS in targeted tests; the filter is not activated and both time slices remain available.
- Other English policy/claim regression: PASS; filtering activated only on the three mandatory Lara current variants, and all other candidate sets/rankings remained unchanged.
- Three-run stability: PASS for every model and case.
- Existing project runtime: PASS (`sentence-transformers=5.1.2`, `transformers=4.57.3`); no dependency change was made.
- Productive activation: intentionally not performed.

## Runtime and compatibility

- `cross-encoder/ms-marco-MiniLM-L-6-v2`: backend=cross_encoder; sentence-transformers=5.1.2; transformers=4.57.3; tokenizer/config limits=512/512; filtered latency range=55.0-972.2ms.
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`: backend=cross_encoder; sentence-transformers=5.1.2; transformers=4.57.3; tokenizer/config limits=512/514; filtered latency range=168.1-2380.9ms.
- `BAAI/bge-reranker-base`: backend=flag_embedding; sentence-transformers=5.1.2; transformers=4.57.3; tokenizer/config limits=512/514; filtered latency range=3667.1-19271.0ms.

All three adapters accepted max_length=512. Inputs are short, so no model-specific truncation was required.

## Per-query top 5

### cross-encoder/ms-marco-MiniLM-L-6-v2

| Case | Filter | Correct rank | Top 5 | Median latency |
| --- | --- | ---: | --- | ---: |
| `policy_test_kfz_2026_1001_en` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 368.2ms |
| `policy_test_kfz_2026_1001_de` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 439.1ms |
| `policy_test_kfz_2026_1001_mixed` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 488.4ms |
| `policy_test_phv_2026_1002_en` | unchanged | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902` | 390.9ms |
| `policy_test_phv_2026_1002_de` | unchanged | 2 | `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101`, `policy_test_phv_2025_1301` | 519.5ms |
| `policy_test_phv_2026_1002_mixed` | unchanged | 3 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_phv_2026_1601` | 424.9ms |
| `policy_test_kfz_2026_1003_en` | applied | 1 | `policy_test_kfz_2026_1003` | 68.2ms |
| `policy_test_kfz_2026_1003_de` | applied | 1 | `policy_test_kfz_2026_1003` | 149.0ms |
| `policy_test_kfz_2026_1003_mixed` | applied | 1 | `policy_test_kfz_2026_1003` | 59.5ms |
| `policy_test_kfz_2026_1101_en` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1901` | 454.6ms |
| `policy_test_kfz_2026_1101_de` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003` | 461.3ms |
| `policy_test_kfz_2026_1101_mixed` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1003` | 523.7ms |
| `policy_test_hh_2026_1201_en` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 428.0ms |
| `policy_test_hh_2026_1201_de` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1801`, `policy_test_phv_2026_1002`, `policy_test_rs_2026_1501` | 507.7ms |
| `policy_test_hh_2026_1201_mixed` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1501` | 408.8ms |
| `policy_test_rs_2026_1202_en` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1801` | 467.7ms |
| `policy_test_rs_2026_1202_de` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1702` | 461.1ms |
| `policy_test_rs_2026_1202_mixed` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 486.2ms |
| `policy_test_phv_2025_1301_en` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 389.5ms |
| `policy_test_phv_2025_1301_de` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1902` | 512.9ms |
| `policy_test_phv_2025_1301_mixed` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002` | 437.7ms |
| `policy_test_kfz_2026_1401_en` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001` | 443.5ms |
| `policy_test_kfz_2026_1401_de` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001` | 459.2ms |
| `policy_test_kfz_2026_1401_mixed` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 491.1ms |
| `policy_test_hh_2026_1402_en` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 444.1ms |
| `policy_test_hh_2026_1402_de` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_rs_2026_1501`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1202` | 573.2ms |
| `policy_test_hh_2026_1402_mixed` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1501` | 479.1ms |
| `policy_test_rs_2026_1501_en` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 634.5ms |
| `policy_test_rs_2026_1501_de` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201` | 551.2ms |
| `policy_test_rs_2026_1501_mixed` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1702` | 645.6ms |
| `policy_test_phv_2026_1601_en` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_phv_2025_1301` | 500.1ms |
| `policy_test_phv_2026_1601_de` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1902` | 687.4ms |
| `policy_test_phv_2026_1601_mixed` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002` | 508.7ms |
| `policy_test_kfz_2026_1701_en` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101` | 519.4ms |
| `policy_test_kfz_2026_1701_de` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001` | 561.8ms |
| `policy_test_kfz_2026_1701_mixed` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003` | 566.7ms |
| `policy_test_rs_2026_1702_en` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001` | 429.9ms |
| `policy_test_rs_2026_1702_de` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1801` | 536.0ms |
| `policy_test_rs_2026_1702_mixed` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1501`, `policy_test_hh_2026_1801` | 431.0ms |
| `policy_test_hh_2026_1801_en` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1501` | 477.0ms |
| `policy_test_hh_2026_1801_de` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1702`, `policy_test_rs_2026_1202` | 488.3ms |
| `policy_test_hh_2026_1801_mixed` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1702`, `policy_test_rs_2026_1501` | 523.4ms |
| `policy_test_kfz_2026_1901_en` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003` | 425.4ms |
| `policy_test_kfz_2026_1901_de` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101` | 512.0ms |
| `policy_test_kfz_2026_1901_mixed` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101` | 451.9ms |
| `policy_test_phv_2026_1902_en` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 507.2ms |
| `policy_test_phv_2026_1902_de` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 582.2ms |
| `policy_test_phv_2026_1902_mixed` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 489.3ms |
| `claim_test_clm_2026_2001_en` | unchanged | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601` | 368.4ms |
| `claim_test_clm_2026_2001_de` | unchanged | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301` | 457.0ms |
| `claim_test_clm_2026_2101_en` | unchanged | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602` | 385.9ms |
| `claim_test_clm_2026_2101_de` | unchanged | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501` | 475.9ms |
| `claim_test_clm_2026_2201_en` | unchanged | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 384.7ms |
| `claim_test_clm_2026_2201_de` | unchanged | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401` | 465.4ms |
| `claim_test_clm_2026_2301_en` | unchanged | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2601` | 389.2ms |
| `claim_test_clm_2026_2301_de` | unchanged | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602` | 475.6ms |
| `claim_test_clm_2026_2401_en` | unchanged | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 403.6ms |
| `claim_test_clm_2026_2401_de` | unchanged | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 519.2ms |
| `claim_test_clm_2026_2501_en` | unchanged | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2101` | 404.0ms |
| `claim_test_clm_2026_2501_de` | unchanged | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2301` | 492.8ms |
| `claim_test_clm_2026_2601_en` | unchanged | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2401` | 387.2ms |
| `claim_test_clm_2026_2601_de` | unchanged | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2101` | 471.9ms |
| `claim_test_clm_2026_2602_en` | unchanged | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2301` | 384.9ms |
| `claim_test_clm_2026_2602_de` | unchanged | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2101` | 490.0ms |

### cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

| Case | Filter | Correct rank | Top 5 | Median latency |
| --- | --- | ---: | --- | ---: |
| `policy_test_kfz_2026_1001_en` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 880.0ms |
| `policy_test_kfz_2026_1001_de` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901` | 1439.1ms |
| `policy_test_kfz_2026_1001_mixed` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 999.9ms |
| `policy_test_phv_2026_1002_en` | unchanged | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902` | 1015.2ms |
| `policy_test_phv_2026_1002_de` | unchanged | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902` | 1074.8ms |
| `policy_test_phv_2026_1002_mixed` | unchanged | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902` | 1034.9ms |
| `policy_test_kfz_2026_1003_en` | applied | 1 | `policy_test_kfz_2026_1003` | 179.2ms |
| `policy_test_kfz_2026_1003_de` | applied | 1 | `policy_test_kfz_2026_1003` | 169.6ms |
| `policy_test_kfz_2026_1003_mixed` | applied | 1 | `policy_test_kfz_2026_1003` | 346.2ms |
| `policy_test_kfz_2026_1101_en` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1701` | 1005.6ms |
| `policy_test_kfz_2026_1101_de` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1003` | 1057.5ms |
| `policy_test_kfz_2026_1101_mixed` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1701` | 1033.5ms |
| `policy_test_hh_2026_1201_en` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 1035.4ms |
| `policy_test_hh_2026_1201_de` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 1042.2ms |
| `policy_test_hh_2026_1201_mixed` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1801` | 863.9ms |
| `policy_test_rs_2026_1202_en` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1801` | 843.6ms |
| `policy_test_rs_2026_1202_de` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1801` | 813.1ms |
| `policy_test_rs_2026_1202_mixed` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702` | 843.3ms |
| `policy_test_phv_2025_1301_en` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 799.9ms |
| `policy_test_phv_2025_1301_de` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1001` | 849.5ms |
| `policy_test_phv_2025_1301_mixed` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1601` | 803.2ms |
| `policy_test_kfz_2026_1401_en` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003` | 818.2ms |
| `policy_test_kfz_2026_1401_de` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001` | 818.8ms |
| `policy_test_kfz_2026_1401_mixed` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1701` | 795.4ms |
| `policy_test_hh_2026_1402_en` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 803.8ms |
| `policy_test_hh_2026_1402_de` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1702` | 801.4ms |
| `policy_test_hh_2026_1402_mixed` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 758.2ms |
| `policy_test_rs_2026_1501_en` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702`, `policy_test_rs_2026_1202`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 809.1ms |
| `policy_test_rs_2026_1501_de` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702`, `policy_test_phv_2026_1002`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001` | 928.6ms |
| `policy_test_rs_2026_1501_mixed` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1702`, `policy_test_rs_2026_1202`, `policy_test_phv_2026_1002` | 897.6ms |
| `policy_test_phv_2026_1601_en` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001` | 830.7ms |
| `policy_test_phv_2026_1601_de` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_phv_2026_1902` | 833.8ms |
| `policy_test_phv_2026_1601_mixed` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1902`, `policy_test_phv_2025_1301`, `policy_test_kfz_2026_1001` | 945.6ms |
| `policy_test_kfz_2026_1701_en` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001` | 1139.2ms |
| `policy_test_kfz_2026_1701_de` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003` | 873.5ms |
| `policy_test_kfz_2026_1701_mixed` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1003` | 963.5ms |
| `policy_test_rs_2026_1702_en` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402` | 903.9ms |
| `policy_test_rs_2026_1702_de` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001` | 894.3ms |
| `policy_test_rs_2026_1702_mixed` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001` | 851.9ms |
| `policy_test_hh_2026_1801_en` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001` | 840.0ms |
| `policy_test_hh_2026_1801_de` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1201`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 798.3ms |
| `policy_test_hh_2026_1801_mixed` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1201`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 779.6ms |
| `policy_test_kfz_2026_1901_en` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001` | 836.9ms |
| `policy_test_kfz_2026_1901_de` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1101` | 830.3ms |
| `policy_test_kfz_2026_1901_mixed` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1701` | 804.9ms |
| `policy_test_phv_2026_1902_en` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1003` | 821.4ms |
| `policy_test_phv_2026_1902_de` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1003` | 1038.5ms |
| `policy_test_phv_2026_1902_mixed` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601` | 903.8ms |
| `claim_test_clm_2026_2001_en` | unchanged | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2201` | 848.5ms |
| `claim_test_clm_2026_2001_de` | unchanged | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2201` | 914.5ms |
| `claim_test_clm_2026_2101_en` | unchanged | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2301` | 875.3ms |
| `claim_test_clm_2026_2101_de` | unchanged | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2001` | 848.7ms |
| `claim_test_clm_2026_2201_en` | unchanged | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2101` | 836.2ms |
| `claim_test_clm_2026_2201_de` | unchanged | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301` | 783.2ms |
| `claim_test_clm_2026_2301_en` | unchanged | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2001` | 748.9ms |
| `claim_test_clm_2026_2301_de` | unchanged | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2602` | 664.0ms |
| `claim_test_clm_2026_2401_en` | unchanged | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301` | 821.3ms |
| `claim_test_clm_2026_2401_de` | unchanged | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2301` | 826.7ms |
| `claim_test_clm_2026_2501_en` | unchanged | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2201` | 785.3ms |
| `claim_test_clm_2026_2501_de` | unchanged | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2201` | 715.0ms |
| `claim_test_clm_2026_2601_en` | unchanged | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2501` | 763.5ms |
| `claim_test_clm_2026_2601_de` | unchanged | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2501` | 1094.8ms |
| `claim_test_clm_2026_2602_en` | unchanged | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2101` | 827.2ms |
| `claim_test_clm_2026_2602_de` | unchanged | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2201` | 866.4ms |

### BAAI/bge-reranker-base

| Case | Filter | Correct rank | Top 5 | Median latency |
| --- | --- | ---: | --- | ---: |
| `policy_test_kfz_2026_1001_en` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1901` | 13102.5ms |
| `policy_test_kfz_2026_1001_de` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901` | 9512.2ms |
| `policy_test_kfz_2026_1001_mixed` | unchanged | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1901` | 13920.3ms |
| `policy_test_phv_2026_1002_en` | unchanged | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1601` | 18382.9ms |
| `policy_test_phv_2026_1002_de` | unchanged | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101`, `policy_test_phv_2026_1601` | 14661.2ms |
| `policy_test_phv_2026_1002_mixed` | unchanged | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101`, `policy_test_phv_2026_1601` | 13394.5ms |
| `policy_test_kfz_2026_1003_en` | applied | 1 | `policy_test_kfz_2026_1003` | 3676.8ms |
| `policy_test_kfz_2026_1003_de` | applied | 1 | `policy_test_kfz_2026_1003` | 5979.7ms |
| `policy_test_kfz_2026_1003_mixed` | applied | 1 | `policy_test_kfz_2026_1003` | 4959.8ms |
| `policy_test_kfz_2026_1101_en` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1701` | 5547.2ms |
| `policy_test_kfz_2026_1101_de` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1701` | 5918.4ms |
| `policy_test_kfz_2026_1101_mixed` | unchanged | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1701` | 5293.1ms |
| `policy_test_hh_2026_1201_en` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_phv_2026_1002`, `policy_test_rs_2026_1702` | 5421.2ms |
| `policy_test_hh_2026_1201_de` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001` | 5737.0ms |
| `policy_test_hh_2026_1201_mixed` | unchanged | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1402` | 5220.7ms |
| `policy_test_rs_2026_1202_en` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702` | 5095.9ms |
| `policy_test_rs_2026_1202_de` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702` | 4527.9ms |
| `policy_test_rs_2026_1202_mixed` | unchanged | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 5499.7ms |
| `policy_test_phv_2025_1301_en` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 4889.9ms |
| `policy_test_phv_2025_1301_de` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1003` | 4888.8ms |
| `policy_test_phv_2025_1301_mixed` | unchanged | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1001` | 5281.6ms |
| `policy_test_kfz_2026_1401_en` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 4423.7ms |
| `policy_test_kfz_2026_1401_de` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1901` | 4893.2ms |
| `policy_test_kfz_2026_1401_mixed` | unchanged | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001` | 5647.1ms |
| `policy_test_hh_2026_1402_en` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1001` | 5048.6ms |
| `policy_test_hh_2026_1402_de` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1201`, `policy_test_kfz_2026_1001` | 4478.1ms |
| `policy_test_hh_2026_1402_mixed` | unchanged | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1201` | 4469.8ms |
| `policy_test_rs_2026_1501_en` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 5105.5ms |
| `policy_test_rs_2026_1501_de` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001` | 5212.3ms |
| `policy_test_rs_2026_1501_mixed` | unchanged | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1202`, `policy_test_phv_2026_1002` | 4452.1ms |
| `policy_test_phv_2026_1601_en` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1101` | 4769.4ms |
| `policy_test_phv_2026_1601_de` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 5377.2ms |
| `policy_test_phv_2026_1601_mixed` | unchanged | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003` | 4493.6ms |
| `policy_test_kfz_2026_1701_en` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001` | 4442.6ms |
| `policy_test_kfz_2026_1701_de` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 5016.5ms |
| `policy_test_kfz_2026_1701_mixed` | unchanged | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001` | 5210.6ms |
| `policy_test_rs_2026_1702_en` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1201` | 4409.4ms |
| `policy_test_rs_2026_1702_de` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 4800.2ms |
| `policy_test_rs_2026_1702_mixed` | unchanged | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1202` | 5211.3ms |
| `policy_test_hh_2026_1801_en` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 5291.1ms |
| `policy_test_hh_2026_1801_de` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 5079.6ms |
| `policy_test_hh_2026_1801_mixed` | unchanged | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1501`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 4530.9ms |
| `policy_test_kfz_2026_1901_en` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1401` | 5320.8ms |
| `policy_test_kfz_2026_1901_de` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001` | 4922.9ms |
| `policy_test_kfz_2026_1901_mixed` | unchanged | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101` | 4535.2ms |
| `policy_test_phv_2026_1902_en` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1003` | 4545.2ms |
| `policy_test_phv_2026_1902_de` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 5244.8ms |
| `policy_test_phv_2026_1902_mixed` | unchanged | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001` | 4320.4ms |
| `claim_test_clm_2026_2001_en` | unchanged | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 4137.0ms |
| `claim_test_clm_2026_2001_de` | unchanged | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2101` | 4219.2ms |
| `claim_test_clm_2026_2101_en` | unchanged | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2201` | 4758.4ms |
| `claim_test_clm_2026_2101_de` | unchanged | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301` | 4245.9ms |
| `claim_test_clm_2026_2201_en` | unchanged | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2301` | 4026.5ms |
| `claim_test_clm_2026_2201_de` | unchanged | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601` | 4251.3ms |
| `claim_test_clm_2026_2301_en` | unchanged | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2401` | 4878.0ms |
| `claim_test_clm_2026_2301_de` | unchanged | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602` | 4241.6ms |
| `claim_test_clm_2026_2401_en` | unchanged | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2601` | 4181.2ms |
| `claim_test_clm_2026_2401_de` | unchanged | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2601` | 4222.6ms |
| `claim_test_clm_2026_2501_en` | unchanged | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2101` | 4366.4ms |
| `claim_test_clm_2026_2501_de` | unchanged | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2201` | 4466.8ms |
| `claim_test_clm_2026_2601_en` | unchanged | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2501` | 4119.8ms |
| `claim_test_clm_2026_2601_de` | unchanged | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301` | 4243.6ms |
| `claim_test_clm_2026_2602_en` | unchanged | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2401` | 5026.7ms |
| `claim_test_clm_2026_2602_de` | unchanged | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2401` | 4455.1ms |
