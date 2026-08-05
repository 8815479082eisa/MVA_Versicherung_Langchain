# Isolated local reranker benchmark

## Scope

This benchmark scored static query-document pairs only. It did not import or invoke the project RAG pipeline, Chroma, BM25, embeddings, LLMs, self-check, CRM, reindexing, or `/api/ask`.

- Cases: 64
- Candidates per case: 8 (production `RETRIEVE_TOP_K` default)
- Top documents reported: 5
- Warm-up runs: 1 per model
- Measured runs: 3 per query and model
- Device: CPU only; PyTorch threads: 8; FP16: disabled
- CrossEncoder runtime: sentence-transformers=5.1.2; transformers=4.57.3
- FlagEmbedding runs in a subprocess using the unchanged project runtime because FlagEmbedding and the Ettin-required Transformers version are mutually incompatible.
- FlagEmbedding's built-in per-call progress handling could not be disabled through its public scoring API and is included in BGE latency.
- Shared settings: max_length=512, batch_size=8
- Model downloads completed before model-load and inference timing.
- Peak RAM was not sampled to avoid background instrumentation affecting this short latency test.
- Host: Windows-11-10.0.26100-SP0

## Comparison

| Model | Parameters | Top-1 Accuracy | MRR@5 | nDCG@5 | Lara case | Mean latency | Median latency | Load time | Notes |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7M | 0.922 | 0.958 | 0.969 | FAIL | 693.8ms | 463.1ms | 0.22s | backend=cross_encoder; rankings stable=True; latency range 319.4-2506.7ms; tokenizer/config limits=512/512; runtime ST/TF=5.1.2/4.57.3 |

## Selection

**No winner selected.** No model passed the Lara Neumann rank-1 requirement in every measured run. No winner is selected.
Further measures should focus on metadata filtering, stronger domain hard negatives, or domain-specific fine-tuning.

## Per-query rankings

### cross-encoder/ms-marco-MiniLM-L-6-v2

| Query case | Correct rank | Top 5 candidate IDs | Median query latency |
| --- | ---: | --- | ---: |
| `policy_test_kfz_2026_1001_en` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 641.8ms |
| `policy_test_kfz_2026_1001_de` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 517.2ms |
| `policy_test_kfz_2026_1001_mixed` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 716.2ms |
| `policy_test_phv_2026_1002_en` | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902` | 1457.6ms |
| `policy_test_phv_2026_1002_de` | 2 | `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101`, `policy_test_phv_2025_1301` | 1171.4ms |
| `policy_test_phv_2026_1002_mixed` | 3 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_phv_2026_1601` | 1160.1ms |
| `policy_test_kfz_2026_1003_en` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 1040.0ms |
| `policy_test_kfz_2026_1003_de` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 1509.0ms |
| `policy_test_kfz_2026_1003_mixed` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 1135.2ms |
| `policy_test_kfz_2026_1101_en` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1901` | 1088.2ms |
| `policy_test_kfz_2026_1101_de` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003` | 1142.1ms |
| `policy_test_kfz_2026_1101_mixed` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1003` | 998.4ms |
| `policy_test_hh_2026_1201_en` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 1051.2ms |
| `policy_test_hh_2026_1201_de` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1801`, `policy_test_phv_2026_1002`, `policy_test_rs_2026_1501` | 1330.7ms |
| `policy_test_hh_2026_1201_mixed` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1501` | 1211.1ms |
| `policy_test_rs_2026_1202_en` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1801` | 983.1ms |
| `policy_test_rs_2026_1202_de` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1702` | 1179.5ms |
| `policy_test_rs_2026_1202_mixed` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 1088.6ms |
| `policy_test_phv_2025_1301_en` | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 1092.1ms |
| `policy_test_phv_2025_1301_de` | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1902` | 1947.4ms |
| `policy_test_phv_2025_1301_mixed` | 1 | `policy_test_phv_2025_1301`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002` | 1188.6ms |
| `policy_test_kfz_2026_1401_en` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001` | 1164.0ms |
| `policy_test_kfz_2026_1401_de` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001` | 1044.4ms |
| `policy_test_kfz_2026_1401_mixed` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 1127.3ms |
| `policy_test_hh_2026_1402_en` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 568.3ms |
| `policy_test_hh_2026_1402_de` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_rs_2026_1501`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1202` | 446.5ms |
| `policy_test_hh_2026_1402_mixed` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1501` | 414.2ms |
| `policy_test_rs_2026_1501_en` | 1 | `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 408.7ms |
| `policy_test_rs_2026_1501_de` | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201` | 497.2ms |
| `policy_test_rs_2026_1501_mixed` | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1702` | 432.3ms |
| `policy_test_phv_2026_1601_en` | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_phv_2025_1301` | 498.8ms |
| `policy_test_phv_2026_1601_de` | 1 | `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1902` | 535.1ms |
| `policy_test_phv_2026_1601_mixed` | 1 | `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002` | 486.1ms |
| `policy_test_kfz_2026_1701_en` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101` | 450.1ms |
| `policy_test_kfz_2026_1701_de` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001` | 498.6ms |
| `policy_test_kfz_2026_1701_mixed` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003` | 414.6ms |
| `policy_test_rs_2026_1702_en` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001` | 423.6ms |
| `policy_test_rs_2026_1702_de` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1801` | 453.6ms |
| `policy_test_rs_2026_1702_mixed` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1501`, `policy_test_hh_2026_1801` | 400.7ms |
| `policy_test_hh_2026_1801_en` | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1501` | 426.2ms |
| `policy_test_hh_2026_1801_de` | 1 | `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1702`, `policy_test_rs_2026_1202` | 470.2ms |
| `policy_test_hh_2026_1801_mixed` | 1 | `policy_test_hh_2026_1801`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1702`, `policy_test_rs_2026_1501` | 421.1ms |
| `policy_test_kfz_2026_1901_en` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003` | 428.8ms |
| `policy_test_kfz_2026_1901_de` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101` | 437.0ms |
| `policy_test_kfz_2026_1901_mixed` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101` | 392.6ms |
| `policy_test_phv_2026_1902_en` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 389.6ms |
| `policy_test_phv_2026_1902_de` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 462.8ms |
| `policy_test_phv_2026_1902_mixed` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 422.7ms |
| `claim_test_clm_2026_2001_en` | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601` | 334.5ms |
| `claim_test_clm_2026_2001_de` | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301` | 356.0ms |
| `claim_test_clm_2026_2101_en` | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602` | 343.2ms |
| `claim_test_clm_2026_2101_de` | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501` | 357.5ms |
| `claim_test_clm_2026_2201_en` | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 337.7ms |
| `claim_test_clm_2026_2201_de` | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401` | 358.5ms |
| `claim_test_clm_2026_2301_en` | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2601` | 330.6ms |
| `claim_test_clm_2026_2301_de` | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602` | 372.3ms |
| `claim_test_clm_2026_2401_en` | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 352.3ms |
| `claim_test_clm_2026_2401_de` | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 386.7ms |
| `claim_test_clm_2026_2501_en` | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2101` | 343.0ms |
| `claim_test_clm_2026_2501_de` | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2301` | 367.3ms |
| `claim_test_clm_2026_2601_en` | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2401` | 331.9ms |
| `claim_test_clm_2026_2601_de` | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2101` | 374.7ms |
| `claim_test_clm_2026_2602_en` | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2301` | 331.5ms |
| `claim_test_clm_2026_2602_de` | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2101` | 355.3ms |

## Input-length compatibility

All adapters were configured with max_length=512. The table notes the tokenizer and model-configuration limits exposed at runtime. No model required a lower benchmark limit; inputs were short and the shared limit applied uniformly.

## Metric interpretation

Raw scores are intentionally omitted because score scales differ between models. All quality metrics use rankings only. nDCG@5 uses fixture relevance grades 0, 1 and 3.
