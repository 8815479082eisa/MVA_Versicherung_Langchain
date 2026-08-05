# Isolated local reranker benchmark

## Scope

This benchmark scored static query-document pairs only. It did not import or invoke the project RAG pipeline, Chroma, BM25, embeddings, LLMs, self-check, CRM, reindexing, or `/api/ask`.

- Cases: 64
- Candidates per case: 8 (production `RETRIEVE_TOP_K` default)
- Top documents reported: 5
- Warm-up runs: 1 per model
- Measured runs: 3 per query and model
- Device: CPU only; PyTorch threads: 8; FP16: disabled
- CrossEncoder runtime: sentence-transformers=5.4.1; transformers=5.7.0
- FlagEmbedding runs in a subprocess using the unchanged project runtime because FlagEmbedding and the Ettin-required Transformers version are mutually incompatible.
- FlagEmbedding's built-in per-call progress handling could not be disabled through its public scoring API and is included in BGE latency.
- Shared settings: max_length=512, batch_size=8
- Model downloads completed before model-load and inference timing.
- Peak RAM was not sampled to avoid background instrumentation affecting this short latency test.
- Host: Windows-11-10.0.26100-SP0

## Comparison

| Model | Parameters | Top-1 Accuracy | MRR@5 | nDCG@5 | Lara case | Mean latency | Median latency | Load time | Notes |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `BAAI/bge-reranker-base` | 278.0M | 0.953 | 0.977 | 0.983 | FAIL | 5857.3ms | 5687.7ms | 2.36s | backend=flag_embedding; rankings stable=True; latency range 4026.4-12400.9ms; tokenizer/config limits=512/514; runtime ST/TF=5.1.2/4.57.3 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7M | 0.922 | 0.958 | 0.969 | FAIL | 400.6ms | 394.4ms | 0.15s | backend=cross_encoder; rankings stable=True; latency range 319.8-575.0ms; tokenizer/config limits=512/512; runtime ST/TF=5.4.1/5.7.0 |

## Selection

**No winner selected.** No model passed the Lara Neumann rank-1 requirement in every measured run. No winner is selected.
Further measures should focus on metadata filtering, stronger domain hard negatives, or domain-specific fine-tuning.

## Per-query rankings

### BAAI/bge-reranker-base

| Query case | Correct rank | Top 5 candidate IDs | Median query latency |
| --- | ---: | --- | ---: |
| `policy_test_kfz_2026_1001_en` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1901` | 10436.1ms |
| `policy_test_kfz_2026_1001_de` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901` | 9492.9ms |
| `policy_test_kfz_2026_1001_mixed` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1901` | 11880.2ms |
| `policy_test_phv_2026_1002_en` | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1601` | 6682.1ms |
| `policy_test_phv_2026_1002_de` | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101`, `policy_test_phv_2026_1601` | 6167.2ms |
| `policy_test_phv_2026_1002_mixed` | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101`, `policy_test_phv_2026_1601` | 5800.8ms |
| `policy_test_kfz_2026_1003_en` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1101` | 5835.8ms |
| `policy_test_kfz_2026_1003_de` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1101` | 6072.4ms |
| `policy_test_kfz_2026_1003_mixed` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1901` | 5972.1ms |
| `policy_test_kfz_2026_1101_en` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1701` | 5807.9ms |
| `policy_test_kfz_2026_1101_de` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1701` | 6018.6ms |
| `policy_test_kfz_2026_1101_mixed` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1701` | 6056.8ms |
| `policy_test_hh_2026_1201_en` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_phv_2026_1002`, `policy_test_rs_2026_1702` | 5858.7ms |
| `policy_test_hh_2026_1201_de` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001` | 5780.6ms |
| `policy_test_hh_2026_1201_mixed` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1402` | 5593.9ms |
| `policy_test_rs_2026_1202_en` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702` | 5880.8ms |
| `policy_test_rs_2026_1202_de` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702` | 5983.9ms |
| `policy_test_rs_2026_1202_mixed` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 5732.3ms |
| `policy_test_phv_2025_1301_en` | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 6089.4ms |
| `policy_test_phv_2025_1301_de` | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1003` | 6278.9ms |
| `policy_test_phv_2025_1301_mixed` | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1001` | 5699.3ms |
| `policy_test_kfz_2026_1401_en` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 5576.9ms |
| `policy_test_kfz_2026_1401_de` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1901` | 7551.2ms |
| `policy_test_kfz_2026_1401_mixed` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001` | 6412.9ms |
| `policy_test_hh_2026_1402_en` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1001` | 6134.4ms |
| `policy_test_hh_2026_1402_de` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1201`, `policy_test_kfz_2026_1001` | 5471.2ms |
| `policy_test_hh_2026_1402_mixed` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1201` | 5556.1ms |
| `policy_test_rs_2026_1501_en` | 1 | `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 5405.2ms |
| `policy_test_rs_2026_1501_de` | 1 | `policy_test_rs_2026_1501`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001` | 5600.9ms |
| `policy_test_rs_2026_1501_mixed` | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1202`, `policy_test_phv_2026_1002` | 5433.0ms |
| `policy_test_phv_2026_1601_en` | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1101` | 6115.5ms |
| `policy_test_phv_2026_1601_de` | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 5613.3ms |
| `policy_test_phv_2026_1601_mixed` | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003` | 5771.7ms |
| `policy_test_kfz_2026_1701_en` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001` | 5769.4ms |
| `policy_test_kfz_2026_1701_de` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 6128.5ms |
| `policy_test_kfz_2026_1701_mixed` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001` | 5567.8ms |
| `policy_test_rs_2026_1702_en` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1201` | 5490.7ms |
| `policy_test_rs_2026_1702_de` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 5546.4ms |
| `policy_test_rs_2026_1702_mixed` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1202` | 5527.1ms |
| `policy_test_hh_2026_1801_en` | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 7023.2ms |
| `policy_test_hh_2026_1801_de` | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 6583.3ms |
| `policy_test_hh_2026_1801_mixed` | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1501`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 5882.7ms |
| `policy_test_kfz_2026_1901_en` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1401` | 5434.8ms |
| `policy_test_kfz_2026_1901_de` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001` | 5920.2ms |
| `policy_test_kfz_2026_1901_mixed` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101` | 5388.7ms |
| `policy_test_phv_2026_1902_en` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1003` | 5406.3ms |
| `policy_test_phv_2026_1902_de` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 5763.5ms |
| `policy_test_phv_2026_1902_mixed` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001` | 5664.6ms |
| `claim_test_clm_2026_2001_en` | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 5249.8ms |
| `claim_test_clm_2026_2001_de` | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2101` | 5532.2ms |
| `claim_test_clm_2026_2101_en` | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2201` | 5017.2ms |
| `claim_test_clm_2026_2101_de` | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301` | 5473.5ms |
| `claim_test_clm_2026_2201_en` | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2301` | 5127.2ms |
| `claim_test_clm_2026_2201_de` | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601` | 5195.4ms |
| `claim_test_clm_2026_2301_en` | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2401` | 5146.0ms |
| `claim_test_clm_2026_2301_de` | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602` | 5967.9ms |
| `claim_test_clm_2026_2401_en` | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2601` | 5111.4ms |
| `claim_test_clm_2026_2401_de` | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2601` | 5267.6ms |
| `claim_test_clm_2026_2501_en` | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2101` | 4453.8ms |
| `claim_test_clm_2026_2501_de` | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2201` | 4359.1ms |
| `claim_test_clm_2026_2601_en` | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2501` | 4169.8ms |
| `claim_test_clm_2026_2601_de` | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301` | 4203.9ms |
| `claim_test_clm_2026_2602_en` | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2401` | 4175.3ms |
| `claim_test_clm_2026_2602_de` | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2401` | 4261.9ms |

### cross-encoder/ms-marco-MiniLM-L-6-v2

| Query case | Correct rank | Top 5 candidate IDs | Median query latency |
| --- | ---: | --- | ---: |
| `policy_test_kfz_2026_1001_en` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 383.6ms |
| `policy_test_kfz_2026_1001_de` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 436.0ms |
| `policy_test_kfz_2026_1001_mixed` | 1 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 394.2ms |
| `policy_test_phv_2026_1002_en` | 1 | `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902` | 377.0ms |
| `policy_test_phv_2026_1002_de` | 2 | `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101`, `policy_test_phv_2025_1301` | 431.2ms |
| `policy_test_phv_2026_1002_mixed` | 3 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_phv_2026_1601` | 392.7ms |
| `policy_test_kfz_2026_1003_en` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 401.1ms |
| `policy_test_kfz_2026_1003_de` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 445.4ms |
| `policy_test_kfz_2026_1003_mixed` | 2 | `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1401` | 407.7ms |
| `policy_test_kfz_2026_1101_en` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1901` | 379.9ms |
| `policy_test_kfz_2026_1101_de` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003` | 424.5ms |
| `policy_test_kfz_2026_1101_mixed` | 1 | `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1003` | 391.0ms |
| `policy_test_hh_2026_1201_en` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 375.4ms |
| `policy_test_hh_2026_1201_de` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1801`, `policy_test_phv_2026_1002`, `policy_test_rs_2026_1501` | 416.8ms |
| `policy_test_hh_2026_1201_mixed` | 1 | `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1501` | 376.2ms |
| `policy_test_rs_2026_1202_en` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1702`, `policy_test_hh_2026_1801` | 385.3ms |
| `policy_test_rs_2026_1202_de` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1702` | 424.6ms |
| `policy_test_rs_2026_1202_mixed` | 1 | `policy_test_rs_2026_1202`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_rs_2026_1501` | 435.7ms |
| `policy_test_phv_2025_1301_en` | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 379.8ms |
| `policy_test_phv_2025_1301_de` | 1 | `policy_test_phv_2025_1301`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1902` | 452.0ms |
| `policy_test_phv_2025_1301_mixed` | 1 | `policy_test_phv_2025_1301`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002` | 396.4ms |
| `policy_test_kfz_2026_1401_en` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001` | 385.9ms |
| `policy_test_kfz_2026_1401_de` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1001` | 428.5ms |
| `policy_test_kfz_2026_1401_mixed` | 1 | `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1701` | 450.2ms |
| `policy_test_hh_2026_1402_en` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_hh_2026_1201`, `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001` | 445.1ms |
| `policy_test_hh_2026_1402_de` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_rs_2026_1501`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1202` | 480.3ms |
| `policy_test_hh_2026_1402_mixed` | 1 | `policy_test_hh_2026_1402`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1501` | 418.8ms |
| `policy_test_rs_2026_1501_en` | 1 | `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1702`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1001` | 381.8ms |
| `policy_test_rs_2026_1501_de` | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1801`, `policy_test_hh_2026_1201` | 466.5ms |
| `policy_test_rs_2026_1501_mixed` | 1 | `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002`, `policy_test_hh_2026_1801`, `policy_test_rs_2026_1702` | 439.1ms |
| `policy_test_phv_2026_1601_en` | 1 | `policy_test_phv_2026_1601`, `policy_test_phv_2026_1902`, `policy_test_phv_2026_1002`, `policy_test_kfz_2026_1401`, `policy_test_phv_2025_1301` | 401.8ms |
| `policy_test_phv_2026_1601_de` | 1 | `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1902` | 476.6ms |
| `policy_test_phv_2026_1601_mixed` | 1 | `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_phv_2026_1002` | 399.3ms |
| `policy_test_kfz_2026_1701_en` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101` | 370.7ms |
| `policy_test_kfz_2026_1701_de` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001` | 433.4ms |
| `policy_test_kfz_2026_1701_mixed` | 1 | `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003` | 410.7ms |
| `policy_test_rs_2026_1702_en` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1501`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001` | 383.6ms |
| `policy_test_rs_2026_1702_de` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_rs_2026_1501`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1801` | 431.9ms |
| `policy_test_rs_2026_1702_mixed` | 1 | `policy_test_rs_2026_1702`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1501`, `policy_test_hh_2026_1801` | 393.8ms |
| `policy_test_hh_2026_1801_en` | 1 | `policy_test_hh_2026_1801`, `policy_test_hh_2026_1402`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1202`, `policy_test_rs_2026_1501` | 376.3ms |
| `policy_test_hh_2026_1801_de` | 1 | `policy_test_hh_2026_1801`, `policy_test_kfz_2026_1001`, `policy_test_hh_2026_1201`, `policy_test_rs_2026_1702`, `policy_test_rs_2026_1202` | 440.9ms |
| `policy_test_hh_2026_1801_mixed` | 1 | `policy_test_hh_2026_1801`, `policy_test_rs_2026_1202`, `policy_test_kfz_2026_1001`, `policy_test_rs_2026_1702`, `policy_test_rs_2026_1501` | 401.3ms |
| `policy_test_kfz_2026_1901_en` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1101`, `policy_test_kfz_2026_1003` | 378.2ms |
| `policy_test_kfz_2026_1901_de` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1701`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101` | 460.5ms |
| `policy_test_kfz_2026_1901_mixed` | 1 | `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1001`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1101` | 457.0ms |
| `policy_test_phv_2026_1902_en` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_phv_2026_1002`, `policy_test_phv_2026_1601`, `policy_test_kfz_2026_1401` | 413.0ms |
| `policy_test_phv_2026_1902_de` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 450.1ms |
| `policy_test_phv_2026_1902_mixed` | 1 | `policy_test_phv_2026_1902`, `policy_test_kfz_2026_1901`, `policy_test_kfz_2026_1003`, `policy_test_kfz_2026_1001`, `policy_test_phv_2026_1002` | 392.2ms |
| `claim_test_clm_2026_2001_en` | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601` | 324.7ms |
| `claim_test_clm_2026_2001_de` | 1 | `claim_test_clm_2026_2001`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301` | 367.4ms |
| `claim_test_clm_2026_2101_en` | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602` | 327.9ms |
| `claim_test_clm_2026_2101_de` | 1 | `claim_test_clm_2026_2101`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501` | 376.6ms |
| `claim_test_clm_2026_2201_en` | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 345.0ms |
| `claim_test_clm_2026_2201_de` | 1 | `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2401` | 385.9ms |
| `claim_test_clm_2026_2301_en` | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2401`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2601` | 342.6ms |
| `claim_test_clm_2026_2301_de` | 1 | `claim_test_clm_2026_2301`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602` | 380.8ms |
| `claim_test_clm_2026_2401_en` | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 353.1ms |
| `claim_test_clm_2026_2401_de` | 1 | `claim_test_clm_2026_2401`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2101` | 397.0ms |
| `claim_test_clm_2026_2501_en` | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2101` | 332.6ms |
| `claim_test_clm_2026_2501_de` | 1 | `claim_test_clm_2026_2501`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2201`, `claim_test_clm_2026_2301` | 360.6ms |
| `claim_test_clm_2026_2601_en` | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2101`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2401` | 327.9ms |
| `claim_test_clm_2026_2601_de` | 1 | `claim_test_clm_2026_2601`, `claim_test_clm_2026_2602`, `claim_test_clm_2026_2301`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2101` | 352.6ms |
| `claim_test_clm_2026_2602_en` | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2301` | 335.1ms |
| `claim_test_clm_2026_2602_de` | 1 | `claim_test_clm_2026_2602`, `claim_test_clm_2026_2601`, `claim_test_clm_2026_2501`, `claim_test_clm_2026_2001`, `claim_test_clm_2026_2101` | 355.7ms |

## Input-length compatibility

All adapters were configured with max_length=512. The table notes the tokenizer and model-configuration limits exposed at runtime. No model required a lower benchmark limit; inputs were short and the shared limit applied uniformly.

## Metric interpretation

Raw scores are intentionally omitted because score scales differ between models. All quality metrics use rankings only. nDCG@5 uses fixture relevance grades 0, 1 and 3.
