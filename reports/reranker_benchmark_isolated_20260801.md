# Isolated local reranker benchmark

## Scope

This benchmark scored static query-document pairs only. It did not import or invoke the project RAG pipeline, Chroma, BM25, embeddings, LLMs, self-check, CRM, reindexing, or `/api/ask`.

- Cases: 6
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
| `BAAI/bge-reranker-base` | 278.0M | 0.667 | 0.833 | 0.893 | PASS | 1731.0ms | 1724.5ms | 1.07s | backend=flag_embedding; rankings stable=True; latency range 1445.5-2079.0ms; tokenizer/config limits=512/514; runtime ST/TF=5.1.2/4.57.3 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22.7M | 0.833 | 0.917 | 0.942 | PASS | 135.0ms | 131.3ms | 0.16s | backend=cross_encoder; rankings stable=True; latency range 112.6-181.6ms; tokenizer/config limits=512/512; runtime ST/TF=5.4.1/5.7.0 |
| `cross-encoder/ettin-reranker-32m-v1` | 31.9M | 0.667 | 0.833 | 0.890 | PASS | 161.0ms | 158.9ms | 0.47s | backend=cross_encoder; rankings stable=True; latency range 148.3-182.0ms; tokenizer/config limits=512/7999; runtime ST/TF=5.4.1/5.7.0 |
| `cross-encoder/ettin-reranker-68m-v1` | 68.1M | 0.667 | 0.833 | 0.896 | PASS | 490.5ms | 465.0ms | 0.60s | backend=cross_encoder; rankings stable=True; latency range 444.2-628.6ms; tokenizer/config limits=512/7999; runtime ST/TF=5.4.1/5.7.0 |

## Selection

**Winner: `cross-encoder/ms-marco-MiniLM-L-6-v2`.** All mandatory-case failures were excluded first. Ranking quality was compared by Top-1 Accuracy, MRR@5 and nDCG@5; median latency broke near-equal quality ties.

## Per-query rankings

### BAAI/bge-reranker-base

| Query case | Correct rank | Top 5 candidate IDs | Median query latency |
| --- | ---: | --- | ---: |
| `lara_current_motor_policy` | 1 | `lara_motor_1003_current`, `lara_liability_1002`, `lara_motor_1001_old`, `lara_customer_summary`, `generic_motor_terms` | 1894.6ms |
| `sophie_active_household_policy` | 1 | `sophie_household_1201`, `sophie_legal_1202`, `sophie_customer_summary`, `generic_household_terms`, `hannah_household_1402` | 1651.2ms |
| `noah_active_motor_policy` | 2 | `noah_customer_summary`, `noah_motor_1701_exact`, `noah_legal_1702`, `generic_motor_policy_record`, `oliver_motor_1101_for_noah` | 1847.3ms |
| `hannah_active_household_term` | 1 | `hannah_household_1402_exact`, `hannah_motor_1401_same_customer`, `hannah_customer_summary`, `generic_household_policy_schedule`, `sophie_household_1201_for_hannah` | 1546.4ms |
| `jonas_cancelled_legal_policy` | 1 | `jonas_legal_1501_exact`, `jonas_customer_summary`, `generic_legal_terms`, `noah_legal_1702_for_jonas`, `sophie_legal_1202_for_jonas` | 1551.7ms |
| `oliver_comprehensive_motor_policy` | 2 | `oliver_customer_summary`, `oliver_motor_1101_exact`, `generic_comprehensive_terms`, `noah_motor_1701_for_oliver`, `leon_motor_1901` | 1804.5ms |

### cross-encoder/ms-marco-MiniLM-L-6-v2

| Query case | Correct rank | Top 5 candidate IDs | Median query latency |
| --- | ---: | --- | ---: |
| `lara_current_motor_policy` | 1 | `lara_motor_1003_current`, `lara_motor_1001_old`, `lara_liability_1002`, `lara_customer_summary`, `generic_motor_terms` | 170.9ms |
| `sophie_active_household_policy` | 1 | `sophie_household_1201`, `sophie_legal_1202`, `sophie_customer_summary`, `hannah_household_1402`, `mila_household_1801` | 131.1ms |
| `noah_active_motor_policy` | 1 | `noah_motor_1701_exact`, `noah_customer_summary`, `noah_legal_1702`, `oliver_motor_1101_for_noah`, `lara_motor_1001_for_noah` | 137.2ms |
| `hannah_active_household_term` | 1 | `hannah_household_1402_exact`, `hannah_motor_1401_same_customer`, `hannah_customer_summary`, `sophie_household_1201_for_hannah`, `generic_household_policy_schedule` | 119.3ms |
| `jonas_cancelled_legal_policy` | 1 | `jonas_legal_1501_exact`, `jonas_customer_summary`, `generic_legal_terms`, `noah_legal_1702_for_jonas`, `sophie_legal_1202_for_jonas` | 116.3ms |
| `oliver_comprehensive_motor_policy` | 2 | `oliver_customer_summary`, `oliver_motor_1101_exact`, `generic_comprehensive_terms`, `leon_motor_1901`, `noah_motor_1701_for_oliver` | 131.5ms |

### cross-encoder/ettin-reranker-32m-v1

| Query case | Correct rank | Top 5 candidate IDs | Median query latency |
| --- | ---: | --- | ---: |
| `lara_current_motor_policy` | 1 | `lara_motor_1003_current`, `lara_motor_1001_old`, `lara_liability_1002`, `lara_customer_summary`, `oliver_motor_1101` | 181.6ms |
| `sophie_active_household_policy` | 1 | `sophie_household_1201`, `sophie_legal_1202`, `sophie_customer_summary`, `hannah_household_1402`, `lara_liability_for_sophie` | 158.9ms |
| `noah_active_motor_policy` | 1 | `noah_motor_1701_exact`, `noah_legal_1702`, `noah_customer_summary`, `generic_motor_policy_record`, `oliver_motor_1101_for_noah` | 158.8ms |
| `hannah_active_household_term` | 1 | `hannah_household_1402_exact`, `hannah_motor_1401_same_customer`, `hannah_customer_summary`, `generic_household_policy_schedule`, `sophie_household_1201_for_hannah` | 153.4ms |
| `jonas_cancelled_legal_policy` | 2 | `jonas_customer_summary`, `jonas_legal_1501_exact`, `generic_legal_terms`, `sophie_legal_1202_for_jonas`, `noah_legal_1702_for_jonas` | 160.7ms |
| `oliver_comprehensive_motor_policy` | 2 | `oliver_customer_summary`, `oliver_motor_1101_exact`, `generic_comprehensive_terms`, `leon_motor_1901`, `hannah_motor_1401_for_oliver` | 159.6ms |

### cross-encoder/ettin-reranker-68m-v1

| Query case | Correct rank | Top 5 candidate IDs | Median query latency |
| --- | ---: | --- | ---: |
| `lara_current_motor_policy` | 1 | `lara_motor_1003_current`, `lara_motor_1001_old`, `lara_liability_1002`, `lara_customer_summary`, `noah_motor_1701` | 611.9ms |
| `sophie_active_household_policy` | 1 | `sophie_household_1201`, `sophie_legal_1202`, `sophie_customer_summary`, `hannah_household_1402`, `generic_household_terms` | 458.5ms |
| `noah_active_motor_policy` | 1 | `noah_motor_1701_exact`, `noah_customer_summary`, `noah_legal_1702`, `generic_motor_policy_record`, `hannah_motor_1401_for_noah` | 478.9ms |
| `hannah_active_household_term` | 1 | `hannah_household_1402_exact`, `hannah_customer_summary`, `hannah_motor_1401_same_customer`, `sophie_household_1201_for_hannah`, `generic_household_policy_schedule` | 444.7ms |
| `jonas_cancelled_legal_policy` | 2 | `jonas_customer_summary`, `jonas_legal_1501_exact`, `generic_legal_terms`, `noah_legal_1702_for_jonas`, `sophie_legal_1202_for_jonas` | 462.8ms |
| `oliver_comprehensive_motor_policy` | 2 | `oliver_customer_summary`, `oliver_motor_1101_exact`, `generic_comprehensive_terms`, `lara_motor_1003_for_oliver`, `hannah_motor_1401_for_oliver` | 461.0ms |

## Input-length compatibility

All adapters were configured with max_length=512. The table notes the tokenizer and model-configuration limits exposed at runtime. No model required a lower benchmark limit; inputs were short and the shared limit applied uniformly.

## Metric interpretation

Raw scores are intentionally omitted because score scales differ between models. All quality metrics use rankings only. nDCG@5 uses fixture relevance grades 0, 1 and 3.
