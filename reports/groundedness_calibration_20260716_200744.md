# Groundedness Calibration Report

- Generated at: `2026-07-16T18:07:44.811901+00:00`
- Algorithm: `fact_aware_claim_support_v4`
- Dataset: `tests/fixtures/groundedness_calibration_cases.json`
- Dataset SHA-256: `cf92177df13ffd58289cfca9f541c02c2b55af55e026b718edbab5063b210edb`
- Cases: `28` (14 supported, 14 unsupported)
- Selected threshold: `0.51`
- Selection rule: Require recall >= 0.80 when feasible; minimize unsupported false acceptance; then maximize F1, recall, and precision; choose the lowest tied threshold.

## Selected Metrics

| TP | TN | FP | FN | Precision | Recall | F1 | False acceptance rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | 14 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0.000 |

## Case Results

| Case | Label | Score | Prediction |
|---|---|---:|---|
| supported_motor_full_answer | supported | 1.000 | accept |
| supported_motor_paraphrase | supported | 0.815 | accept |
| supported_contract_identifier | supported | 1.000 | accept |
| supported_deductible | supported | 0.887 | accept |
| supported_exclusion | supported | 0.800 | accept |
| supported_liability | supported | 0.849 | accept |
| supported_coverage_dates | supported | 0.903 | accept |
| supported_home_deductible | supported | 1.000 | accept |
| supported_travel_limit | supported | 1.000 | accept |
| supported_dental_percentage | supported | 1.000 | accept |
| supported_two_claims_with_citation | supported | 1.000 | accept |
| supported_short_numeric_answer | supported | 1.000 | accept |
| unsupported_wrong_motor_deductible | unsupported | 0.142 | reject |
| unsupported_wrong_contract_identifier | unsupported | 0.146 | reject |
| unsupported_wrong_coverage_type | unsupported | 0.142 | reject |
| unsupported_reversed_coverage | unsupported | 0.150 | reject |
| unsupported_zero_deductible | unsupported | 0.128 | reject |
| unsupported_rental_car_extension | unsupported | 0.333 | reject |
| unsupported_wrong_policy_dates | unsupported | 0.146 | reject |
| unsupported_wrong_home_deductible | unsupported | 0.146 | reject |
| unsupported_wrong_baggage_limit | unsupported | 0.142 | reject |
| unsupported_wrong_dental_percentage | unsupported | 0.146 | reject |
| unsupported_mixed_extra_claim | unsupported | 0.500 | reject |
| unsupported_cancellation_claim | unsupported | 0.278 | reject |
| supported_customer_association | supported | 0.583 | accept |
| supported_concise_coverage | supported | 1.000 | accept |
| unsupported_liability_distractor_answer | unsupported | 0.141 | reject |
| unsupported_information_missing | unsupported | 0.076 | reject |

## Interpretation

The selected operating point prioritizes rejecting unsupported claims. The threshold is loaded from the stable calibration artifact, not copied into `.env`.
