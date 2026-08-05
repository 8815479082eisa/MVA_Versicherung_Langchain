# Groundedness v5: Vergleich 0.780000 vs. 0.788800

## Ergebnis

**Neither 0.780000 nor 0.788800 is sufficient under the absolute all-confidence safety criteria.** Im direkten Vergleich gilt zusätzlich: **0.780000 is not preferred over 0.788800.** Der niedrigere Wert akzeptiert 1 zusätzlichen Fall: 0 PASS, 1 FAIL und 0 High-Risk-Proxy-FAILs. Es gibt einen einzelnen maßgeblichen Grenzfall.

## Integritätsprüfung der gespeicherten Scores

- 498 eindeutige Fälle und 498 vollständige Scores.
- Algorithmusversion: `fact_aware_claim_support_v5_experimental`.
- Dataset-SHA, Fall-IDs, Splits und Leakage-Gruppen stimmen überein.
- Der v5-Code ist zeitlich nicht neuer als die gespeicherten Scores.
- Scores wurden **nicht neu berechnet**.
- Provenienzgrenze: In allen 498 gespeicherten Scorezeilen fehlt `input_sha256`; auch ein damaliger Code-Hash fehlt. Aktuelle Input-Hashes wurden berechnet und für Boundary Cases gespeichert, können aber nicht rückwirkend kryptografisch verglichen werden.

## Calibration

| Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | Coverage | High-Risk-FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.780000 | 110 | 4 | 146 | 17 | 2.67% | 96.49% | 86.61% | 91.29% | 41.16% | 4 |
| 0.788800 | 110 | 4 | 146 | 17 | 2.67% | 96.49% | 86.61% | 91.29% | 41.16% | 4 |

## Validation

| Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | Coverage | High-Risk-FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.780000 | 41 | 0 | 45 | 2 | 0.00% | 100.00% | 95.35% | 97.62% | 46.59% | 0 |
| 0.788800 | 41 | 0 | 45 | 2 | 0.00% | 100.00% | 95.35% | 97.62% | 46.59% | 0 |

## Technical Weak-Label Hold-out

| Threshold | TP | FP | TN | FN | FAR | Precision | Recall | F1 | Coverage | High-Risk-FA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.780000 | 62 | 1 | 68 | 2 | 1.45% | 98.41% | 96.88% | 97.64% | 47.37% | 0 |
| 0.788800 | 62 | 0 | 69 | 2 | 0.00% | 100.00% | 96.88% | 98.41% | 46.62% | 0 |

## Boundary-Case-Audit

| case_id | Score | Split | Quelle | Weak Label | Mutation | technische Auswirkung |
|---|---:|---|---|---|---|---|
| ext-iqa-fail-049 | 0.781316 | LOCKED_HOLDOUT_CANDIDATE | insuranceqa | FAIL | partially_supported_answer | additional_false_accept_of_technical_fail |

Quellen der Boundary Cases: `{"insuranceqa": 1}`. Die vollständigen Fragen, Antworten und Context-Zusammenfassungen stehen in `groundedness_v5_threshold_078_boundary_cases.csv`.

## Fester Group-CV-Vergleich

Beide Thresholds wurden in exakt denselben 50 Testfolds ausgewertet; es fand keine Neuauswahl je Fold statt.

| Threshold | Mean FAR | Median FAR | Mean Precision | Mean Recall | Mean F1 | Mean Coverage | Folds FAR>5% | High-Risk-Folds | Anteil | Maximum/Fold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.780000 | 0.00% | 0.00% | 100.00% | 91.86% | 95.62% | 69.65% | 0 | 0 | 0.00% | 0 |
| 0.788800 | 0.00% | 0.00% | 100.00% | 91.86% | 95.62% | 69.65% | 0 | 0 | 0.00% | 0 |

## Sicherheitsentscheidung

Beide Thresholds halten auf allen drei vollständigen Splits die FAR-Grenze von 5 % ein. Beide akzeptieren jedoch in Calibration vier High-Risk-Proxy-FAILs aus Confidence-Stufen außerhalb der primären HIGH-Menge und verfehlen damit die absolute Sicherheitsbedingung. Auf der primären HIGH-Menge sowie in den 50 festen Group-CV-Folds akzeptiert keiner der beiden Werte einen High-Risk-Proxy-FAIL.

0.780000 gewinnt keinen technischen PASS-Fall, sondern akzeptiert ausschließlich einen zusätzlichen FAIL-Fall im Hold-out. Recall bleibt unverändert; die minimale Coverage-Zunahme ist eine falsche Akzeptanz und kein Nutzen. Die Vergleichsentscheidung hängt vollständig von diesem einzelnen Boundary Case ab. Somit ist keiner der beiden Thresholds nach der absoluten All-Confidence-Regel ausreichend; 0.788800 bleibt innerhalb des experimentellen Shadow-Vergleichs der sicherere Kandidat.

Der produktive Algorithmus bleibt `fact_aware_claim_support_v4`, der produktive Threshold bleibt `0.51`.

No human ground truth was used. This comparison is an experimental weak-label shadow evaluation and does not authorize a production threshold change.
