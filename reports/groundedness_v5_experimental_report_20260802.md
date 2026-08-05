# Experimentelle Groundedness-v5-Auswertung

## Ergebnis

Die isolierte experimentelle Variante `fact_aware_claim_support_v5_experimental` liefert auf den technischen Weak Labels den Punkt **0.788800**. Dieser Wert ist noch **kein produktiver oder human-validierter Threshold**. Die produktive v4-Implementierung und Konfiguration wurden nicht verändert.

| Split | n | technische FAR | Precision | Recall | F1 | Coverage | akzeptierte High-Risk-Proxies |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibration | 237 | 0.00% | 100.00% | 87.29% | 93.21% | 43.46% | 0 |
| Validation | 52 | 0.00% | 100.00% | 95.12% | 97.50% | 75.00% | 0 |
| Technical Weak-Label Hold-out | 87 | 0.00% | 100.00% | 96.72% | 98.33% | 67.82% | 0 |

## Was v5 verändert

- Einzelne unbelegte Zahlen, Prozentsätze, Geldbeträge, Daten und Policenkennungen können nicht mehr durch lange, ansonsten kopierte Antworten verdünnt werden.
- Strukturierte Kunden-, Policen-, Status- und Deckungswidersprüche erzeugen konservative Caps.
- `current policy` wird anhand Referenzdatum, Status, Produkt und jüngstem Startdatum geprüft.
- PDF-Quelle und Seitenzahl von Zitaten werden gegen Dokumentmetadaten geprüft.
- Der minimale Claim-Support beeinflusst den Gesamtscore, damit ein schwacher Teilclaim nicht im Mittelwert verschwindet.

## False-Acceptance-Audit

Am rein diagnostischen v4-Punkt 0.871795 wurden 13 HIGH-Confidence-FAIL-Fälle akzeptiert. Unter v5 bei 0.788800 bleiben davon **0** akzeptiert. Details stehen in `groundedness_v5_false_acceptance_audit_20260802.csv`.

## Group Cross-Validation

Gültige Folds: 50/50. Threshold-Median: 0.7888; Spanne: [0.546488, 0.833286]. Mittlere Test-FAR: 0.022985797805604852. High-Risk-False-Acceptance-Anteil: 0.2.

## Wissenschaftliche Einschränkung

Alle Kennzahlen wurden weiterhin gegen `generator_expected_label` berechnet. Diese Labels sind technische Weak Labels und kein menschliches Ground Truth. Die Regeln wurden anhand derselben konstruierten Fallfamilien entwickelt; deshalb kann die sehr gute Trennung teilweise konstruktionsspezifisch sein.

## Nächster zwingender Schritt

Das Review-Paket enthält 365 Fälle aus Calibration und Validation in 103 ungeteilten Leakage-Gruppen. Zwei Reviewer bearbeiten getrennte, identische und verblindete CSV-Dateien. Danach werden Konflikte adjudiziert und v5 ausschließlich anhand der Human Labels neu kalibriert. Für eine finale Bewertung muss zusätzlich ein neuer, bisher nicht zur Regelentwicklung verwendeter Human Hold-out erhoben werden.

## Produktionsentscheidung

`0.788800` darf bis zur menschlichen Validierung nur für Shadow-Evaluation verwendet werden. Keine automatische Aktivierung.

No human ground truth was used. The experimental value is not a final human-validated production threshold.
