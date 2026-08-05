# Entscheidung zur MiniLM-Umstellung

## Ergebnis

Die produktive Reranker-Konfiguration wurde **nicht** auf MiniLM umgestellt.
Der erweiterte isolierte Regressionstest erfüllt die zuvor festgelegte
Qualitätsbedingung nicht: Weder `BAAI/bge-reranker-base` noch
`cross-encoder/ms-marco-MiniLM-L-6-v2` setzt die aktuelle Lara-Neumann-Police
`TEST-KFZ-2026-1003` in den drei englischen, deutschen und gemischten
Pflichtvarianten auf Rang 1. Beide Modelle ordnen stattdessen die ältere Police
`TEST-KFZ-2026-1001` auf Rang 1 und die aktuelle Police auf Rang 2 ein.

## Testumfang

- 64 statische, reproduzierbare Fälle aus den vorhandenen Policen- und
  Schaden-Fixtures
- 24 englische, 24 deutsche und 16 gemischte Queries
- 8 Kandidaten pro Query, darunter kunden-, produkt-, status- und
  schadentypnahe Hard Negatives
- Kandidatentexte mit 660 bis 791 Zeichen
- CPU-only, FP16 deaktiviert, `max_length=512`, `batch_size=8`
- ein Warm-up und drei gemessene Läufe pro Query und Modell
- keine Retrieval-, Embedding-, LLM-, CRM-, Self-Check- oder Reindex-Aufrufe

## Vergleich

| Modell | Top-1 | MRR@5 | nDCG@5 | Lara | Median CPU-Latenz | Load-Zeit |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `BAAI/bge-reranker-base` | 0,953 | 0,977 | 0,983 | FAIL | 5.687,7 ms | 2,36 s |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 0,922 | 0,958 | 0,969 | FAIL | 394,4 ms | 0,15 s |

MiniLM ist in dieser Messung rund 14-mal schneller, liegt aber bei der
Ranking-Qualität unter BGE und darf wegen des Lara-Pflichtfalls nicht als Sieger
ausgewählt werden.

## Sprachgruppen

| Modell | Englisch Top-1 | Deutsch Top-1 | Gemischt Top-1 |
| --- | ---: | ---: | ---: |
| BGE Base | 0,958 | 0,958 | 0,938 |
| MiniLM L6 | 0,958 | 0,917 | 0,875 |

MiniLM verfehlt zusätzlich Laras Privathaftpflichtfall in der deutschen
Variante auf Rang 2 und in der gemischten Variante auf Rang 3. Alle Rankings
waren über die drei Messläufe stabil.

## Projekt-Runtime-Verifikation

MiniLM wurde zusätzlich im unveränderten Projekt-Runtime mit
`sentence-transformers 5.1.2` und `transformers 4.57.3` ausgeführt. Die Rankings
aller 64 Fälle waren identisch mit dem ersten erweiterten Lauf. Die Qualität
blieb bei Top-1 0,922, MRR@5 0,958 und nDCG@5 0,969; die Median-Latenz betrug
463,1 ms. Die Entscheidung hängt daher nicht von der isolierten neueren
Benchmark-Laufzeit ab.

## Empfehlung

Vor einem Modellwechsel sollte die Auswahl der aktuellen Police deterministisch
über strukturierte Metadaten erfolgen: Produkt, Kunde, Status und insbesondere
Gültigkeitsbeginn/-ende beziehungsweise „neueste wirksame Police“. Der
Cross-Encoder sollte anschließend nur noch semantisch passende Kandidaten
innerhalb dieser gefilterten Menge sortieren. Danach ist derselbe Regressionstest
erneut auszuführen. Erst wenn MiniLM alle Lara-Pflichtvarianten auf Rang 1 setzt
und die deutsch/gemischte Qualität nicht relevant abfällt, ist die produktive
Backend-Umstellung sinnvoll.

## Artefakte

- Fixture: `tests/fixtures/reranker_regression_cases.json`
- Generator: `scripts/generate_reranker_regression_cases.py`
- Vergleichsbericht: `reports/reranker_regression_benchmark_20260801.md`
- Detaildaten: `reports/reranker_regression_benchmark_20260801_details.csv`
- Projekt-Runtime-Bericht: `reports/reranker_regression_minilm_project_runtime_20260801.md`
- Projekt-Runtime-Details: `reports/reranker_regression_minilm_project_runtime_20260801_details.csv`
