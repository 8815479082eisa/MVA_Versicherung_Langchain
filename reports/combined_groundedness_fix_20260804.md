# Combined CRM+RAG Groundedness Fix, 2026-08-04

## Analyse

### Rekonstruktion des fehlgeschlagenen Laufs

- Request-ID: `1fc84b1acbbf4a8395dee4a22b928056`; Route: `combined`.
- CRM-Subquery: Lara Neumann, aktive Motor-Police und Vertragsfelder. Knowledge-Subquery: Windschutzscheibenschaden, Reparatur statt Austausch und Selbstbehalt.
- Effective Retrieval Query: Knowledge-Subquery plus `BaloiseDirect Motor Vehicle 2026`, `Motor Insurance`, `Partial Coverage`, `glass breakage`, `windscreen damage`, `repaired rather than replaced`, `deductible not applied`, `partner repair service` und `customer service notification`.
- CRM-Kontext: `TEST-KFZ-2026-1003`, BaloiseDirect Motor Vehicle 2026, Motor Insurance, Partial Coverage, Active, 300 EUR Selbstbehalt, 720 EUR Jahresprämie.
- BM25, Rang 1-8: ausschließlich `motor-vehicle-insurance-sti.pdf`, physische Seiten 9, 15, 15, 10, 14, 17, 18 und 20.
- Vector, Rang 1-8: Motor Seite 9; Household Seite 16; Motor Seite 14; Household Seite 13; Motor Seite 14; Household Seite 7; Household Seite 12; Household Seite 11.
- `240_1184_e.pdf` war weder im BM25- noch im Vector-Candidate-Set.
- Merge, Rang 1-8: Motor S. 9 (`8374770a...`), Motor S. 14 (`a0aef006...`), Household S. 16 (`08aaa02a...`), Motor S. 15 (`bf560...`), Motor S. 14 (`530a1be0...`), Motor S. 15 (`956f...`), Household S. 13 (`ef70e403...`), Motor S. 10 (`bc646...`).
- Neighbor Expansion ergänzte Motor S. 15 (`ecad...`), S. 8 (`51d...`) und S. 13 (`f5b...`). Der unmittelbar vorhergehende Same-Page-Chunk mit `G10.2 You will not have to bear a deductible` wurde wegen der bisherigen Bedingung `abs(page difference) == 1` nicht ergänzt.

### Finale PDF-Chunks des alten Laufs

| Rang | Datei / Seite / Chunk | Score | Glasbruch | Waiver vollständig | Produktspezifisch | Notwendig |
|---:|---|---:|---|---|---|---|
| 1 | motor-vehicle-insurance-sti.pdf / 14 / `530a1be0...` | -1.58502746 | teilweise | Nein | Motor, aber Helvetia | Nein |
| 2 | motor-vehicle-insurance-sti.pdf / 14 / `a0aef006...` | -2.26115608 | Ja | Nein | Motor, aber Helvetia | Ja |
| 3 | household-contents-private-liability-sti.pdf / 13 / `ef70e403...` | -2.33587170 | Nein | Nein | Nein | Nein |
| 4 | motor-vehicle-insurance-sti.pdf / 9 / `8374770a...` | -2.52827215 | Ja, Regel g) | Nein, G10.2-Überschrift fehlt | Motor, aber Helvetia | Ja |
| 5 | household-contents-private-liability-sti.pdf / 16 / `08aaa02a...` | -2.75811744 | Nein | Nein | Nein | Nein |

Sanitisierte Bedeutung: Rang 1 behandelt Parkschäden und andere Deckung; Rang 2 nennt Reparatur/Austausch von Fahrzeugglas und Ausschlüsse bei nicht ausgeführter Reparatur beziehungsweise Fahrzeugzeitwert; Rang 3 betrifft gelegentliche Fahrer in der Privathaftpflicht; Rang 4 enthält unter anderem die Regel für eine reparierte statt ersetzte Frontscheibe, aber ohne ihre Waiver-Überschrift; Rang 5 betrifft Selbstbehalt/Bonusverlust bei fremden Fahrzeugen.

Keine der fünf Passagen enthielt als vollständigen, selbständig verständlichen Beleg die Waiver-Regel mit ihrer G10.2-Überschrift. Baloise Customer Service, Baloise-Partner und der Ausschluss von Full-Service-Leasing waren nicht im Kontext. Der Kontext reichte deshalb nicht für eine vollständige Baloise-Antwort.

### Source Drift

`reports/helvetia_reindex_summary_20260801.md` belegt, dass `240_1184_e.pdf` zusammen mit weiteren Baloise-PDFs nach `data/raw/excluded_pdfs` verschoben und der aktive Chroma-Bestand mit Helvetia-Dokumenten neu aufgebaut wurde. `motor-vehicle-insurance-sti.pdf` ist Helvetia, Ausgabe März 2026, und damit nicht dasselbe Produkt/dieselbe Bedingungsversion wie `240_1184_e.pdf`. Das CRM-Produkt blieb BaloiseDirect. Das ist ein bestätigter Product-/Issuer-Mapping- und Source-Drift; kein Laufzeitfilter und kein schlechtes Reranking konnte die aus dem aktiven Index entfernte Baloise-Quelle auswählen.

Im früheren erfolgreichen Lauf waren `240_1184_e.pdf` S. 6/7 im Candidate-Set und finalen Kontext. Frühere Knowledge-Subquery, vollständige Effective Retrieval Query und Metadata-Filter sind aus den vorhandenen Artefakten nicht rekonstruierbar. Belegt sind der frühere BGE-Reranker, die Baloise-Finalchunks und der spätere Wechsel auf MiniLM. Im aktuellen Lauf war die Query Expansion bereits ausreichend; die entscheidende Quellenänderung war der Corpus-Wechsel.

### Antwort und v5-Aggregation

Die rohe Antwort gab Police, Partial Coverage, 300 EUR und 720 EUR korrekt wieder. Sie behauptete jedoch, bei Reparatur werde nur eine Zeitwertdifferenz entschädigt und der individuelle Selbstbehalt könne weiterhin gelten. Danach hängte der Completeness-Guard irrelevante Bedingungen wie `provided that the tank is closed` und eine Privathaftpflicht-Ausnahme an.

| Claim/Zeile | Bester Support | Support | Ergebnis |
|---|---|---:|---|
| Police, Coverage Type, 300 EUR, 720 EUR | CRM-Policy-Chunk | 1.0 je Feld | PASS |
| Glas grundsätzlich gedeckt | Motor S. 14 | hoch | fachlich nur allgemein Helvetia |
| Zeitwertdifferenz bei Reparatur | Motor S. 14 | ca. 0.90 lexikalisch | semantisch falsch zusammengefasst |
| 300 EUR könne bei Reparatur gelten | Motor S. 9/CRM | ca. 0.745 | Waiver-Überschrift fehlte |
| `### Conditions` | kein semantischer Claim | 0.0 | v5-Fehlbewertung |
| `Source:` | kein semantischer Claim | 0.0 | v5-Fehlbewertung |
| Quellen-/Qualifier-Labels | kein semantischer Claim | 0.015-0.073 | v5-Fehlbewertung |
| Kein finaler Claim-Entscheid | Benutzerinstruktion | 0.25 | Präsentations-/Prozesssatz |

v5 meldete `base_v4=0.525375`, `minimum_claim_support=0.0`; die Aggregation `min(base, 0.72*base + 0.28*minimum)` ergab exakt `0.37827`. Es wurden keine Coverage-, Structured-, Citation-, Polarity- oder sonstigen Caps angewendet. Der Scoreverlust entstand aus mehreren schwachen/falsch extrahierten Claims, nicht aus einem einzelnen Cap. Auch der frühere Threshold 0.51 wäre unterschritten worden.

### Ursachenklassifikation

| Kategorie | Bewertung |
|---|---|
| falsches produktspezifisches Dokument / relevante Quelle nicht retrieved | bestätigt; Baloise nicht im aktiven Index |
| relevante Quelle retrieved, aber schlecht rerankt | ausgeschlossen für `240_1184_e.pdf` |
| relevanter Neighbor-Chunk fehlt | bestätigt |
| Final Top-K zu klein | nicht bestätigt; zwei Plätze waren Distraktoren, aber die Quelle fehlte bereits vorher |
| Metadata-/Product-Mapping fehlt; generische Quelle verdrängt passende Quelle | bestätigt |
| Query Expansion unzureichend | ausgeschlossen |
| Generation/Completeness unzureichend | bestätigt |
| Citation-Postprocessing-Problem | nicht bestätigt |
| Groundedness-v5-Präsentationsfehler | bestätigt |
| Coverage-Taxonomiefehler, CRM-Feldfehler, Threshold als Hauptursache | ausgeschlossen |

## Implementierung

- Allgemeine Produktdomänen-Affinität priorisiert passende Source-/Produktmetadaten und wertet widersprechende Versicherungszweige ab; kein fester Dateiname.
- Neighbor Expansion akzeptiert nun kontrolliert unmittelbare Vorgänger/Nachfolger auch auf derselben PDF-Seite und bevorzugt gemeinsam vollständige Reparatur/Waiver-Belege.
- Der Reranker bleibt MiniLM; seine Auswahl berücksichtigt Produkt-Affinität und erhält vollständige Governing-Heading/Regel-Paare.
- Material Qualifiers werden nur ergänzt, wenn lokaler Passagenkontext und Produktdomäne zur Query passen.
- Der deterministische Completeness-Guard korrigiert/ergänzt den Windschutzscheiben-Waiver ausschließlich, wenn Waiver-Überschrift und Reparatur-statt-Austausch-Regel gemeinsam im finalen Kontext liegen.
- v5 ignoriert Markdown-Überschriften, reine Quellenlabels, Citations und den geforderten Nicht-Entscheidungs-Disclaimer; `will not have to bear a deductible` wird als `deductible:none` normalisiert.
- Kein Hardcoding von Lara, Police, Query, Source oder Chunk-ID: Ja, keines vorhanden.
- PDFs, Chroma, CRM, Threshold, Reranker, Self-Check und Auth-Konfiguration wurden nicht geändert.

## Tests

`29 passed / 0 failed`: `tests/unit/test_retrieval_service.py` und `tests/unit/test_experimental_groundedness_v5.py`. Abgedeckt sind Produktpriorität, zweites Produktfixture, Same-Page-Waiver-Neighbor, generischer Distraktor, vollständige/fehlende Waiver-Evidenz, Qualifier-Relevanz, Coverage-Konsistenz sowie v5 PASS/FAIL bei korrekter/falscher Waiver-Polarität.

## Live-Verifikation

- Combined Request ausgeführt: Ja; Anzahl Live-E2E-Requests: exakt 1.
- Request-ID: `2ef2b2181e074ff692ac1bca3e39692e`; HTTP: `503`; geplante Route: `combined`.
- Runtime-Beleg: `CRM integration is disabled. Set CRM_ENABLED=true to enable it.` / `CRM_UNAVAILABLE`; Serverlog: `CRM MCP not started: disabled`.
- Ausgewählte Police/PDFs/Reranker-Ergebnisse: keine, da der Lauf vor CRM-Lesen und Retrieval abbrach.
- Answer Generation: Nein. Groundedness v5: nicht ausgeführt. Score/Threshold-Ergebnis: nicht verfügbar. Citations: keine.
- Finale API-Antwort: sicherer Fehler `CRM_UNAVAILABLE`.
- Final result: FAIL (Runtime-Konfiguration blockiert Live-Verifikation vor dem geänderten Pfad).

`SELF_CHECK_ENABLED=false` und `PIPELINE_AUTHENTICATED=false` blieben unverändert. Ein zweiter Request wurde wegen der Vorgabe exakt eines Live-Requests nicht gesendet.

## Schlussbewertung

1. `0.37827` entstand durch fehlende/inkonsistente Evidenz, falsche Antwortzusammenfassung, irrelevante Completeness-Zusätze und v5-Präsentationszeilen mit Support 0; keine Caps.
2. Primär waren Corpus/Product-Mapping und Same-Page-Retrieval betroffen; Generation/Completeness und v5 verstärkten den Fehler.
3. Implementiert wurden allgemeine Produktpriorisierung, kontrollierte Same-Page-Neighbor-Erweiterung, evidenzgebundene Waiver-Vervollständigung und semantische Claim-Extraktion.
4. Die Auswahl ist durch Unit-Tests belegt; im Live-Lauf konnte sie wegen `CRM_UNAVAILABLE` nicht erreicht werden.
5. Ein Live-v5-PASS bei 0.7888 ist daher nicht belegt.
6. Für einen erneuten Live-Nachweis muss die bereits vorgesehene read-only CRM-Runtime aktiviert sein. Für eine fachlich echte Baloise-Antwort müssen CRM-Produkt und aktiver PDF-Corpus außerdem denselben Issuer/dieselbe Bedingungsversion abbilden; das war wegen des Verbots von CRM-, PDF- und Reindex-Änderungen nicht Teil dieses Laufs.
