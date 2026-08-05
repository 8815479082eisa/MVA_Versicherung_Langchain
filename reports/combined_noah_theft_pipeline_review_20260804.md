# Combined-CRM+RAG-Pipelineprüfung und Live-Einzeltest

Datum: 2026-08-04  
Request-ID: `2199a847d64c4fc98eab165e97a35e97`

## Ergebnis

**Finaler Status: FAIL**

Der reale Request erreichte HTTP 200, Route `combined`, bestand Groundedness und alle Safety-Prüfungen und lieferte die richtigen CRM-Vertragswerte sowie die wesentlichen PDF-Regeln. Der strenge fachliche Gesamtstatus ist dennoch FAIL, weil der Completeness-Postprozessor an die korrekte OpenAI-Antwort die sachfremde Feuerklausel `unless caused by a fire` anhängte. Dieser konkrete Fehler wurde nach dem einzigen erlaubten Live-Request korrigiert und durch Unit-Tests verifiziert. Ein zweiter Live-Request wurde nicht gesendet.

## Testperson und Schadenfall

Ausgewählt wurde **Noah Weber**, eine von Lara Neumann verschiedene, real im EspoCRM vorhandene Testperson.

Begründung:

- aktive Kfz-Police `TEST-KFZ-2026-1701`
- Produktdomäne Motor und Coverage Type `Partially comprehensive cover`
- realer CRM-Testschaden `TEST-CLM-2026-2501`, Schadenart `Theft`, Status `Under Review`
- die individuellen Werte Police, Coverage Type, Selbstbehalt und Prämie stammen aus dem CRM
- Deckungstatbestand, Ausschluss und Meldepflichten stammen aus den aktiven Kfz-PDFs
- deshalb ist die Route `combined` fachlich erforderlich

## Verwendete Query

```text
Noah Weber reports the theft of his car. Is this loss generally covered by his current motor insurance, and which documented conditions apply?

Please provide the relevant active policy number, coverage type, individual deductible and annual premium. Clearly distinguish the general insurance terms from Noah's individual contract data, and do not make a final claim decision.
```

## Laufzeitkonfiguration

| Einstellung | Tatsächlicher Wert |
|---|---:|
| CRM | enabled, EspoCRM live read-only |
| Self-Check | disabled |
| Pipeline authenticated | false |
| Answer Provider / Modell | OpenAI / `gpt-4o-mini` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Retrieval / Rerank Top-K | 8 / 5 |
| Groundedness | `fact_aware_claim_support_v5` |
| Threshold | `0.7888`, Quelle `calibration_file` |
| Safety | enabled, `enforce`, Backend `nemo` |

Die Konfiguration und der Threshold wurden nicht geändert. CRM, PDFs und ChromaDB wurden nicht verändert oder reindexiert. Die bestehende Collection enthielt nach dem Test weiterhin 1.585 Chunks; `pdfs_have_changed=False` und `embedding_model_has_changed=False`.

## Vorgenommene Änderungen

1. Deaktivierte Hilfskomponenten werden beim Pipeline-Start nicht mehr unnötig instanziiert: Compressor, Query-Rewrite, Self-Check und der nicht verwendete LLM-Router bleiben bei deaktivierter Konfiguration aus.
2. Die Combined-Retrieval-Query wird für Kfz-Ereignisse kompakt und ereignisspezifisch aufgebaut. Für den Test wurde effektiv `motor vehicle insurance Partially comprehensive cover vehicle theft loss disappearance destruction insured vehicle police without delay` verwendet.
3. Eindeutig produktfremde Kandidaten werden vor dem Cross-Encoder entfernt, sofern genügend produktkompatible Evidenz verbleibt. Im Live-Lauf wurde dadurch ein Household-Contents-Chunk entfernt.
4. Die fachliche Evidenzpriorisierung berücksichtigt Diebstahldeckung, Familienausschluss und polizeiliche Meldepflichten und verdrängt sachfremde Kfz-Nebenprodukte.
5. Groundedness ignoriert reine Überschriften, Quellenzeilen und den vorgeschriebenen Entscheidungs-Disclaimer bereits vor der semantischen Bewertung. Coverage-Aliase für `partial`/`partially comprehensive` wurden vereinheitlicht.
6. Material-Qualifier werden nur für das abgefragte Schadenereignis erzwungen. Nach dem Live-Befund wurde zusätzlich korrigiert, dass ein später im selben überlappenden Chunk genannter Diebstahlabschnitt eine vorherige Feuerklausel nicht mehr fälschlich relevant macht.
7. Notwendige Safety-Prüfungen für Prompt Injection, Secrets und PII blieben aktiv. Im Test wurde lediglich die E-Mail-Adresse aus dem CRM-Kontext redigiert.

## Ausgewählte Police und CRM-Fakten

| Feld | Wert |
|---|---|
| Kunde | Noah Weber |
| Police | `TEST-KFZ-2026-1701` |
| Produkt | Helvetia Motor Vehicle Insurance |
| Coverage Type | Partially comprehensive cover |
| Status | Active |
| Laufzeit | 2026-05-01 bis 2027-04-30 |
| Individueller Selbstbehalt | 300 EUR |
| Jahresprämie | 735 EUR |

Die Policy-Selektion bewertete die Kfz-Police mit Score 460 und wählte sie eindeutig vor der ebenfalls vorhandenen, aber fachfremden Rechtsschutzpolice aus.

## Finale PDF-Chunks

| Rang | Datei | Seite | Chunk-ID | Reranker-Score | Fachliche Einordnung |
|---:|---|---:|---|---:|---|
| 1 | `motor-vehicle-insurance-sti.pdf` | 13 | `dfa04e4f-b781-40d9-baa2-eac6f7a2ed58` | 2.59433007 | Diebstahldeckung, versichertes Fahrzeug/Zubehör, Versuch, Familienausschluss |
| 2 | `motor-vehicle-insurance-sti.pdf` | 13 | `f78066a2-fb42-404e-b1bb-1bd7ee8b8312` | -1.96668172 | überlappender Vorgängerchunk; endet mit Beginn von K2.1.4 Theft, enthält überwiegend Feuer/Naturgefahren |
| 3 | `motor-vehicle-insurance-sti.pdf` | 17 | `768239c2-f3ef-4806-9b99-41ed105144d3` | 2.17699337 | unverzügliche Polizeimeldung, Auslandsfall, Information bei Wiederauffinden |
| 4 | `motor-vehicle-insurance-product-sheet.pdf` | 3 | `f69012fa-1c36-4fbb-9e71-f92bb1188b11` | 4.47695160 | Theft als Leistung der partially comprehensive insurance |
| 5 | `motor-vehicle-insurance-sti.pdf` | 16 | `7c0096a6-e783-4c1a-976e-0360b31d8ed3` | 0.17321973 | allgemeine Betriebsschadenausschlüsse; für die konkrete Antwort nur tangential |

## Tatsächlich ausgelieferte Antwort

```text
Noah Weber's current motor insurance covers loss due to theft, as stated in the insurance documents. The relevant conditions are as follows:

### Coverage Information
- **Policy Number:** TEST-KFZ-2026-1701
- **Coverage Type:** Partially comprehensive cover
- **Individual Deductible:** 300 EUR
- **Annual Premium:** 735 EUR

### General Insurance Terms
- The insurance covers loss, disappearance, destruction, or damage caused by theft, misappropriation, or robbery of the insured vehicle and accessories, or the attempt to carry out such acts if the damage has occurred involuntarily.
- No compensation is paid if the act was committed by family members.

### Conditions
- The theft must be reported to the responsible police without delay.
- If the theft occurs abroad, it must also be reported to the police station at Noah's Swiss place of residence.
- If the vehicle is found or if anything is learned regarding its whereabouts, Noah must inform the insurance provider without delay.

This information is general and does not constitute a final claim decision.

**Source:**
- [motor-vehicle-insurance-sti.pdf, page 13]
- [motor-vehicle-insurance-sti.pdf, page 17]
- [CRM: TEST-KFZ-2026-1701]

Material documented conditions [motor-vehicle-insurance-sti.pdf, page 13]:
- unless caused by a fire.
```

Die `rawOpenAIAnswer` endete korrekt nach der CRM-Citation. Erst `afterCompletenessPostprocessing` ergänzte die letzten beiden Zeilen.

## Citations und Konsistenz

PDF-Citations in der Antwort:

- `motor-vehicle-insurance-sti.pdf`, Seite 13
- `motor-vehicle-insurance-sti.pdf`, Seite 17

Zusätzliche PDF-Quelle im API-Quellensatz:

- `motor-vehicle-insurance-product-sheet.pdf`, Seite 3

CRM-Citation:

- `[CRM: TEST-KFZ-2026-1701]`, EspoCRM Policy

Coverage Consistency: **PASS**. Es gab keine Coverage-Type-, Policy-Number-, Zahlen-, Polaritäts- oder strukturierten Mismatches. Die individuelle Coverage `Partially comprehensive cover` stimmt mit der in den PDFs dokumentierten Produktdomäne für Theft überein.

## Metriken

| Metrik | Ergebnis | Einordnung |
|---|---:|---|
| Groundedness v5 | 0.862544 | PASS gegen 0.7888 |
| Groundedness v4 Basis | 0.771507 | Diagnosewert, nicht der enforce-Wert |
| Minimum Claim Support | 0.800000 | kein Claim ohne positive Unterstützung |
| Average Claim Support | 0.924077 | keine Caps oder Mismatches |
| Chunk Precision | 0.8000 | 4 von 5 finalen Chunks enthalten direkt oder überlappend relevante Evidenz |
| Chunk Recall | 1.0000 | alle drei vorab definierten Kernevidenzen wurden gefunden: Produktleistung, K2.1.4, K5.4 |
| Chunk F1 | 0.8889 | harmonisches Mittel aus Precision und Recall |
| Context Precision | 0.8000 | Einzeltest-Adjudikation, kein Korpusbenchmark |
| Citation Presence | 1.0000 | PDF- und CRM-Citations vorhanden |
| Citation Support | 1.0000 | keine Citation-Mismatches; die spätere Feuerklausel war belegt, aber für die Query irrelevant |

Precision, Recall, F1 und Context Precision sind für diesen Einzeltest deterministisch auf Chunk-Ebene bewertet. Sie sind keine statistisch belastbare Gesamtbewertung der Pipeline.

## Safety

| Phase | Ergebnis |
|---|---|
| Pre-Action | allow |
| Context-Action | redact |
| Redigierter Inhalt | CRM-E-Mail-Adresse |
| Post-Action | allow |
| Safety-Fallback | nein |

Die PII-Redaktion war erforderlich und blockierte die fachlich zulässigen Vertragsdaten nicht.

## Status und Latenzen

| Messpunkt | Ergebnis |
|---|---:|
| HTTP | 200 |
| Route | `combined` |
| CRM | 235.307 ms |
| Retrieval gesamt | 4,249.863 ms |
| Reranking | 13,011.287 ms |
| OpenAI Answer Generation | 10,591.550 ms |
| Groundedness | 5,595.273 ms |
| Guardrails | 5,801.664 ms |
| Backend E2E | 50,164.337 ms |
| gemessene Client-Wall-Time | 65,376.260 ms |

Der Reranker ist in diesem Einzeltest der langsamste einzelne fachliche Schritt. Aus einem einzigen Lauf lässt sich keine belastbare Latenzregression ableiten; seine Entfernung wäre wegen des messbaren Retrieval-Nutzens nicht begründet.

## Exakte Fehlerursache

`_ensure_material_qualifiers` untersuchte ein lokales Fenster um die erkannte Klausel. Im überlappenden Chunk `f78066a2-...` stand zuerst die Feuerklausel `unless caused by a fire` und später der Beginn des Diebstahlabschnitts K2.1.4. Der spätere Begriff `Theft` ließ das lokale Fenster zur Theft-Query passend erscheinen, obwohl die eigentliche Klausel zum Ereignis `fire` gehörte. Groundedness blockierte dies nicht, weil die Klausel im zitierten Chunk tatsächlich belegt war; betroffen war Relevanz, nicht Quellenstützung.

Nach dem Live-Lauf verwendet der Guard primär die Ereignisterme des eigentlichen Qualifier-Extrakts und nur bei ereignisneutralen Bedingungen den lokalen Kontext. Der exakte Regressionstest sowie insgesamt 81 fokussierte Tests bestehen. Gemäß Vorgabe wurde diese Nachkorrektur nicht durch einen zweiten Live-Request geprüft.

## Geänderte Dateien

- `src/api/rag_service.py`
- `src/core/insurance_tool_routing.py`
- `src/guardrails/integrations/nemo_actions.py`
- `scripts/experimental_groundedness_v5.py`
- `tests/unit/test_retrieval_service.py`
- `tests/unit/test_insurance_tool_routing.py`
- `tests/unit/test_experimental_groundedness_v5.py`

