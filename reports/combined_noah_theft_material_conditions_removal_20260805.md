# Combined-CRM+RAG-Test nach Entfernung von `Material documented conditions`

Datum: 2026-08-05  
Request-ID: `9f7b2c37e6264f3d98fcfa0628dcab69`

## Ergebnis

**Finaler Status: FAIL**

Das automatische Anhängen des separaten Abschnitts `Material documented conditions` wurde vollständig entfernt. Im einzigen Live-Test erschienen weder dieser Abschnitt noch die sachfremde Klausel `unless caused by a fire`. HTTP, Route, Policy-Selektion, CRM-Werte, Groundedness, Citations und Safety waren erfolgreich.

Der Test ist dennoch FAIL, weil die OpenAI-Generation zwei notwendige, im ausgewählten K5.4-Chunk belegte Theft-Bedingungen ausließ: die unverzügliche Meldung an die Polizei und die unverzügliche Information an den Versicherer bei Wiederauffinden beziehungsweise bekanntem Aufenthaltsort. Der Familienausschluss wurde korrekt genannt.

## Geänderte Dateien und Funktionen

### `src/api/rag_service.py`

- `_ensure_material_qualifiers` entfernt; damit gibt es keinen Postprozessor mehr, der Zusatzbedingungen an Antworten anhängt.
- `_material_qualifier_extract`, `_material_qualifier_notice` und die zugehörigen Qualifier-Patterns/Event-Hilfen entfernt.
- `_format_context_with_sources` gibt nur noch Source-Label und originalen Chunk-Inhalt an die Generation weiter; es erzeugt keine `MATERIAL QUALIFIERS PRESENT`-Marker mehr.
- `build_generation_chain` verlangt weiterhin direkt szenariorelevante, belegte Bedingungen in der normalen Antwortstruktur, verbietet aber separate automatisch extrahierte Bedingungsabschnitte.
- `generate_answer` ruft weder im normalen noch im Empty-Answer-Fallback einen Material-Qualifier-Postprozessor auf.
- Citation-, Windscreen-, Groundedness- und Safety-Verarbeitung blieben erhalten.

### `tests/unit/test_retrieval_service.py`

- Regressionstest: Generierungskontext enthält keine Qualifier-Marker.
- Regressionstest: Eine belegte Theft-Antwort mit Police Reporting, Family Exclusion, Wiederauffinden sowie PDF-/CRM-Citations wird unverändert weitergegeben.
- Veraltete Tests für den entfernten automatischen Anhang wurden entfernt.

## Regressionstests

| Testgruppe | Ergebnis |
|---|---:|
| Generierung und Retrieval | 21 passed |
| Routing, Groundedness, Citation/Coverage und Safety | 55 passed |
| Gesamt | **76 passed** |

Es blieben nur bekannte Deprecation-Warnungen aus NeMo Guardrails, Pydantic und LangChain Community. Syntaxprüfung und `git diff --check` waren erfolgreich.

Vor dem Live-Test wurden zusätzlich bestätigt:

- `CRM_ENABLED=true`
- `SELF_CHECK_ENABLED=false`
- `PIPELINE_AUTHENTICATED=false`
- Groundedness-Threshold `0.7888` aus der Kalibrierungsdatei
- Safety `enabled`, Modus `enforce`
- ChromaDB unverändert mit 1.585 Chunks
- `pdfs_have_changed=False`
- `embedding_model_has_changed=False`

## Backend-Neustart

Der Container `mva-backend` wurde vollständig mit `docker restart` neu gestartet. Vor dem Live-Request waren `pipelineReady`, `crmReady`, `embeddingReady` und `retrievalReady` jeweils `true`; `pipelineInitError` war `null`.

## Verwendete Query

```text
Noah Weber reports the theft of his car. Is this loss generally covered by his current motor insurance, and which documented conditions apply?

Please provide the relevant active policy number, coverage type, individual deductible and annual premium. Clearly distinguish the general insurance terms from Noah's individual contract data, and do not make a final claim decision.
```

Es wurde genau ein `POST /api/ask` gesendet. Es gab keinen Request-Retry; die Answer Generation hatte genau einen OpenAI-Aufruf.

## HTTP, Route und Police

| Feld | Ergebnis |
|---|---|
| HTTP-Status | 200 |
| API-Status | `complete` |
| Route | `combined` |
| ausgewählte Police | `TEST-KFZ-2026-1701` |
| Policy-Selection-Score | 460 |
| Safety-Fallback | nein |

## CRM-Fakten

| Feld | Wert |
|---|---|
| Kunde | Noah Weber |
| Police | `TEST-KFZ-2026-1701` |
| Produkt | Helvetia Motor Vehicle Insurance |
| Coverage Type | Partially comprehensive cover |
| Status | Active |
| Laufzeit | 2026-05-01 bis 2027-04-30 |
| individueller Selbstbehalt | 300 EUR |
| Jahresprämie | 735 EUR |

Die CRM-E-Mail-Adresse wurde vor der Generation durch die Safety-Kontextprüfung redigiert.

## Finale PDF-Chunks

| Rang | Datei | Seite | Chunk-ID | Reranker-Score | Inhalt |
|---:|---|---:|---|---:|---|
| 1 | `motor-vehicle-insurance-sti.pdf` | 13 | `dfa04e4f-b781-40d9-baa2-eac6f7a2ed58` | 2.59433007 | Theft-Deckung, involuntary damage, Family Exclusion |
| 2 | `motor-vehicle-insurance-sti.pdf` | 13 | `f78066a2-fb42-404e-b1bb-1bd7ee8b8312` | -1.96668172 | überlappender Vorgängerchunk; überwiegend Fire/Natural Forces, endet mit Beginn K2.1.4 Theft |
| 3 | `motor-vehicle-insurance-sti.pdf` | 17 | `768239c2-f3ef-4806-9b99-41ed105144d3` | 2.17699337 | Police Reporting, Auslandsregel und Information bei Wiederauffinden |
| 4 | `motor-vehicle-insurance-product-sheet.pdf` | 3 | `f69012fa-1c36-4fbb-9e71-f92bb1188b11` | 4.47695160 | Theft als Leistung der partially comprehensive insurance |
| 5 | `motor-vehicle-insurance-sti.pdf` | 16 | `7c0096a6-e783-4c1a-976e-0360b31d8ed3` | 0.17321973 | allgemeine Betriebsschadenausschlüsse; für die Query nur tangential |

## Vollständige finale Antwort

```text
Yes, the loss due to theft is generally covered by Noah Weber's current motor insurance under the partially comprehensive coverage. The insurance covers loss, disappearance, destruction, or damage caused by theft, misappropriation, or robbery of the insured vehicle, provided the damage occurred involuntarily. However, no compensation is paid if the act was committed by family members.

Here are the details of Noah Weber's policy:

- **Policy Number**: TEST-KFZ-2026-1701
- **Coverage Type**: Partially comprehensive cover
- **Individual Deductible**: 300 EUR
- **Annual Premium**: 735 EUR

**General Insurance Terms**:
- Coverage includes loss due to theft, but excludes compensation if the theft was committed by family members.

This information is general and does not constitute a final claim decision.

Source:
- [motor-vehicle-insurance-sti.pdf, page 13]
- [CRM: TEST-KFZ-2026-1701]
```

`rawOpenAIAnswer`, `afterCompletenessPostprocessing`, `afterCitationPostprocessing` und die finale Antwort waren inhaltlich identisch. Damit ist technisch belegt, dass kein automatischer Zusatzabschnitt mehr erzeugt wurde und dass die fehlenden K5.4-Bedingungen bereits durch die Modellgeneration ausgelassen wurden.

## Bedingungen und Citations

| Kriterium | Ergebnis |
|---|---|
| Theft-Deckung | enthalten |
| involuntary damage | enthalten |
| Family Exclusion | enthalten |
| Police Reporting | **fehlt** |
| Information bei Wiederauffinden | **fehlt** |
| Auslands-Reporting | fehlt |
| `unless caused by a fire` | nicht enthalten |
| `Material documented conditions` | nicht enthalten |

Finale Inline-Citations:

- PDF: `[motor-vehicle-insurance-sti.pdf, page 13]`
- CRM: `[CRM: TEST-KFZ-2026-1701]`

Der API-Quellensatz enthielt zusätzlich die ausgewählten PDF-Quellen auf Seite 17 und Seite 3. Da deren Bedingungen in der Antwort nicht verwendet wurden, wurden sie vom Modell nicht inline zitiert.

## Groundedness und Qualitätsmetriken

| Metrik | Ergebnis | Bewertung |
|---|---:|---|
| Groundedness-Algorithmus | `fact_aware_claim_support_v5` | ausgeführt |
| Groundedness-Score | 0.856132 | PASS |
| Threshold | 0.7888 | unverändert |
| Groundedness v4 Basis | 0.767484 | Diagnosewert |
| Minimum Claim Support | 0.830000 | keine unbelegten Claims |
| Average Claim Support | 0.900246 | keine Caps |
| Context Precision | 0.8000 | 4 von 5 finalen Chunks direkt oder überlappend relevant |
| Citation Presence | 1.0000 | PDF- und CRM-Citation vorhanden |
| Citation Support | 1.0000 | keine Citation-Mismatches für die tatsächlich genannten Claims |
| Coverage Consistency | PASS | keine Coverage-, Policy-, Zahlen- oder Polaritäts-Mismatches |

Groundedness bewertet die Quellenstützung vorhandener Claims, nicht die Vollständigkeit ausgelassener Bedingungen. Deshalb konnte der Groundedness-Check die fehlenden K5.4-Pflichten nicht erkennen.

## Safety

| Phase | Ergebnis |
|---|---|
| Pre-Action | allow |
| Context-Action | redact |
| Redigierter Inhalt | CRM-E-Mail-Adresse |
| Post-Action | allow |
| Safety-Fallback | nein |

## Latenzen

| Phase | Latenz |
|---|---:|
| CRM | 1,241.142 ms |
| Retrieval gesamt | 6,026.292 ms |
| Reranking | 19,729.775 ms |
| OpenAI Answer Generation | 10,460.867 ms |
| Groundedness | 3,464.126 ms |
| Guardrails gesamt | 3,695.194 ms |
| Backend E2E | 46,814.671 ms |
| Client Wall-Time | 47,377.560 ms |

## Exakte FAIL-Ursache

Der ausgewählte Chunk `768239c2-f3ef-4806-9b99-41ed105144d3` auf PDF-Seite 17 enthielt vollständig die unverzügliche Polizeimeldung und die Informationspflicht bei Wiederauffinden. Die OpenAI-Generation verwendete diesen Chunk trotz Query nach den dokumentierten Bedingungen nicht. Da das automatische Anhängen absichtlich entfernt wurde und Groundedness nur vorhandene Aussagen auf Quellenstützung prüft, blieb die Antwort fachlich unvollständig. Es gab keinen Retrieval-, Citation-, Safety-, Threshold- oder Coverage-Consistency-Fehler.

