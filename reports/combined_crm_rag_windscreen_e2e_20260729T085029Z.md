# Reale Combined-CRM+RAG-E2E-Prüfung: Lara Neumann / Windschutzscheibe

Ausführungszeitraum: 29.07.2026, 08:35–08:36 UTC  
Produktiver Endpoint: `POST http://localhost:8000/api/ask`  
Live-Läufe: 3  
Gesamtergebnis: **FAIL – kein Full PASS**

## Kurzfazit

Alle drei produktiven Aufrufe wurden technisch erfolgreich mit HTTP `200`,
API-Status `complete`, Route `combined`, OpenAI `gpt-4o-mini`,
`SELF_CHECK_ENABLED=false` und aktivem Safety-/Groundedness-Pfad ausgeführt.
Das Ergebnis ist dennoch in allen drei Läufen semantisch falsch:

1. Der CRM-Kontext enthielt neben der erwarteten Police
   `TEST-KFZ-2026-1003` auch die ältere Police `TEST-KFZ-2026-1001` und einen
   mit ihr verknüpften Glasschaden.
2. Das Modell wählte reproduzierbar die alte Police und gab deshalb
   `150 EUR` Selbstbehalt und `684 EUR` Jahresprämie statt `300 EUR` und
   `720 EUR` aus.
3. Der richtige Baloise-Chunk zur Reparatur der Windschutzscheibe,
   zum Wegfall des Selbstbehalts und zu den Partnerbedingungen war zwar bereits
   in Chroma vorhanden, gelangte aber in keinem Lauf in den Generation Context.
4. Alle Antworten enthielten nur den Platzhalter `[Doc-ID:page]`, also keine
   gültige PDF-Citation.

Die Groundedness-Prüfung ließ alle drei Antworten passieren
(`0,636553`, `0,569547`, `0,634433` bei Threshold `0,51`). Das zeigt hier eine
wichtige Grenze der Metrik: Eine Antwort kann gegenüber einem fehlerhaft
zusammengestellten Kontext hinreichend „grounded“ sein und trotzdem die falsche
Police auswählen und die fachliche Ground Truth verfehlen.

## Exakt verwendete Benutzerfrage

```text
Lara Neumann reports damage to the windscreen of her car. Is this damage
generally covered by her current motor insurance, and would she still have to
pay her individual deductible if the windscreen can be repaired instead of
replaced?

Please provide the relevant active policy number, coverage type, individual
deductible and annual premium. Clearly distinguish the general insurance terms
from Lara's individual contract data, and do not make a final claim decision.
```

Die Frage wurde in allen drei Läufen exakt so übertragen. Es wurden kein
Dateiname, kein Dokumenttitel, keine Seite, kein Retrieval-Hinweis und kein
Source-Filter ergänzt.

## Testbedingungen und produktive Konfiguration

| Eigenschaft | Tatsächlicher Wert |
|---|---|
| Pipeline | bereit |
| CRM | aktiviert und bereit |
| Answer Provider | OpenAI |
| Modell | `gpt-4o-mini` |
| Answer Model Ready | `true` |
| Self Check | `false`, Skip-Grund `disabled_by_configuration` |
| Context Compression | `false` |
| Force Retrieval | `true` |
| Retrieval `top_k` | 8 |
| BM25 `k` | 5 |
| Vector `k` | 5 |
| Reranker `top_k` | 5 |
| Safety | aktiv, Modus `enforce`, Backend `nemo` |
| Groundedness Threshold | `0,51` |
| Threshold-Quelle | `config/groundedness_calibration.json` |
| Mocks | keine |
| Reindex/PDF-Verarbeitung | keine |
| CRM-Zugriff | ausschließlich read-only |

Der Health-Endpoint meldete zusätzlich `llmReady=false`, weil die separate
Ollama-Readiness-Prüfung für das Guardrail-Modell keine Verbindung erhielt.
Das produktive OpenAI-Antwortmodell war jedoch mit `answerModelReady=true`
bereit. Answer Generation und die drei NeMo-Safety-Stufen wurden in allen
Läufen real abgeschlossen.

## 1–3. Route, Decomposition und CRM-Auswahl

### Tatsächlich gewählte Route

```text
combined
```

### Erkannte CRM-Subquery

```json
{
  "customer_name": "Lara Neumann",
  "needs_policies": true,
  "needs_claims": true,
  "policy_number": null,
  "claim_number": null
}
```

`needs_claims=true` ist für diese Frage unerwartet. Die Formulierung
„do not make a final claim decision“ enthält das Schlüsselwort `claim`. Dadurch
wurde der vorhandene alte Glasschaden abgerufen und später in den
Generation Context aufgenommen.

### Erkannte Knowledge-Subquery

```text
reports damage to the windscreen of her car. Is this damage generally covered by her current motor insurance would she still have to pay her individual deductible if the windscreen can be repaired instead of replaced? Please provide the relevant active policy number, coverage type, individual deductible and annual premium. Clearly distinguish the general insurance terms from Lara's individual contract data
```

`requested_pdf_filename` war `null`; die Suche lief damit ohne
Dokumentfilter über den gesamten bestehenden Chroma-Bestand.

### CRM-Kontakt

| Feld | Wert |
|---|---|
| Kontakt-ID | `6a65e0b6babe7371d` |
| Name | Lara Neumann |
| E-Mail-Hinweis | `l***@example.test` |

### Abgerufene Policen

| Police | Produkt / Deckung | Status | Laufzeit | Selbstbehalt | Jahresprämie | Generation Context |
|---|---|---|---|---:|---:|---|
| `TEST-KFZ-2026-1003` | Motor / Partial Coverage | Active | 01.07.2026–30.06.2027 | 300 EUR | 720 EUR | **ja** |
| `TEST-PHV-2026-1002` | Personal Liability / Liability | Active | 01.01.–31.12.2026 | 0 EUR | 96 EUR | nein |
| `TEST-KFZ-2026-1001` | Motor / Partial Coverage | Active | 01.01.–31.12.2026 | 150 EUR | 684 EUR | **ja, fehlerhaft** |

Zusätzlich wurde `TEST-CLM-2026-2001` abgerufen: Glass Damage, Under Review,
780 EUR, Referenz `Lara Neumann Motor 2026`. Diese Referenz zeigt auf die
ältere Motorpolice und verstärkte die falsche Auswahl.

Es fand somit keine exklusive Auswahl einer einzigen relevanten Police statt.
Der Haftpflichtvertrag wurde korrekt vor dem Generation Context entfernt; die
alte Motorpolice nicht.

## 4–7. Retrieval-Evidence

Das Produktions-Audit speichert die acht zusammengeführten Kandidaten, aber
nicht die getrennten BM25- und Vector-Listen. Um diese geforderte Evidenz zu
erhalten, wurden die beiden Listen nach den Live-Aufrufen rein lesend mit
demselben Knowledge-Subquery, derselben Collection und denselben `k`-Werten
rekonstruiert. Dabei wurde ausdrücklich
`build_vectorstore(..., allow_reindex=false)` verwendet. Diese Diagnose hat
keine Chunks für die Antwort ausgewählt und keinen Generation Context
verändert.

### BM25-Kandidaten

| Rang | Datei | Metadata-Seite | Physische Seite | Startindex | Bewertung |
|---:|---|---:|---:|---:|---|
| 1 | `englisch_vvg.pdf` | 58 | 59 | 804 | irrelevanter Text zur Vertragsfortsetzung |
| 2 | `englisch_vvg.pdf` | 35 | 36 | 746 | irrelevantes Sachverständigenverfahren |
| 3 | `englisch_vvg.pdf` | 20 | 21 | 1628 | irrelevante Maklerberatung |
| 4 | `englisch_vvg.pdf` | 60 | 61 | 887 | irrelevanter Einwilligungstext |
| 5 | `englisch_vvg.pdf` | 16 | 17 | 1628 | irrelevante Versicherungsnehmerrechte |

### Vector-Kandidaten

| Rang | Datei | Metadata-Seite | Physische Seite | Startindex | Bewertung |
|---:|---|---:|---:|---:|---|
| 1 | `Insurance_Handbook_20103.pdf` | 10 | 11 | 799 | Windschutzscheibenbruch unter Comprehensive |
| 2 | `Insurance_Handbook_20103.pdf` | 10 | 11 | 0 | Übergang Collision/Comprehensive |
| 3 | `consumer-auto-shopping-tool.pdf` | 10 | 11 | 769 | Fragen zu Glass Shop und Selbstbehalt, keine Regel |
| 4 | `publication-aut-pp-consumer-auto.pdf` | 11 | 12 | 1587 | allgemeine Einkaufsfragen |
| 5 | `publication-aut-pp-consumer-auto.pdf` | 7 | 8 | 791 | Glasschaden plus Hinweis, dass manche Anbieter bei Reparatur keinen Selbstbehalt berechnen |

### Merge und Deduplizierung

Der produktive Hybrid-Merge hängt zuerst alle BM25- und anschließend alle
Vector-Kandidaten an und stoppt bei `top_k=8`.

| Merge-Rang | Herkunft | Datei | Metadata-Seite | Physische Seite | Startindex |
|---:|---|---|---:|---:|---:|
| 1 | BM25 | `englisch_vvg.pdf` | 58 | 59 | 804 |
| 2 | BM25 | `englisch_vvg.pdf` | 35 | 36 | 746 |
| 3 | BM25 | `englisch_vvg.pdf` | 20 | 21 | 1628 |
| 4 | BM25 | `englisch_vvg.pdf` | 60 | 61 | 887 |
| 5 | BM25 | `englisch_vvg.pdf` | 16 | 17 | 1628 |
| 6 | Vector | `Insurance_Handbook_20103.pdf` | 10 | 11 | 799 |
| 7 | Vector | `Insurance_Handbook_20103.pdf` | 10 | 11 | 0 |
| 8 | Vector | `consumer-auto-shopping-tool.pdf` | 10 | 11 | 769 |

Die Vector-Ränge 4 und 5 wurden wegen der festen Grenze bereits **vor dem
Reranking verworfen**. Besonders relevant ist Vector-Rang 5, weil er zumindest
eine allgemeine Reparatur-/Selbstbehalt-Aussage enthielt.

### Finale rerankte PDF-Chunks

Diese fünf Chunks waren in allen drei Läufen identisch:

| Finalrang | Datei | Metadata-Seite | PDF-Label | Physische PDF-Seite | Startindex | Fachlicher Inhalt |
|---:|---|---:|---:|---:|---:|---|
| 1 | `Insurance_Handbook_20103.pdf` | 10 | 4 | 11 | 799 | Allgemeine Comprehensive-Deckung; cracked/shattered windshield |
| 2 | `Insurance_Handbook_20103.pdf` | 10 | 4 | 11 | 0 | Collision und Beginn Comprehensive |
| 3 | `consumer-auto-shopping-tool.pdf` | 10 | 9 | 11 | Fragen zu Werkstatt/Glas und Selbstbehalt |
| 4 | `englisch_vvg.pdf` | 58 | 59 | 59 | irrelevante Vertragsfortsetzung |
| 5 | `englisch_vvg.pdf` | 20 | 21 | 21 | irrelevante Maklerberatung |

Die API-Metadatenseite ist nullbasiert. Deshalb ist die physische PDF-Seite
`metadata.page + 1`. `page_label` ist das aus dem PDF extrahierte gedruckte
Seitenlabel und weicht bei Handbook und Shopping Tool von der physischen Seite
ab.

### Waren die benötigten Aussagen im Generation Context?

| Benötigte Aussage | Vorhanden? |
|---|---|
| Allgemeiner Hinweis, dass eine gerissene/zerbrochene Windschutzscheibe unter Comprehensive fallen kann | ja |
| Baloise-spezifische Deckung von Glasbruch unter Part Comprehensive | nein |
| Selbstbehalt entfällt, wenn repariert statt ersetzt wird | **nein** |
| Meldung an Baloise Customer Service und Organisation über Baloise Partner | **nein** |
| Ausschluss bei Full-Service-Leasing als Bedingung der Zusatzleistung | **nein** |

### Forensischer Chroma-Nachweis – nicht für Generation verwendet

Die fehlenden Baloise-Passagen waren bereits indexiert:

- `240_1184_e.pdf`, Metadata-Seite 5, physische Seite 6, Startindex 1684:
  Part Comprehensive nennt den Bruch von Windschutz-, Seiten- und Heckscheiben
  als versichertes Ereignis.
- `240_1184_e.pdf`, Metadata-Seite 6, physische Seite 7, Startindex 3108:
  Bei Reparatur statt Austausch der Windschutzscheibe wird der Selbstbehalt
  nicht angewendet. Die Zusatzleistung gilt konditional, wenn Baloise Customer
  Service informiert wurde, die Leistung über einen Baloise-Partner organisiert
  wurde und kein Full-Service-Leasingvertrag besteht.

Die Datei enthält 78 bestehende Chunks; 33 davon enthalten mindestens einen der
forensisch geprüften Begriffe. Keiner der beiden benötigten Chunks wurde in den
realen Läufen ausgewählt.

## 8–16. Generation Context, OpenAI, Antwort, Sources und Guardrails

### Kontext im OpenAI-Aufruf

Jeder Lauf übergab:

| Kontextart | Anzahl |
|---|---:|
| PDF-Chunks | 5 |
| CRM-Kontexte | 4 |
| Gesamt | 9 |

Die vier CRM-Kontexte waren Kontakt, Police `1003`, Police `1001` und Claim
`2001`. Der Safety-Context-Schritt redigierte die Claim-ID, ließ die beiden
Policennummern als zulässige Geschäftskennungen bestehen.

### Answer-Generation-Aufrufe

Pro Lauf erfolgte genau eine erfolgreiche, anwendungsseitige
Answer-Generation-Initiierung und es gab `0` Pipeline-Retries. Der Adapter ruft
pro `invoke` genau einmal `OpenAI responses.create(...)` auf. Die OpenAI-SDK-
Konfiguration erlaubt bei einem transienten Transportfehler maximal einen
internen Retry; in den drei Läufen wurde kein Fehler oder Retry beobachtet.

### Lauf 1 – vollständige Antwort

```text
Generally, damage to the windscreen is covered under comprehensive insurance, which may reimburse the policyholder for a cracked or shattered windshield. If the windscreen can be repaired instead of replaced, the individual deductible may still apply, depending on the specific terms of the policy.

For Lara Neumann's current motor insurance:
- **Active Policy Number**: TEST-KFZ-2026-1001
- **Coverage Type**: Partial Coverage
- **Individual Deductible**: 150 EUR
- **Annual Premium**: 684 EUR

The general insurance terms indicate that glass damage can be covered, but the specifics regarding the deductible would depend on her policy details. 

[Doc-ID:page]
```

### Lauf 2 – vollständige Antwort

```text
Generally, damage to the windscreen is covered under comprehensive insurance, which may reimburse the policyholder for repairs or replacements. However, if the windscreen can be repaired instead of replaced, Lara would typically still have to pay her individual deductible.

Lara Neumann's active policy details are as follows:
- **Policy Number**: TEST-KFZ-2026-1001
- **Coverage Type**: Motor Insurance, Partial Coverage
- **Individual Deductible**: 150 EUR
- **Annual Premium**: 684 EUR

General insurance terms indicate that comprehensive coverage often includes windscreen damage, but specifics about deductibles may vary based on the policy and the nature of the repair. 

Sources:
- [Doc-ID:page]
- CRM FACT
```

### Lauf 3 – vollständige Antwort

```text
Generally, damage to the windscreen is covered under comprehensive insurance, which may reimburse the policyholder if a windshield is cracked or shattered. If the windscreen can be repaired instead of replaced, the individual deductible would typically still apply.

For Lara Neumann's specific policy:
- **Active Policy Number**: TEST-KFZ-2026-1001
- **Coverage Type**: Motor Insurance, Partial Coverage
- **Individual Deductible**: 150 EUR
- **Annual Premium**: 684 EUR

Source:
- General insurance terms: [Doc-ID:page]
- Lara's individual contract data: [espocrm:6a673465b147398ce]
```

### Antwort-Sources

Die API gab in jedem Lauf dieselben Sources zurück:

- PDF: zwei Chunks aus `Insurance_Handbook_20103.pdf`, Metadata-Seite 10.
- PDF: ein Chunk aus `consumer-auto-shopping-tool.pdf`, Metadata-Seite 10.
- PDF: zwei Chunks aus `englisch_vvg.pdf`, Metadata-Seiten 58 und 20.
- CRM: Lara-Kontakt, Police `1003`, Police `1001`, Claim `2001`.

Die Rückgabe der Source-Liste ist nicht gleichbedeutend mit einer gültigen
Inline-Citation. In allen drei Antworten war `[Doc-ID:page]` ein unveränderter
Platzhalter. Lauf 3 nannte zwar die echte CRM-ID der Police `1003`, der
Antworttext behauptete aber Police `1001`; die Citation ist deshalb
entailment-seitig falsch.

**Citation Validity: 0 %.** Es existiert keine gültige PDF-Citation und keine
kohärente CRM-Citation mit korrekter Zuordnung der Antwortbehauptung.

### Groundedness und Safety

| Lauf | Groundedness | Threshold | Ergebnis | Pre | Context | Post |
|---:|---:|---:|---|---|---|---|
| 1 | 0,636553 | 0,51 | bestanden | allow | redact | allow |
| 2 | 0,569547 | 0,51 | bestanden | allow | redact | allow |
| 3 | 0,634433 | 0,51 | bestanden | allow | redact | allow |

Safety Decision war jeweils `context_redact`. Gründe waren erkannte PII im
Kontext, zulässige Geschäftskennungen und die redigierte Claim-ID. Die
Endantwort wurde jeweils erlaubt. Es wurde keine endgültige Schadenentscheidung
ausgegeben; die fachlich falsche Selbstbehaltaussage wurde vom
Groundedness-Guardrail jedoch nicht erkannt.

## 17–20. Qualitätsmetriken

Die Metriken wurden als transparente, deterministische Claim-Prüfung gegen die
vom Auftrag vorgegebene Ground Truth berechnet.

### Bewertungsrubrik

- **Answer Correctness:** sechs atomare Antwortbehauptungen: allgemeine
  Glasdeckung, Reparaturregel, Policennummer, Coverage Type, Selbstbehalt und
  Jahresprämie.
- **Required-Fact Completeness:** zehn Anforderungen: Police `1003`, Partial
  Coverage, 300 EUR, 720 EUR, Glasdeckung, Reparatur-Waiver,
  Partnerbedingungen, 300 EUR nicht als Limit, klare Trennung CRM/Dokument und
  keine finale Schadenentscheidung.
- **Hallucination Rate:** nicht durch den Generation Context gestützte
  Behauptungen geteilt durch sechs. Aus dem falschen CRM-Datensatz übernommene
  Werte zählen als Selection-/Correctness-Fehler, nicht als Halluzination.
- **Source Completeness:** sieben belegpflichtige Kernaussagen im Generation
  Context: Glasdeckung, Reparatur-Waiver, Partnerbedingungen sowie Police
  `1003`, Partial Coverage, 300 EUR und 720 EUR.

| Metrik | Lauf 1 | Lauf 2 | Lauf 3 |
|---|---:|---:|---:|
| Answer Correctness | 33,33 % | 33,33 % | 33,33 % |
| Required-Fact Completeness | 40,00 % | 40,00 % | 40,00 % |
| Hallucination Rate | 16,67 % | 16,67 % | 16,67 % |
| Generation-Context Source Completeness | 71,43 % | 71,43 % | 71,43 % |
| Citation-basierte Source Completeness | 0,00 % | 0,00 % | 0,00 % |

Die niedrige Hallucination Rate darf nicht als gute Antwortqualität interpretiert
werden: Policennummer, 150 EUR und 684 EUR standen tatsächlich im Kontext, aber
sie gehörten zur falschen Police. Das ist primär ein Entity-/Policy-Selection-
Fehler.

## 21. Latenzen

### Einzelwerte

| Stage in ms | Lauf 1 | Lauf 2 | Lauf 3 |
|---|---:|---:|---:|
| Client End-to-End | 43.078,509 | 17.077,474 | 16.327,131 |
| Backend End-to-End | 42.969,988 | 16.974,051 | 16.221,228 |
| Route | 1,736 | 0,584 | 0,317 |
| CRM | 297,496 | 196,571 | 94,518 |
| Retrieval gesamt | 1.891,642 | 826,845 | 899,060 |
| BM25/lexical | 219,213 | 186,277 | 194,979 |
| Vector | 1.671,590 | 639,628 | 703,316 |
| Merge | 0,217 | 0,035 | 0,060 |
| Reranking | 27.732,533 | 8.524,333 | 8.661,006 |
| OpenAI Answer Generation | 6.718,028 | 3.488,871 | 2.590,475 |
| Guardrail gesamt | 909,529 | 199,892 | 201,987 |

### Stabilitätsaggregate

| Latenz | Minimum | Maximum | Mittelwert |
|---|---:|---:|---:|
| Client End-to-End | 16.327,131 ms | 43.078,509 ms | 25.494,371 ms |
| Backend End-to-End | 16.221,228 ms | 42.969,988 ms | 25.388,422 ms |
| CRM | 94,518 ms | 297,496 ms | 196,195 ms |
| Retrieval | 826,845 ms | 1.891,642 ms | 1.205,849 ms |
| Reranking | 8.524,333 ms | 27.732,533 ms | 14.972,624 ms |
| OpenAI | 2.590,475 ms | 6.718,028 ms | 4.265,791 ms |
| Guardrail | 199,892 ms | 909,529 ms | 437,136 ms |

Der größte Latenzanteil war der Reranker. Der erste Lauf enthielt zusätzlich
einen deutlichen Warm-up-Effekt. Eine separate Groundedness-Latenz wird von der
Produktionsdiagnostik nicht erfasst; sie ist Bestandteil der aggregierten
Guardrail-Latenz und wird daher nicht künstlich geschätzt.

## 22–23. Status und Audit Request IDs

| Lauf | HTTP | API-Status | Semantischer Status | Audit/API Request ID |
|---:|---:|---|---|---|
| 1 | 200 | `complete` | **FAIL** | `6bdee29d62ab4920b88ce6b269cd749d` |
| 2 | 200 | `complete` | **FAIL** | `3a599f84270a4ae3ad227490877527fa` |
| 3 | 200 | `complete` | **FAIL** | `0d254f6da8fa4a529ed8ec40aa2086d9` |

Die Audit-Datei erhielt genau drei neue Zeilen, eine pro Lauf. Ihr Schema
enthält aktuell kein `request_id`-Feld; die Korrelation erfolgte über die
Request ID der API-Diagnostics und den jeweiligen Audit-Zeitstempel. Das ist
eine Instrumentierungslücke, keine fehlende Ausführung des Audits.

## 24–25. Vorher-/Nachher-Integrität

### Chroma

| Prüfpunk | Vorher | Nachher | Ergebnis |
|---|---|---|---|
| Collection | `insurance_rag_collection` | gleich | PASS |
| Chunk-Anzahl | 8.551 | 8.551 | PASS |
| Logischer SHA-256 über alle IDs, Texte und Metadaten | `734e1970...2143` | `734e1970...2143` | **PASS** |
| Dateien | 46 | 46 | PASS |
| Gesamtgröße | 289.108.612 Bytes | 289.108.612 Bytes | PASS |
| PDF-Hashcache SHA-256 | `0b9c6b97...7dcd` | `0b9c6b97...7dcd` | PASS |
| Binärer Dateibaum-SHA-256 | `d2ef348b...1232e` | `f4fb4d4b...ea5e` | nicht byte-identisch |

Der binäre Chroma-Dateibaum ist nicht byte-identisch, obwohl der vollständige
kanonische Digest aller 8.551 IDs, Texte und Metadaten exakt gleich ist. Es
wurde keine `add`, `upsert`, `delete`, Collection-Neuanlage oder
Reindex-Funktion aufgerufen. Das Verhalten ist mit internem SQLite-/Chroma-
Persistenzverkehr bei realen Lese- und Vector-Query-Operationen vereinbar und
stellt keine logische Änderung des Indexbestands dar.

### CRM

Der normalisierte Digest der Lara-Projektion aus Kontakt, drei Policen und einem
Claim ist vor und nach dem Test identisch:

```text
17ddd5c06b5c65158175587c432fc365f8c840dbb871a14d9375384da574d5aa
```

Im eigentlichen Testfenster wurden neun EspoCRM-Requests beobachtet:

- 3 × `GET Contact`
- 3 × `GET Contact/{id}/mvaPolicies`
- 3 × `GET Contact/{id}/mvaClaims`
- 0 × `POST`, `PUT`, `PATCH` oder `DELETE` gegen EspoCRM

Damit ist bestätigt:

- kein CRM-Write;
- kein Reindex;
- keine PDF-Neuladung, kein Split und keine Neuindexierung;
- keine manuelle Chunk-Auswahl;
- keine produktive Datenänderung.

Das Audit selbst ist erwartungsgemäß append-only von 1.063 auf 1.066 Zeilen
gewachsen.

## Full-PASS-Matrix

| Kriterium | Ergebnis | Begründung |
|---|---|---|
| Route `combined` | PASS | alle drei Läufe |
| Nur `TEST-KFZ-2026-1003` im Generation Context | **FAIL** | `1001` zusätzlich enthalten |
| Erforderliche PDF-Passagen im finalen Context | **FAIL** | Baloise-Waiver und Partnerbedingungen fehlen |
| Glasbruch korrekt erklärt | PARTIAL | allgemeine Comprehensive-Aussage korrekt, aber nicht Baloise-/Part-Comprehensive-spezifisch |
| Konditionale Reparaturregel korrekt | **FAIL** | fehlt bzw. Antwort behauptet weiter anwendbaren Selbstbehalt |
| 300 EUR nicht als Deckungsgrenze | **FAIL** | 300 EUR fehlt; stattdessen falsche 150 EUR |
| CRM- und Dokumentaussagen getrennt | PASS | strukturell getrennt |
| Mindestens eine gültige PDF-Citation | **FAIL** | nur `[Doc-ID:page]` |
| Groundedness über Threshold | PASS | alle drei über 0,51 |
| Keine unbelegten Aussagen / keine finale Entscheidung | **FAIL** | keine finale Entscheidung, aber unbelegte Reparaturregel |
| Alle drei semantisch vollständig und sicher | **FAIL** | technisch sicher, fachlich unvollständig/falsch |

## Ursachenanalyse

Der Fehler entstand vor der Antwortgenerierung an zwei Stellen:

1. **CRM Entity/Policy Selection:** Die aktuelle Policy-Selektion lässt beide
   aktiven Motorpolicen in den Kontext. Durch den fälschlich aktivierten
   Claims-Zweig kommt zusätzlich ein alter Glasschaden mit alter
   Policenreferenz hinzu.
2. **Hybrid Retrieval Merge:** Fünf irrelevante BM25-Treffer werden vor den
   Vector-Treffern einsortiert. Bei `top_k=8` erreichen nur drei Vector-Treffer
   den Reranker. Der Baloise-Chunk wird gar nicht erst gerankt.

Danach verstärken sich beide Fehler: OpenAI erhält gleichzeitig die richtige
und die alte Police, aber nicht die entscheidende Baloise-Regel. Der
Groundedness-Score prüft die Stützung durch diesen fehlerhaften Kontext und
erkennt die falsche fachliche Auswahl nicht.

## Schlussentscheidung

Die Pipeline ist technisch ausführbar und reproduzierbar, erfüllt für diese
Frage aber die fachlichen und evidenzbezogenen Full-PASS-Kriterien nicht.
**Gesamtergebnis: FAIL.**

Die vollständige strukturierte Evidenz steht in
`combined_crm_rag_windscreen_e2e_20260729T085029Z.json`.
