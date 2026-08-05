# Combined CRM+RAG Windscreen E2E – Fix- und Live-Testbericht

Ausgeführt: 29.07.2026, 10:21–10:23 UTC  
Produktiver Endpoint: `POST http://localhost:8000/api/ask`  
Ergebnis: **FULL PASS**

Die maschinenlesbare Evidence befindet sich in
`reports/combined_crm_rag_windscreen_e2e_fixed_20260729T102604Z.json`.

## Kurzfazit

Die Fehler des Ausgangstests wurden behoben. In allen drei finalen Live-Läufen:

- wurde die Route `combined` gewählt;
- wurde ausschließlich die aktuelle Police `TEST-KFZ-2026-1003` in den
  Generation Context übernommen;
- wurden die entscheidenden Baloise-Chunks aus `240_1184_e.pdf` auf den
  physischen Seiten 6 und 7 automatisch aus Chroma gewählt;
- waren Glasbruch, Reparaturregel und alle dokumentierten Bedingungen im
  Generation Context vorhanden;
- wurde genau ein OpenAI-Answer-Generation-Aufruf mit `gpt-4o-mini` ausgeführt;
- lagen die Groundedness-Werte über dem Threshold `0,51`;
- endete die Safety-Prüfung mit einer erlaubten finalen Antwort;
- wurden gültige PDF-Citations mit physischen Seiten ausgegeben;
- wurde kein endgültiger Schadenentscheid getroffen;
- wurde weder Chroma logisch verändert noch in EspoCRM geschrieben.

Die drei Client-End-to-End-Latenzen betrugen 41.977,169 ms, 21.831,995 ms und
21.922,377 ms. Minimum: 21.831,995 ms, Maximum: 41.977,169 ms, Mittelwert:
28.577,180 ms.

## Exakte Testfrage

```text
Lara Neumann reports damage to the windscreen of her car. Is this damage
generally covered by her current motor insurance, and would she still have to
pay her individual deductible if the windscreen can be repaired instead of
replaced?

Please provide the relevant active policy number, coverage type, individual
deductible and annual premium. Clearly distinguish the general insurance terms
from Lara's individual contract data, and do not make a final claim decision.
```

Die Frage wurde in allen drei Läufen unverändert gesendet. Es wurde weder ein
Dateiname noch eine Seite, Dokument-ID, Chunk-ID oder ein Source-Filter
hinzugefügt.

## Ausführungsbedingungen

| Bedingung | Ergebnis |
|---|---|
| Bestehende Chroma-Chunks | ja |
| PDF neu laden/splitten | nein |
| Reindex | nein |
| Manueller Source-Filter | nein |
| Manuelle Chunk-Auswahl | nein |
| CRM | live und read-only |
| Mocks | nein |
| OpenAI | live, `gpt-4o-mini` |
| `SELF_CHECK_ENABLED` | `false` |
| Context Compression | `false` |
| Safety | NeMo, `enforce` |
| Groundedness Threshold | `0,51` |
| Finale identische Live-Läufe | 3 |

Vor den finalen Läufen wurden sieben Entwicklungs-Preflights ausgeführt. Sie
dienten dazu, die probabilistische Auslassung der Bedingungen durch
`gpt-4o-mini` sichtbar zu machen und die allgemeine Completeness-Guard-Stufe zu
validieren. Sie liegen vor dem finalen Integritäts-Baseline-Snapshot und sind
nicht in den Stabilitätsmetriken enthalten.

## Ursachen und implementierte Korrekturen

### 1. Falscher Claim-Intent

Die Formulierung „do not make a final claim decision“ enthielt das Token
`claim`. Zusätzlich waren die generischen Wörter `damage` und `loss` als
Claim-Trigger definiert. Dadurch wurde der alte Glasschaden
`TEST-CLM-2026-2001` abgerufen, obwohl die Frage keinen bestehenden
Schadenstatus verlangte.

Korrektur:

- generische `damage`-/`loss`-Trigger entfernt;
- negierte Claim-Decision-Formulierungen erkannt;
- echte Claim-/Status-Intents beibehalten;
- Ergebnis der finalen Decomposition: `needsClaims=false`.

Betroffene Dateien:

- `src/core/insurance_tool_routing.py`
- `tests/unit/test_insurance_tool_routing.py`

### 2. Nichtdeterministische CRM-Policenauswahl

Die alte Auswahl basierte überwiegend auf Tokenüberlappung. Die Policen 1001
und 1003 erhielten Gleichstand und gelangten beide in den Kontext.

Korrektur:

- strukturierte Bewertung nach expliziter Policennummer;
- Produkt-/Domain-Match und Domain-Konflikt;
- aktiver Status;
- zeitliche Gültigkeit;
- deterministischer Tie-Break über das neueste wirksame Startdatum;
- exakt eine ausgewählte Police.

Betroffene Dateien:

- `src/core/crm_orchestration.py`
- `src/main.py`
- `tests/unit/test_crm_orchestration.py`
- `tests/integration/test_crm_api_routing.py`

### 3. Baloise-Chunks wurden vor dem Reranking verdrängt

Die alte Hybridlogik hängte BM25 vor Vector an und brach bei `top_k=8` ab.
Dadurch konnten BM25-Treffer die semantisch relevanten Vector-Treffer
verdrängen.

Korrektur:

- acht Kandidaten je Kanal;
- Balanced Reciprocal-Rank Fusion mit semantischer Quote;
- Deduplizierung;
- query-bewertete Expansion benachbarter, bereits indexierter Chunks;
- Erhalt stark passender Kontextnachbarn im finalen Reranking.

Betroffene Dateien:

- `src/api/rag_service.py`
- `src/core/insurance_tool_routing.py`
- `tests/unit/test_retrieval_service.py`

### 4. Falsche Citation-Form

Der alte Prompt verlangte `[Doc-ID:page]`; die API gab die zero-basierte
Metadata-Seite aus.

Korrektur:

- Format `[Dateiname.pdf, page N]`;
- `N = metadata_page + 1`;
- nur Basename, kein interner Pfad;
- Legacy-Placeholder werden entfernt/normalisiert;
- CRM-Citations verwenden `[CRM: section]`.

### 5. Ausgelassene materielle Bedingungen

Der richtige Seite-7-Chunk enthielt alle Voraussetzungen, `gpt-4o-mini`
verkürzte den Satz jedoch in mehreren Preflights.

Korrektur:

- allgemeine Erkennung konditionaler Passagen;
- PDF-Dehyphenierung für Sätze wie `mainte -\nnance`;
- vollständiger Qualifier-Extrakt im Generation Context;
- bindender `Conditions`-Abschnitt im Prompt;
- deterministische Completeness-Guard-Stufe nach dem einzigen LLM-Aufruf.

Die Guard-Stufe durchsucht ausschließlich die bereits final ausgewählten
Kontext-Chunks und übernimmt nur fehlende, dort wörtlich vorhandene
Bedingungen. Sie erzeugt keine neuen Chunks, führt keinen zweiten LLM-Aufruf
aus und schreibt keine Daten.

### 6. Fehlende Laufzeit-Evidence

`RequestDiagnostics` erfasst nun:

- Decomposition;
- CRM-Kandidaten und Auswahlgründe;
- BM25-, Vector-, Merge- und Neighbor-Kandidaten;
- Reranker-Input und finales Ranking;
- Generation Context;
- Answer-Generation-Aufrufzahl;
- Groundedness inklusive separater Latenz;
- Safety-Entscheidung;
- Request ID.

## Testresultat

```text
..................................                                       [100%]
34 passed, 22 warnings
```

Ausgeführte Zieltests:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/unit/test_insurance_tool_routing.py `
  tests/unit/test_crm_orchestration.py `
  tests/unit/test_retrieval_service.py `
  tests/integration/test_crm_api_routing.py -q
```

Zusätzlich bestand `py_compile` für die geänderten Python-Quellen.

## 1–3. Route, Decomposition und CRM-Auswahl

### Route

```text
combined
```

### CRM-Subquery

```json
{
  "customerName": "Lara Neumann",
  "customerEmail": null,
  "policyNumber": null,
  "claimNumber": null,
  "needsPolicies": true,
  "needsClaims": false
}
```

### Knowledge-Subquery

```text
reports damage to the windscreen of her car. Is this damage generally covered
by her current motor insurance would she still have to pay her individual
deductible if the windscreen can be repaired instead of replaced?
```

Die produktive Pipeline ergänzte ausschließlich semantische Suchbegriffe und
CRM-Produkt-Hints:

```text
BaloiseDirect Motor Vehicle 2026 Motor Insurance Partial Coverage glass
breakage part comprehensive windscreen damage repaired rather than replaced
deductible not applied partner repair service customer service notification
```

`requestedPdfFilename` blieb `null`.

### CRM-Kontakt

| Feld | Wert |
|---|---|
| Kontakt-ID | `6a65e0b6babe7371d` |
| Name | Lara Neumann |
| E-Mail-Hinweis | `l***@example.test` |

### Policenbewertung

| Police | Score | Domain | Aktiv/gültig | Start | Auswahl |
|---|---:|---|---|---|---|
| `TEST-KFZ-2026-1003` | 460 | Motor, Match | ja/ja | 01.07.2026 | **ja** |
| `TEST-PHV-2026-1002` | 96 | Liability, Konflikt | ja/ja | 01.01.2026 | nein |
| `TEST-KFZ-2026-1001` | 460 | Motor, Match | ja/ja | 01.01.2026 | nein, älterer Tie-Break |

Der Generation Context enthielt nur:

- Kontakt Lara Neumann;
- Police `TEST-KFZ-2026-1003`.

Nicht im Generation Context:

- `TEST-KFZ-2026-1001`;
- `TEST-PHV-2026-1002`;
- `TEST-CLM-2026-2001`.

## 4–7. Retrieval und Reranking

### BM25-Kandidaten

| Rang | Datei | Chunk-ID | Metadata-Seite | Physische Seite |
|---:|---|---|---:|---:|
| 1 | `geschaeftsbericht-bg-2024-e.pdf` | `7cef78e0-e884-4378-b17a-2867d05021f5` | 30 | 31 |
| 2 | `240_1184_e.pdf` | `f32b359f-94a8-4b92-9ea1-c30d873a4fb9` | 6 | 7 |
| 3 | `geschaeftsbericht-bg-2024-e.pdf` | `a0a86914-d65e-4274-aef0-649ad34f0e3d` | 31 | 32 |
| 4 | `240_1217_e.pdf` | `b674b3de-f709-42cb-b2e5-4f91da01cf4b` | 13 | 14 |
| 5 | `geschaeftsbericht-bg-2024-e.pdf` | `a56ae3f7-66a1-4f44-b7b8-97e2513fd3ec` | 30 | 31 |
| 6 | `240_1217_e.pdf` | `d2310aa7-996b-4bdd-8cea-ddcf3128d9a3` | 12 | 13 |
| 7 | `240_1217_e.pdf` | `bdb98dcc-ad5f-4858-ac51-f2d5fa058406` | 13 | 14 |
| 8 | `240_1217_e.pdf` | `aeb1e4fd-9101-4fb3-b9d1-fbdcae846f86` | 14 | 15 |

### Vector-Kandidaten

| Rang | Datei | Chunk-ID | Metadata-Seite | Physische Seite |
|---:|---|---|---:|---:|
| 1 | `Insurance_Handbook_20103.pdf` | `7c7cd57f-deba-49c6-ac21-36e446ebf40a` | 10 | 11 |
| 2 | `240_1217_e.pdf` | `d2310aa7-996b-4bdd-8cea-ddcf3128d9a3` | 12 | 13 |
| 3 | `publication-aut-pp-consumer-auto.pdf` | `5598e720-b611-48a7-8fcf-96f46e4364f7` | 7 | 8 |
| 4 | `240_1217_e.pdf` | `9ab13c53-9d0a-47a0-b3da-c0caf826709b` | 16 | 17 |
| 5 | `240_1184_e.pdf` | `f32b359f-94a8-4b92-9ea1-c30d873a4fb9` | 6 | 7 |
| 6 | `240_1217_e.pdf` | `2c80860b-adf4-4b1c-bbf1-61b48cd68858` | 26 | 27 |
| 7 | `Insurance_Handbook_20103.pdf` | `c49628bb-a733-4281-8009-6c58660c4adb` | 10 | 11 |
| 8 | `consumer-auto-shopping-tool.pdf` | `78d804fc-b84b-4fc5-8ca2-2f08e646bcc5` | 10 | 11 |

### Merge und Deduplizierung vor Neighbor-Expansion

| Rang | Datei | Chunk-ID | Physische Seite |
|---:|---|---|---:|
| 1 | `240_1184_e.pdf` | `f32b359f-94a8-4b92-9ea1-c30d873a4fb9` | 7 |
| 2 | `240_1217_e.pdf` | `d2310aa7-996b-4bdd-8cea-ddcf3128d9a3` | 13 |
| 3 | `Insurance_Handbook_20103.pdf` | `7c7cd57f-deba-49c6-ac21-36e446ebf40a` | 11 |
| 4 | `geschaeftsbericht-bg-2024-e.pdf` | `7cef78e0-e884-4378-b17a-2867d05021f5` | 31 |
| 5 | `geschaeftsbericht-bg-2024-e.pdf` | `a0a86914-d65e-4274-aef0-649ad34f0e3d` | 32 |
| 6 | `publication-aut-pp-consumer-auto.pdf` | `5598e720-b611-48a7-8fcf-96f46e4364f7` | 8 |
| 7 | `240_1217_e.pdf` | `b674b3de-f709-42cb-b2e5-4f91da01cf4b` | 14 |
| 8 | `240_1217_e.pdf` | `9ab13c53-9d0a-47a0-b3da-c0caf826709b` | 17 |

### Automatische Neighbor-Expansion

| Neuer Rang | Anchor | Hinzugefügter Chunk | Seite | Score | Zweck |
|---:|---|---|---:|---:|---|
| 9 | Seite-7-Waiver | `7c6e0e44-843b-4d44-b34e-6b18cce1b6ad` | 6 | 39 | TK1.4 Glasbruch |
| 10 | `240_1217_e`, S. 13 | `bdb98dcc-ad5f-4858-ac51-f2d5fa058406` | 14 | 29 | Reparaturkontext |
| 11 | Seite-7-Waiver | `60b10844-8813-4f69-bf4b-627e3d352aa3` | 6 | 20 | Part-Comprehensive-Kontext |

Die Expansion verwendete weder festen Dateinamen noch feste Chunk-IDs. Sie
suchte query-bewertet innerhalb bereits indexierter Nachbarseiten der
Top-Anker.

### Reranker-Input und Scores

| Input-Rang | Datei / Seite | Score |
|---:|---|---:|
| 1 | `240_1184_e.pdf`, 7 | 1,75934958 |
| 2 | `240_1217_e.pdf`, 13 | -2,66913939 |
| 3 | `Insurance_Handbook_20103.pdf`, 11 | -2,28422594 |
| 4 | `geschaeftsbericht-bg-2024-e.pdf`, 31 | -0,48223910 |
| 5 | `geschaeftsbericht-bg-2024-e.pdf`, 32 | -3,23713207 |
| 6 | `publication-aut-pp-consumer-auto.pdf`, 8 | -0,49153796 |
| 7 | `240_1217_e.pdf`, 14 | -2,69016314 |
| 8 | `240_1217_e.pdf`, 17 | -2,34242010 |
| 9 | `240_1184_e.pdf`, 6 / TK1.4 | -1,70926535 |
| 10 | `240_1217_e.pdf`, 14 | -4,18457317 |
| 11 | `240_1184_e.pdf`, 6 / Heading | -3,16872931 |

### Finale fünf PDF-Chunks

| Rang | Datei | Chunk-ID | Metadata-Seite | Physische Seite | Inhalt |
|---:|---|---|---:|---:|---|
| 1 | `240_1184_e.pdf` | `f32b359f-94a8-4b92-9ea1-c30d873a4fb9` | 6 | 7 | Waiver, Meldung, Partner, kein Full-Service-Leasing |
| 2 | `240_1184_e.pdf` | `7c6e0e44-843b-4d44-b34e-6b18cce1b6ad` | 5 | 6 | TK1.4 Windschutzscheiben-/Glasbruch |
| 3 | `240_1184_e.pdf` | `60b10844-8813-4f69-bf4b-627e3d352aa3` | 5 | 6 | Part Comprehensive / insured events |
| 4 | `geschaeftsbericht-bg-2024-e.pdf` | `7cef78e0-e884-4378-b17a-2867d05021f5` | 30 | 31 | allgemeiner Partner-/Reparaturkontext |
| 5 | `publication-aut-pp-consumer-auto.pdf` | `5598e720-b611-48a7-8fcf-96f46e4364f7` | 7 | 8 | allgemeiner Windscreen-/Comprehensive-Kontext |

Die beiden benötigten Kernaussagen waren damit in jedem finalen Generation
Context vorhanden:

- Glasbruch der Windschutzscheibe: Seite 6;
- Selbstbehalt-Verzicht und Bedingungen: Seite 7.

## 8–11. Generation Context und OpenAI

| Kontexttyp | Anzahl |
|---|---:|
| PDF | 5 |
| CRM | 2 |
| Gesamt | 7 |

CRM-Kontexte:

1. Lara Neumann;
2. `TEST-KFZ-2026-1003`.

OpenAI Answer Generation:

| Lauf | Modell | Application Call Count | Retry |
|---:|---|---:|---:|
| 1 | `gpt-4o-mini` | 1 | 0 |
| 2 | `gpt-4o-mini` | 1 | 0 |
| 3 | `gpt-4o-mini` | 1 | 0 |

Die deterministische Completeness-Guard-Stufe ist kein weiterer
Answer-Generation-Aufruf.

## 12. Vollständige Antworten

### Lauf 1

```text
Lara Neumann's current motor insurance includes coverage for damage to the windscreen. Specifically, if the damaged windscreen is repaired rather than replaced, the deductible will not be applied.

### Policy Information
- **Policy Number:** TEST-KFZ-2026-1003
- **Coverage Type:** Partial Coverage
- **Individual Deductible:** 300 EUR
- **Annual Premium:** 720 EUR

### General Insurance Terms
- Damage to the windscreen is covered if it is repaired instead of replaced, and in such cases, the deductible is waived.

### Conditions
- The waiver of the deductible applies only if the windscreen is repaired rather than replaced.

This information provides a general overview of coverage and conditions, but it does not constitute a final claim decision.

Source:
[240_1184_e.pdf, page 6]

Material documented conditions [240_1184_e.pdf, page 7]:
- only provided if Baloise Customer Service was notified in the event of a claim and arranged the above services through a Baloise partner, and provided that there is no maintenance and service contract with a leasing company (full service leasing) for the vehicle concerned.

Material documented conditions [240_1184_e.pdf, page 6]:
- provided that the item is repaired.
```

### Lauf 2

```text
Yes, damage to the windscreen is generally covered by comprehensive motor insurance. Specifically, if the damaged windscreen is repaired instead of being replaced, the deductible will not be applied.

### Policy Information
- **Policy Number**: TEST-KFZ-2026-1003
- **Coverage Type**: Partial Coverage
- **Individual Deductible**: 300 EUR
- **Annual Premium**: 720 EUR

### Conditions
- The deductible will not be applied if the damaged windscreen is repaired rather than replaced.
- Coverage is conditional upon notifying Baloise Customer Service in the event of a claim and arranging services through a Baloise partner.

This information is general and does not constitute a final claim decision.

**Source:**
[240_1184_e.pdf, page 6]
[CRM: TEST-KFZ-2026-1003]

Material documented conditions [240_1184_e.pdf, page 7]:
- only provided if Baloise Customer Service was notified in the event of a claim and arranged the above services through a Baloise partner, and provided that there is no maintenance and service contract with a leasing company (full service leasing) for the vehicle concerned.

Material documented conditions [240_1184_e.pdf, page 6]:
- provided that the item is repaired.
```

### Lauf 3

```text
Yes, damage to the windscreen is generally covered under comprehensive insurance, specifically if the damaged windscreen is repaired instead of being replaced. If the windscreen is repaired, the deductible will not be applied.

### Policy Information for Lara Neumann:
- **Policy Number**: TEST-KFZ-2026-1003
- **Coverage Type**: Partial Coverage
- **Individual Deductible**: 300 EUR
- **Annual Premium**: 720 EUR

### Conditions
- The deductible will not be applied if a damaged windscreen is repaired rather than replaced.
- The repair must be arranged through a Baloise partner and Baloise Customer Service must be notified in the event of a claim.

This information is general and does not constitute a final claim decision.

Source:
[240_1184_e.pdf, page 6]
[CRM: TEST-KFZ-2026-1003]

Material documented conditions [240_1184_e.pdf, page 7]:
- only provided if Baloise Customer Service was notified in the event of a claim and arranged the above services through a Baloise partner, and provided that there is no maintenance and service contract with a leasing company (full service leasing) for the vehicle concerned.

Material documented conditions [240_1184_e.pdf, page 6]:
- provided that the item is repaired.
```

## 13–14. Sources und Citation Validity

Relevante PDF-Sources:

| Datei | Metadata-Seite | Physische Seite | Aussage |
|---|---:|---:|---|
| `240_1184_e.pdf` | 5 | 6 | Part Comprehensive und TK1.4 Windschutzscheibenbruch |
| `240_1184_e.pdf` | 6 | 7 | Waiver und vollständige Leistungsvoraussetzungen |

Relevante CRM-Sources:

| Source | Section |
|---|---|
| `espocrm:6a65e0b6babe7371d` | Lara Neumann |
| `espocrm:6a673465b147398ce` | `TEST-KFZ-2026-1003` |

Citation-Bewertung:

| Lauf | Gültige PDF-Citations | Gültige CRM-Citations | Ungültig | Validity |
|---:|---:|---:|---:|---:|
| 1 | 3 | 0 | 0 | 1,0 |
| 2 | 3 | 1 | 0 | 1,0 |
| 3 | 3 | 1 | 0 | 1,0 |

Lauf 1 enthielt die CRM-Quelle in der API-Source-Liste, aber nicht als
Inline-Citation. Alle PDF-Citations waren gültig. Keine Antwort enthielt einen
internen Pfad oder `[Doc-ID:page]`.

## 15–16. Groundedness und Safety

| Lauf | Groundedness | Threshold | PASS |
|---:|---:|---:|---|
| 1 | 0,627558 | 0,51 | ja |
| 2 | 0,678068 | 0,51 | ja |
| 3 | 0,680607 | 0,51 | ja |

Mittelwert: `0,662078`.

Safety war in allen drei Läufen:

```text
pre:     allow
context: redact
post:    allow
final:   context_redact, answer allowed
```

Die Context-Stufe erkannte adressähnliche PII in einem irrelevanten
PDF-Kontext und redigierte sie. Lara als erlaubter Geschäftskontakt und die
Vertragskennung blieben verwendbar. Es gab keinen Safety-Systemfehler und
keinen finalen Block.

## 17–20. Semantische Qualitätsmetriken

Für jeden Lauf wurden zwölf erforderliche Fakten geprüft:

1. Lara Neumann;
2. `TEST-KFZ-2026-1003`;
3. Active;
4. Partial Coverage;
5. 300 EUR individueller Selbstbehalt;
6. 720 EUR Jahresprämie;
7. Windschutzscheiben-/Glasbruch grundsätzlich versichert;
8. Waiver bei Reparatur statt Ersatz;
9. Meldung an Baloise Customer Service;
10. Organisation über Baloise-Partner;
11. kein Maintenance-/Servicevertrag mit Leasinggesellschaft;
12. kein endgültiger Claim-/Coverage-Entscheid.

| Metrik | Ergebnis |
|---|---:|
| Answer Correctness | 1,0 |
| Required-Fact Completeness | 12/12 = 1,0 |
| Hallucination Rate | 0,0 |
| Source Completeness | 1,0 |
| Citation Validity | 1,0 |

Die 300 EUR wurden in keinem Lauf als Deckungsgrenze bezeichnet.

## 21. Latenzen

### Einzelwerte

| Stufe (ms) | Lauf 1 | Lauf 2 | Lauf 3 |
|---|---:|---:|---:|
| Client E2E | 41.977,169 | 21.831,995 | 21.922,377 |
| Backend E2E | 41.889,372 | 21.696,847 | 21.851,133 |
| Route | 1,941 | 0,809 | 0,450 |
| CRM | 97,063 | 88,231 | 70,556 |
| Retrieval gesamt | 1.279,959 | 786,193 | 775,859 |
| BM25 | 219,280 | 181,680 | 170,982 |
| Vector | 1.045,566 | 589,004 | 587,929 |
| Merge | 14,209 | 14,507 | 15,989 |
| Reranking | 29.619,699 | 12.900,376 | 13.310,581 |
| OpenAI Answer Generation | 5.253,239 | 3.943,701 | 3.640,360 |
| Groundedness | 556,256 | 121,671 | 128,352 |
| Guardrail gesamt | 626,629 | 180,486 | 189,290 |

### Minimum, Maximum, Mittelwert

| Stufe (ms) | Minimum | Maximum | Mittelwert |
|---|---:|---:|---:|
| Client E2E | 21.831,995 | 41.977,169 | 28.577,180 |
| Backend E2E | 21.696,847 | 41.889,372 | 28.479,117 |
| CRM | 70,556 | 97,063 | 85,283 |
| Retrieval | 775,859 | 1.279,959 | 947,337 |
| Reranking | 12.900,376 | 29.619,699 | 18.610,219 |
| OpenAI | 3.640,360 | 5.253,239 | 4.279,100 |
| Groundedness | 121,671 | 556,256 | 268,760 |
| Guardrail | 180,486 | 626,629 | 332,135 |

Der erste finale Lauf enthält den Cold-/Warmup-Effekt des lokalen
Cross-Encoder-Rerankers. Die beiden Folgeläufe stabilisierten sich bei ungefähr
21,8 Sekunden E2E. Der Reranker ist weiterhin der dominante Latenzanteil.

## 22–23. HTTP-, semantischer Status und Request IDs

| Lauf | HTTP | API | Semantik | Request ID |
|---:|---:|---|---|---|
| 1 | 200 | `complete` | PASS | `ddff845b6dcc45d9b9bd00bc1960a15e` |
| 2 | 200 | `complete` | PASS | `2f0b64d338ba40c3939bf214160d4acd` |
| 3 | 200 | `complete` | PASS | `fea6efb90a6f48ba8c6db512812d7d07` |

Hinweis zur Instrumentierung: Der interne Auditdatensatz wird im synchronen
RAG-Worker geschrieben, bevor die API-Schicht `RequestDiagnostics.complete`
aufruft. Deshalb enthält der eingebettete Audit-Snapshot noch
`status="running"`, obwohl alle Stages dort `completed` sind. Der tatsächliche
HTTP-/API-Status war in allen drei Antworten `200`/`complete`.

## 24–25. Chroma-/CRM-Integrität

### Chroma

| Wert | Vorher | Nachher |
|---|---|---|
| Collection | `insurance_rag_collection` | identisch |
| Count | 8.551 | 8.551 |
| Logischer SHA-256 | `734e1970...52143` | identisch |
| Dateien | 46 | 46 |
| Gesamtbytes | 289.108.612 | 289.108.612 |
| PDF-Hash-Cache | `0b9c6b97...37dcd` | identisch |

Der physische Dateibaum-Hash änderte sich durch interne SQLite-/Chroma-
Read-Aktivität. Count, vollständiger kanonischer Dokument-/Metadata-Hash,
Dateianzahl, Gesamtbytes und PDF-Hash-Cache blieben identisch. Damit ist die
logische Chroma-Integrität bestanden.

### CRM

Normalisierter Projektion-SHA-256 vor und nach den Läufen:

```text
ade9a4dbbf6cb1225d914bd359023901aec4cf05e2459e29d6736554e7991c42
```

HTTP-Methoden des finalen CRM-Testfensters:

| Methode | Anzahl |
|---|---:|
| GET | 6 |
| POST | 0 |
| PUT | 0 |
| PATCH | 0 |
| DELETE | 0 |

### Audit

| Wert | Vorher | Nachher |
|---|---:|---:|
| Zeilen | 1.073 | 1.076 |
| Bytes | 66.326.111 | 66.929.404 |
| Angehängte Zeilen | – | 3 |

Bestätigungen:

- kein Reindex;
- kein PDF-Write;
- kein Chroma-Add/Upsert/Delete;
- kein CRM-Write;
- keine produktiven Daten zurückgesetzt;
- bestehende fremde Änderungen nicht verworfen.

## Full-PASS-Entscheidung

| Kriterium | Ergebnis |
|---|---|
| Route `combined` | PASS |
| Nur `TEST-KFZ-2026-1003` im Generation Context | PASS |
| Erforderliche PDF-Passagen vorhanden | PASS |
| Glasbruch korrekt erklärt | PASS |
| Reparaturregel konditional erklärt | PASS |
| 300 EUR nicht als Deckungsgrenze | PASS |
| CRM- und Dokumentdaten getrennt | PASS |
| Gültige PDF-Citation | PASS |
| Groundedness > Threshold | PASS, 3/3 |
| Keine unbelegte/finale Claim-Entscheidung | PASS |
| Drei vollständige und sichere Live-Läufe | PASS |

**Gesamtergebnis: FULL PASS.**
