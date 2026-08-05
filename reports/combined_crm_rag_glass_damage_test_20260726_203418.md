# Combined CRM + RAG Glasschaden E2E-Test

Zeitpunkt: 26. Juli 2026  
Repository: `C:\Users\mirae\MVA_Versicherung_Langchain_main`

## 1. Ergebnis in einem Satz

Der reale Combined-Pfad hat CRM und Retrieval korrekt ausgeführt, ist aber in beiden maßgeblichen Läufen nach jeweils rund 30 Sekunden im Stage `self_check` abgebrochen. Die Anwendung antwortete korrekt mit HTTP 206 und ausschließlich belegten CRM-Fakten. Deshalb gilt: **Functional PARTIAL, Complete Combined E2E FAIL, Overall PARTIAL**.

## 2. Ziel und Testfrage

Die Frage musste zwingend individuelle EspoCRM-Daten und allgemeine PDF-Fachinformation verbinden:

> Lara Neumann reported windshield glass damage. Is glass damage generally covered under partial coverage motor insurance? Also provide the coverage type, specific deductible, and policy number stored for the motor policy of Lara Neumann. Separate the general document rule from the individual CRM policy data.

Benötigte Fakten:

1. Identität des Kontakts;
2. Zuordnung Kontakt ↔ Kfz-Police;
3. Policennummer;
4. Deckungsart;
5. individuelle Selbstbeteiligung;
6. allgemeine, dokumentbelegte Regel zur Glasschaden-Deckung.

Es wurden keine CRM-Schreiboperation, kein Export, kein Reindex und keine Modell-, Timeout-, k-, Reranker-, Guardrail- oder Retry-Änderung vorgenommen.

## 3. Systemzustand

| Prüfung | Ergebnis |
|---|---|
| `mva-backend` | running, healthy |
| `mva-espocrm` | running, healthy |
| `mva-espocrm-db` | running, healthy |
| `/health` | `status=ok` |
| `pipelineReady` | `true` |
| `crmReady` | `true` |
| `retrievalReady` | `true` |
| `llmReady` | `true` |
| EspoCRM | HTTP 200 auf `localhost:8080` |
| Ollama | erreichbar; 6 Modelle durch `/api/tags` gemeldet |
| Chroma vor Test | `insurance_rag_collection = 8551` |
| Safety | aktiviert, Modus `enforce`, Backend NeMo |

Die fünf entdeckten CRM-MCP-Tools waren genau:

- `find_customer`
- `get_customer_policies`
- `get_policy`
- `get_customer_claims`
- `get_claim_status`

Alle verwendeten CRM-Zugriffe waren read-only.

## 4. CRM Ground Truth

Verwendete Entities: `Contact`, `MvaPolicy`, `MvaClaim`. Zugriff über die bestehenden read-only MCP-Tools und EspoCRM REST API.

| Feld | Tatsächlich gelesener Wert |
|---|---|
| Contact | Lara Neumann |
| E-Mail in Ausgabe | `l***@example.test` |
| relevante Police | Lara Neumann Motor 2026 |
| Policy Number | `TEST-KFZ-2026-1001` |
| Produkt | Motor Insurance |
| Coverage Type | Partial Coverage |
| Status | Active |
| Selbstbeteiligung | 150 EUR |
| Jahresprämie | 684 EUR |
| Laufzeit | 2026-01-01 bis 2026-12-31 |

Lara hat außerdem `TEST-PHV-2026-1002` (Personal Liability). Diese Police ist für die Kfz-Glasschadenfrage nicht relevant. Die Auswahl von `TEST-KFZ-2026-1001` beruht somit auf Produkt und Zuordnung, nicht auf blindem Hardcoding.

Der vorhandene Claim `TEST-CLM-2026-2001` hat den Typ Glass Damage, den Status Under Review, ein Meldedatum vom 10. Juli 2026 und einen beanspruchten Betrag von 780 EUR. Dieser Claim belegt keine Leistungsentscheidung.

## 5. RAG/PDF Ground Truth

Die dedizierte reine Retrieval-Abfrage lief ohne Generation und ohne Reindex:

> Under partial coverage motor insurance, is windshield glass damage covered, and what conditions or limitations apply?

Sie lieferte 8 Retrieval-Kandidaten und 5 rerankte Chunks. Vier der fünf waren fachlich relevant (`4/5 = 0,8`).

Wichtige Nachweise:

| Retrieval-Rang | Datei | Metadata Page | PDF-Seite | Nachgewiesene Information |
|---:|---|---:|---:|---|
| 1 | `publication-aut-pp-consumer-auto.pdf` | 7 | 8 | Windschutzscheiben mit Rissen oder Beschädigungen können durch Comprehensive gedeckt sein; bei manchen Anbietern entfällt bei Reparatur die Selbstbeteiligung. |
| 2 | `Insurance_Handbook_20103.pdf` | 10 | 11 | Comprehensive kann gerissene oder zerbrochene Windschutzscheiben erstatten; separate Glasdeckung kann mit oder ohne Selbstbeteiligung angeboten werden. |
| 3 | `240_1217_e.pdf` | 12 | 13 | Part comprehensive, TK1.4, nennt Bruch von Windschutz-, Seiten- und Heckscheiben. |
| 4 | `240_1184_e.pdf` | 1 | 2 | Part comprehensive schließt Glasschäden ausdrücklich ein. |

Für die exakte Combined-Frage wurden die fünf Kandidaten vor `self_check` zusätzlich ohne Generation reproduziert:

| Rang | Datei / Seite | Zeichen | Bewertung | Begründung |
|---:|---|---:|---|---|
| 1 | `Insurance_Handbook_20103.pdf`, S. 11 | 922 | relevant | allgemeine Windschutzscheiben- und Selbstbeteiligungsregel |
| 2 | `consumer-auto-shopping-tool.pdf`, S. 11 | 989 | irrelevant | nur Fragen an Verbraucher, keine positive Deckungsregel |
| 3 | `140_1261_e.pdf`, S. 20 | 997 | irrelevant | Gebäude-/Haushaltsglas statt Kfz |
| 4 | `240_1217_e.pdf`, S. 14 | 975 | relevant | konditionale Werkstattbindung und erhöhte Selbstbeteiligung bei vereinbartem EasyRepair |
| 5 | `240_1184_e.pdf`, S. 2 | 993 | relevant | Glasschaden ausdrücklich unter Part comprehensive |

Damit beträgt die für die tatsächliche Combined-Frage maßgebliche Chunk-Relevanz `3/5 = 0,6`.

Die Dateien existieren. Die physischen Seiten 8, 13, 14 und 2 wurden mit Poppler gerendert und visuell geprüft. Die Zuordnung `metadata.page + 1 = menschliche PDF-Seite` stimmt bei diesen Nachweisen; es liegt keine Off-by-one-Verwechslung vor.

Fachliche Ground Truth: Teilkasko/Part comprehensive deckt Glasschäden einschließlich Windschutzscheiben grundsätzlich. Bedingungen und Selbstbeteiligung können variieren. Eine Werkstattbindung aus `240_1217_e.pdf` gilt nur, wenn EasyRepair Glass oder EasyRepair Plus konkret vereinbart ist. Das CRM belegt eine Selbstbeteiligung von 150 EUR, aber keine Deckungssumme.

## 6. Erwartete kombinierte Antwort

Eine vollständige, vorsichtig formulierte Antwort hätte die Quellen so getrennt:

> **Allgemeine Dokumentregel:** Teilkasko/Partial Coverage schließt Glasbruch und Windschutzscheibenschäden grundsätzlich ein. Je nach Bedingungswerk können eine Selbstbeteiligung und – sofern konkret vereinbart – Vorgaben zu zertifizierten Reparaturpartnern gelten.  
> **Individuelle CRM-Daten:** Lara Neumann ist der aktiven Kfz-Police `TEST-KFZ-2026-1001` mit Partial Coverage und 150 EUR Selbstbeteiligung zugeordnet. Die 150 EUR sind eine Selbstbeteiligung, keine nachgewiesene Deckungsgrenze. Eine konkrete Leistungsentscheidung bleibt von den individuellen Vertragsbedingungen und der Schadenprüfung abhängig.

## 7. Tatsächliche Live-Response

Beide maßgeblichen Requests gingen real über `POST /api/ask`; CRM- und Retrieval-Ergebnisse wurden nicht simuliert.

```text
CRM facts:
Customer: Lara Neumann (l***@example.test).
Policy TEST-PHV-2026-1002: Personal Liability, Liability, Active; term 2026-01-01 to 2026-12-31; deductible 0 EUR; annual premium 96 EUR.
Policy TEST-KFZ-2026-1001: Motor Insurance, Partial Coverage, Active; term 2026-01-01 to 2026-12-31; deductible 150 EUR; annual premium 684 EUR.
Claim TEST-CLM-2026-2001: Glass Damage, status Under Review; reported 2026-07-10; claimed amount 780 EUR; policy Lara Neumann Motor 2026.

Document evidence:
Unavailable. No coverage or claim decision was inferred from CRM data.
```

Die vier gelieferten Sources waren ausschließlich CRM-Sources (Contact, zwei Policies, Claim). `knowledgeResult` war `null`; eine PDF-Source wurde nicht an die Antwort angehängt.

Das Verhalten ist sicher und vertragstreu: Das System erfand weder die Glasschaden-Deckung noch eine Leistungsentscheidung aus CRM-Daten.

### Transparenz zum ersten Vorversuch

Eine anfängliche Formulierung begann mit „Does Lara Neumann's …“. Der konservative Namensparser interpretierte dabei `Does Lara` als Namen; die Route blieb zwar `combined`, CRM fand aber keinen Kontakt. Diese zwei Responses werden nicht zur funktionalen Bewertung herangezogen. Danach wurde ausschließlich die isolierte Testfrage so umformuliert, dass sie mit `Lara Neumann` beginnt. Produktiver Routing-Code und Runtime-Konfiguration blieben unverändert.

Der erste Vorversuch hatte 76.035,718 ms und der direkte Warm-Lauf 46.910,022 ms (`1,6209×`). Die autoritative korrigierte Paarmessung fand nach diesem Vorversuch statt und ist deshalb als First/Cold-Label, nicht als vollständig prozesskalter Start zu verstehen.

## 8. Functional Evaluation

| Kriterium | Ergebnis |
|---|---|
| Route tatsächlich `combined` | PASS |
| CRM real aufgerufen | PASS |
| Retrieval real aufgerufen | PASS |
| richtige Person und Kfz-Police | PASS |
| Policy Number korrekt | PASS |
| Selbstbeteiligung korrekt | PASS |
| Retrieval und Reranking abgeschlossen | PASS |
| HTTP 206 statt 500/inkorrektem 200 | PASS |
| unbelegte Deckungsaussage vermieden | PASS |
| Dokumentbeleg in finaler Response | FAIL |
| vollständige Combined-Antwort | FAIL |

Ergebnis: **Functional PARTIAL**.  
`Combined Partial Contract = PASS`; `Complete Combined E2E = FAIL`.

## 9. Quality Metrics

### Kontextrelevanz

Definition gemäß der im Projekt verwendeten Aussage-/Kontextlogik: relevante rerankte Chunks geteilt durch alle fünf Kandidaten der exakten Combined-Frage.

`3 / 5 = 0,6`

Die dedizierte fachliche Ground-Truth-Abfrage erzielte zusätzlich `4 / 5 = 0,8`. Der niedrigere und strengere Score `0,6` wird als Gesamtwert verwendet, weil er zur exakten Combined-Frage gehört.

### Kontextgenügsamkeit

| Benötigter Fakt | Im kombinierten Kontext vorhanden? |
|---|---|
| Person | ja, CRM |
| Police/Zuweisung | ja, CRM |
| Vertragsnummer | ja, CRM |
| Deckungsart | ja, CRM |
| Selbstbeteiligung | ja, CRM |
| Glasschaden-Deckung | ja, PDF/RAG |

`6 / 6 = 1,0`

Der Kontext war vollständig; der Timeout entstand bei seiner Laufzeitverarbeitung.

### Antwort-Halluzinationsrate

Die tatsächlich ausgegebene Partial-Antwort enthält 22 einzeln überprüfbare CRM-Fakten. Alle sind durch die vier CRM-Sources gestützt; keine PDF-Aussage wurde vorgetäuscht.

`0 nicht gestützt / 22 überprüfbar = 0,0`

Einschränkung: Der Score gilt nur für die ausgegebene Partial-Antwort. Er bewertet nicht die fehlende Glasschaden-Aussage einer vollständigen Antwort.

### Answer Correctness

| Aussage | Erwarteter Wert | Antwortwert | Quelle | Ergebnis |
|---|---|---|---|---|
| Kontakt | Lara Neumann | Lara Neumann | CRM | PASS |
| Zuordnung | Lara Neumann Motor 2026 | Lara Neumann Motor 2026 | CRM | PASS |
| Policennummer | `TEST-KFZ-2026-1001` | `TEST-KFZ-2026-1001` | CRM | PASS |
| Deckungsart | Partial Coverage | Partial Coverage | CRM | PASS |
| Selbstbeteiligung | 150 EUR | 150 EUR | CRM | PASS |
| allgemeine Glasschaden-Regel | grundsätzlich unter Teilkasko, vorbehaltlich Bedingungen | nicht ausgegeben | PDF/RAG | FAIL |

`5 / 6 = 0,8333`

### Citation Validity

**FAIL für die tatsächliche Combined-Response**, weil kein PDF-Zitat vorhanden ist. Die vier CRM-Quellen sind valide, erfüllen aber die Dokumentanforderung nicht.

Die unabhängige Ground-Truth-Prüfung der PDF-Dateien und Seiten ist **PASS**. Dieser Befund ersetzt kein Zitat in der tatsächlichen Antwort.

### Source Completeness

- Für tatsächlich ausgegebene Fakten: `22/22 = 1,0`.
- Für alle sechs geforderten Antwortbestandteile: `5/6 = 0,8333`, da der PDF-gestützte Deckungsbestandteil fehlt.

### Routing Accuracy

Erwartet `combined`, tatsächlich in beiden maßgeblichen Läufen `combined`: `1/1 = 1,0`, **PASS**.

### Vorhandene deterministische Thesis-Metrik

Zusätzlich wurde `src.evaluation.thesis_metrics.build_qa_metric_row` auf die Warm-Partial-Antwort und den nachgewiesenen Kontext angewandt:

| Metrik | Wert |
|---|---:|
| Retrieval Support Hit | 1,0 |
| Retrieval Context Precision | 0,8889 |
| Answer Exact Match | 0,0 |
| Answer Token F1 | 0,2973 |
| Citation Source Presence | 1,0 |
| Citation Support Rate | 0,625 |

`Citation Source Presence = 1,0` entsteht durch die CRM-Sources und ist deshalb kein Nachweis einer PDF-Citation.

## 10. Performance und Cold/Warm

| Stage | First/Cold-Label | Warm |
|---|---:|---:|
| HTTP | 206 | 206 |
| Status | partial | partial |
| Wall Latency | 50.432,080 ms | 47.679,596 ms |
| Diagnostics Total | 50.414,193 ms | 47.671,251 ms |
| Route Planning | 0,830 ms | 1,374 ms |
| Guardrail | 58,211 ms | 68,048 ms |
| CRM | 193,419 ms | 209,056 ms |
| Embedding separat | nicht gemeldet | nicht gemeldet |
| Lexical Retrieval | 179,639 ms | 199,479 ms |
| Vector Retrieval | 860,192 ms | 700,551 ms |
| Hybrid Merge | 0,060 ms | 0,071 ms |
| Retrieval gesamt | 1.040,889 ms | 900,713 ms |
| Reranking | 11.370,049 ms | 9.247,496 ms |
| Self-check | 30.045,660 ms | 30.050,028 ms |
| Rewrite | nicht gestartet | nicht gestartet |
| Compression | nicht gestartet | nicht gestartet |
| Answer | nicht gestartet | nicht gestartet |
| Retries | 0 | 0 |

Cold-/Warm-Verhältnis: `50.432,080 / 47.679,596 = 1,0577`. Der Warm-Lauf war 2.752,484 ms beziehungsweise 5,46 % schneller.

Test-Client-Cap: 190 Sekunden. Kein Request-Level-Timeout wurde ausgelöst. Der bestehende Stage-Timeout von 30 Sekunden wurde in `self_check` erreicht.

**Performance FAIL**, weil beide Läufe keine vollständige Combined-Antwort erzeugten. Die Runtime-Stabilisierung selbst funktioniert: Der Stage-Timeout ist begrenzt, klassifiziert und führt zu HTTP 206 statt zu HTTP 500.

## 11. Fehleranalyse

Exakter Fehler in beiden maßgeblichen Läufen:

- Error Code: `LLM_STAGE_TIMEOUT`
- Error Stage: `self_check`
- HTTP Status: `206`
- Response Status: `partial`
- Timeout Stage Budget: 30 Sekunden
- Retries: 0

Vor `self_check` waren CRM, Guardrail, Hybrid Retrieval und Reranker erfolgreich abgeschlossen.

An `self_check` verfügbarer Kontext, ohne Offenlegung interner Prompts:

- 5 Dokumentchunks;
- Chunklängen: 922, 989, 997, 975 und 993 Zeichen;
- zusammen 4.876 Zeichen Chunkinhalt;
- 5.006 Zeichen formatierter Dokumentkontext;
- Output-Limit des Checks: 8 Tokens.

Der Fehler ist daher als **Laufzeitverarbeitungsproblem**, nicht als fehlender Datenbestand klassifiziert. Rewrite, Compression und Answer wurden danach nicht mehr gestartet.

## 12. Safety, Regression und Integrität

Fokussierte Regression:

```text
47 passed, 22 warnings in 79.13s
```

Getestet wurden LLM-Runtime-Stabilisierung, Insurance Routing, CRM Orchestration, Retrieval Service, CRM MCP Tools sowie CRM API Routing, Live-EspoCRM und RAG Smoke. Die 22 Meldungen sind Dependency-Deprecation-Warnings; es gab keinen Testfehler.

| Integritätsprüfung | Vorher | Nachher | Ergebnis |
|---|---:|---:|---|
| Chroma `insurance_rag_collection` | 8551 | 8551 | PASS |
| CRM Contact | 10 | 10 | PASS |
| CRM MvaPolicy | 15 | 15 | PASS |
| CRM MvaClaim | 8 | 8 | PASS |
| PDF Git Status/Diff | sauber | sauber | PASS |

Keine Secrets wurden in die Evidence-Datei übernommen. Es gab 0 CRM-Writes, 0 Reindex-Läufe, 0 PDF-Änderungen, 0 Runtime-Konfigurationsänderungen und 0 Änderungen am produktiven Pipeline-Code für diesen Test. Der bereits vorher umfangreich geänderte Worktree wurde nicht zurückgesetzt.

## 13. Gesamturteil

| Bewertung | Ergebnis |
|---|---|
| CRM Data Retrieval | **PASS** |
| RAG Retrieval | **PASS** |
| Combined Routing | **PASS** |
| Combined Partial Contract | **PASS** |
| Complete Combined E2E | **FAIL** |
| Context Relevance | **0,6** |
| Context Sufficiency | **1,0** |
| Answer Hallucination Rate | **0,0** |
| Citation Validity | **FAIL** |
| Functional | **PARTIAL** |
| Quality | **PARTIAL** |
| Performance | **FAIL** |
| Safety and Integrity | **PASS** |
| Overall | **PARTIAL** |

## 14. Empfohlener nächster Schritt

Den bestehenden `self_check` mit unverändertem Modell, Timeout, Retrieval-k, Reranker, Guardrails und Retry-Policy profilieren und seine Implementierung optimieren. Danach exakt dieses isolierte Testskript erneut ausführen. Jede spätere Konfigurationsänderung sollte als getrennt genehmigtes Experiment gegen diese Baseline gemessen werden.

## 15. Wesentliche tatsächlich ausgeführte Befehle

```powershell
docker compose ps

Invoke-RestMethod -Uri http://localhost:8000/health | ConvertTo-Json -Depth 8

Invoke-WebRequest -Uri http://localhost:8080/ -Method Head -UseBasicParsing

Invoke-RestMethod -Uri http://localhost:11434/api/tags

docker exec mva-backend python scripts/test_mcp_crm_client.py

docker exec mva-backend python scripts/test_combined_crm_rag_glass_damage.py retrieval --output reports/.combined_crm_rag_glass_damage_retrieval_20260726_203418.json

.\.venv\Scripts\python.exe scripts/test_combined_crm_rag_glass_damage.py live --output reports/.combined_crm_rag_glass_damage_live_corrected_20260726_203418.json --backend-url http://localhost:8000 --timeout-seconds 190

docker exec mva-backend python /app/tmp/compute_existing_qa_metrics.txt

docker exec --env ESPOCRM_PUBLIC_URL=http://espocrm mva-backend python -m pytest -q tests/unit/test_llm_runtime_stabilization.py tests/unit/test_insurance_tool_routing.py tests/unit/test_crm_orchestration.py tests/unit/test_retrieval_service.py tests/unit/test_crm_mcp_tools.py tests/integration/test_crm_api_routing.py tests/integration/test_espocrm_live.py tests/integration/test_rag_smoke.py

docker exec mva-backend python /app/tmp/check_combined_integrity.py

git status --short -- data/raw/pdfs

git diff --name-only -- data/raw/pdfs
```

## 16. Testbezogene Dateien

Neu erstellt:

- `scripts/test_combined_crm_rag_glass_damage.py`
- `reports/combined_crm_rag_glass_damage_test_20260726_203418.md`
- `reports/combined_crm_rag_glass_damage_test_20260726_203418.json`

Keine produktive Datei wurde für diesen Test geändert.
