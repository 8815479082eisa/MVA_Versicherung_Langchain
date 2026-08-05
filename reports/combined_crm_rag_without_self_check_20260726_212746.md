# Combined CRM + RAG ohne Self-check - kontrollierter A/B-Test

Zeitpunkt: 26. Juli 2026  
Repository: `C:\Users\mirae\MVA_Versicherung_Langchain_main`

## 1. Ziel und Hypothese

Der vorherige reale Combined-Test für Lara Neumann wurde mit derselben tatsächlich verwendeten Frage und denselben Datenquellen wiederholt. Die einzige beabsichtigte experimentelle Variable war:

```text
SELF_CHECK_ENABLED=false
```

Zu prüfende Hypothese:

> Der Self-check verändert in diesem konkreten Szenario nicht die fachlichen Fakten der Antwort, verhindert aber aufgrund des aktuellen `LLM_STAGE_TIMEOUT` die vollständige End-to-End-Ausführung.

Ergebnis vorweg: **Die Hypothese wird nur teilweise gestützt.** Der Self-check-Timeout verschwindet und die Pipeline erreicht die Answer-Stage. Die Answer-Stage scheitert jedoch ebenfalls, sodass keine vollständige Combined-Antwort entsteht. Die fachliche Gleichheit einer vollständigen Antwort kann deshalb nicht geprüft werden.

## 2. Exakte Testfrage

Für den wissenschaftlichen A/B-Vergleich wurde exakt der String des vorherigen autoritativen Tests verwendet:

> Lara Neumann reported windshield glass damage. Is glass damage generally covered under partial coverage motor insurance? Also provide the coverage type, specific deductible, and policy number stored for the motor policy of Lara Neumann. Separate the general document rule from the individual CRM policy data.

Die deutsche Formulierung im Auftrag ist die fachliche Entsprechung dieser Frage.

## 3. Genaue Konfigurationsänderung

Es existierte vorher kein Feature Flag. Ergänzt wurde ein minimaler zentraler Schalter:

```text
SELF_CHECK_ENABLED=true
```

Eigenschaften:

- Code-Default: `true`
- dokumentierter Default in `.env.llm.example`: `true`
- produktive `.env`: nicht geändert
- `false` überspringt nur `self_check`
- Retrieval, Reranking, Kontextübergabe, Answer-Generation und Safety bleiben aktiv
- Diagnostics:

```json
{
  "selfCheckEnabled": false,
  "selfCheckSkipped": true,
  "selfCheckSkipReason": "controlled_test_configuration",
  "stageStatus": {
    "self_check": "skipped"
  }
}
```

- Audit-Serializer enthält äquivalente snake_case-Felder.
- Der Stage wird nicht als `completed` und nicht mit einer erfundenen Laufzeit protokolliert.

Für den Test wurde im bestehenden Container ein isolierter zweiter Uvicorn-Prozess auf `127.0.0.1:8001` mit prozesslokalem `SELF_CHECK_ENABLED=false` gestartet. Das normale Backend auf Port 8000 blieb bei `true`. Nach dem Test wurde nur der isolierte Prozess beendet.

Unverändert blieben unter anderem:

| Parameter | Wert |
|---|---|
| Embedding | `BAAI/bge-m3` |
| Retrieval top-k | 8 |
| Strategie | Hybrid Retrieval |
| Reranker | `BAAI/bge-reranker-base` |
| Rerank top-k | 5 |
| Answer-Modell | `qwen2.5:7b-instruct` |
| Query Rewrite | deaktiviert |
| Self-check-Timeout | 30 s |
| Answer-Timeout | 120 s |
| Request-Timeout | 180 s |
| LLM-Retries | 1 |
| Retrieval-Retries | 1 |
| Safety | aktiviert, `enforce`, NeMo |

Answer-Prompt, Temperatur, Tokenlimits, Quellenformat, CRM-Daten, PDFs und Chroma wurden nicht verändert.

## 4. Preflight

| Prüfung | Normaler Prozess | Kontrollierter Prozess |
|---|---|---|
| Backend Health | ok | ok |
| Pipeline ready | ja | ja |
| CRM ready | ja | ja |
| Retrieval ready | ja | ja |
| LLM ready | ja | ja |
| Self-check | `true` | `false` |
| Safety | aktiv, enforce | aktiv, enforce |

Weitere Baseline:

- Chroma `insurance_rag_collection`: 8.551 Einträge
- CRM: 10 Contacts, 15 MvaPolicies, 8 MvaClaims
- PDF Git-Status: sauber
- PDF Git-Diff: leer
- bestehender Worktree: bereits vor diesem Auftrag umfangreich geändert; nichts zurückgesetzt

Während einer CPU-intensiven abschließenden Validierungsabfrage wurde `mva-backend` vorübergehend von Docker als `unhealthy` markiert, weil der auf fünf Sekunden begrenzte Container-Healthcheck dreimal auslief. Der direkte `/health`-Endpoint meldete weiterhin alle Readiness-Felder als bereit. Nach dem regulären nächsten Zyklus waren Backend, EspoCRM und Datenbank wieder `healthy`, `FailingStreak=0`.

## 5. Ground Truth

### CRM

Die Werte wurden live über die vorhandenen fünf read-only MCP-Tools erneut gelesen:

| Feld | Ground Truth |
|---|---|
| Kontakt | Lara Neumann |
| relevante Police | Lara Neumann Motor 2026 |
| Policennummer | `TEST-KFZ-2026-1001` |
| Produkt | Motor Insurance |
| Deckungsart | Partial Coverage |
| Status | Active |
| Selbstbeteiligung | 150 EUR |
| Jahresprämie | 684 EUR |
| Laufzeit | 2026-01-01 bis 2026-12-31 |

Die ebenfalls vorhandene Police `TEST-PHV-2026-1002` ist Personal Liability und für die Kfz-Frage nicht relevant.

Der Claim `TEST-CLM-2026-2001` ist ein Glass Damage Claim mit Status Under Review und stellt keine Leistungsentscheidung dar.

### RAG/PDF

Die allgemeine Ground-Truth-Abfrage lief erneut ohne Generation und ohne Reindex:

- 8 Retrieval-Kandidaten
- 5 rerankte Chunks
- 4 relevant, 1 irrelevant
- fachliche Relevanz: `4/5 = 0,8`

Für die exakte Combined-Frage wurden die fünf Kandidaten ebenfalls erneut reproduziert:

| Rang | Datei | Metadata Page | PDF-Seite | Bewertung |
|---:|---|---:|---:|---|
| 1 | `Insurance_Handbook_20103.pdf` | 10 | 11 | relevant: Windschutzscheiben und mögliche Selbstbeteiligung |
| 2 | `consumer-auto-shopping-tool.pdf` | 10 | 11 | irrelevant: nur Verbraucherfragen |
| 3 | `140_1261_e.pdf` | 19 | 20 | irrelevant: Gebäude-/Haushaltsglas |
| 4 | `240_1217_e.pdf` | 13 | 14 | relevant: konditionale EasyRepair-Werkstattregel |
| 5 | `240_1184_e.pdf` | 1 | 2 | relevant: Glasbruch unter Part comprehensive |

Kontextrelevanz der exakten Frage: `3/5 = 0,6`.

Zusätzliche direkte Deckungsnachweise:

- `publication-aut-pp-consumer-auto.pdf`, Metadata Page 7, physische PDF-Seite 8: beschädigte oder gerissene Windschutzscheiben können durch Comprehensive gedeckt sein; Selbstbeteiligungen können variieren.
- `240_1217_e.pdf`, Metadata Page 12, physische Seite 13: TK1.4 nennt Windschutz-, Seiten- und Heckscheiben.
- `240_1184_e.pdf`, Metadata Page 1, physische Seite 2: Part comprehensive nennt Glasschaden ausdrücklich.

Diese drei Seiten wurden mit Poppler neu gerendert und visuell geprüft. Die Off-by-one-Zuordnung `metadata.page + 1 = physische PDF-Seite` stimmt.

Fachliche Ground Truth: Teilkasko/Part comprehensive deckt Glasschäden und Windschutzscheibenschäden grundsätzlich. Bedingungen und Selbstbeteiligungen können variieren. Eine EasyRepair-Werkstattbindung darf nur als konditional genannt werden, wenn die entsprechende Vereinbarung gilt. Die 150 EUR aus CRM sind eine individuelle Selbstbeteiligung, keine Deckungssumme.

## 6. Tatsächliche Live-Läufe

Beide Requests liefen real über `POST /api/ask`; CRM, Retrieval, Reranking und LLM waren nicht simuliert.

### Lauf 1

| Feld | Wert |
|---|---|
| HTTP | 206 |
| Status | partial |
| Route | combined |
| Error Code | `LLM_UNAVAILABLE` |
| Error Stage | answer |
| Wall Latency | 36.619,076 ms |
| CRM | erfolgreich |
| Retrieval/Reranking | erfolgreich |
| Self-check | skipped |
| Knowledge Result | `null` |
| PDF-Sources | 0 |
| Answer-Retry | 1 |

Stage Timings:

| Stage | ms |
|---|---:|
| Route Planning | 1,300 |
| CRM | 139,943 |
| Guardrail | 78,574 |
| Retrieval | 1.244,540 |
| Reranking | 27.460,014 |
| Self-check | skipped, keine Laufzeit |
| Answer | 1.966,216, failed |

### Warm-Lauf

| Feld | Wert |
|---|---|
| HTTP | 206 |
| Status | partial |
| Route | combined |
| Error Code | `LLM_STAGE_TIMEOUT` |
| Error Stage | answer |
| Wall Latency | 138.144,851 ms |
| CRM | erfolgreich |
| Retrieval/Reranking | erfolgreich |
| Self-check | skipped |
| Knowledge Result | `null` |
| PDF-Sources | 0 |
| Answer-Retry | 1 |

Stage Timings:

| Stage | ms |
|---|---:|
| Route Planning | 0,501 |
| CRM | 136,476 |
| Guardrail | 95,569 |
| Retrieval | 706,406 |
| Reranking | 9.035,139 |
| Self-check | skipped, keine Laufzeit |
| Answer | 123.344,846, timeout |

Cold-/Warm-Verhältnis: `0,2651`. Dieser Wert bedeutet nicht, dass der erste Lauf effizienter vollständig arbeitete; er brach lediglich früher mit `LLM_UNAVAILABLE` ab.

## 7. Tatsächliche Antwort

Beide Läufe lieferten dieselbe sichere Partial-Antwort:

```text
CRM facts:
Customer: Lara Neumann (l***@example.test).
Policy TEST-PHV-2026-1002: Personal Liability, Liability, Active; term 2026-01-01 to 2026-12-31; deductible 0 EUR; annual premium 96 EUR.
Policy TEST-KFZ-2026-1001: Motor Insurance, Partial Coverage, Active; term 2026-01-01 to 2026-12-31; deductible 150 EUR; annual premium 684 EUR.
Claim TEST-CLM-2026-2001: Glass Damage, status Under Review; reported 2026-07-10; claimed amount 780 EUR; policy Lara Neumann Motor 2026.

Document evidence:
Unavailable. No coverage or claim decision was inferred from CRM data.
```

Die vier Sources waren ausschließlich CRM-Sources. Es gab keine PDF-Citation.

## 8. Fachliche Einzelprüfung

| Aussage | Erwarteter Ground Truth | Tatsächliche Antwort | Quelle | Ergebnis |
|---|---|---|---|---|
| Kontakt | Lara Neumann | Lara Neumann | CRM | PASS |
| Kfz-Zuordnung | Lara Neumann Motor 2026 | Lara Neumann Motor 2026 | CRM | PASS |
| Policennummer | `TEST-KFZ-2026-1001` | `TEST-KFZ-2026-1001` | CRM | PASS |
| Deckungsart | Partial Coverage | Partial Coverage | CRM | PASS |
| Selbstbeteiligung | 150 EUR | 150 EUR | CRM | PASS |
| allgemeine Glasdeckung | grundsätzlich unter Teilkasko | nicht ausgegeben | PDF/RAG | FAIL |
| allgemeine vs. individuelle Daten getrennt | beide Abschnitte vollständig | Dokumentabschnitt unavailable | beide | FAIL |
| nur belegte Einschränkungen | konditionale Regeln | nichts ausgegeben | PDF/RAG | nicht auswertbar |
| PDF-Citation | Datei und korrekte Seite | keine | PDF/RAG | FAIL |

## 9. Qualitätsmetriken

| Metrik | Ergebnis | Interpretation |
|---|---:|---|
| Context Relevance | 0,6 | 3 von 5 exakten Reranker-Kandidaten relevant |
| Context Sufficiency | 1,0 | alle 6 benötigten Fakten im kombinierten Kontext vorhanden |
| Full Answer Correctness | nicht auswertbar | keine Full Answer |
| Partial Answer Correctness | 0,8333 | 5 von 6 geforderten Fakten ausgegeben |
| Full Answer Hallucination Rate | nicht auswertbar | keine Full Answer |
| Partial Answer Hallucination Rate | 0,0 | 0 von 22 ausgegebenen CRM-Aussagen unbelegt |
| Citation Validity | FAIL | keine PDF-Citation |
| Full Source Completeness | nicht auswertbar | keine Full Answer |
| Partial Required-Fact Completeness | 0,8333 | PDF-Fakt fehlt |
| Emitted-Claim Source Completeness | 1,0 | alle tatsächlich ausgegebenen Fakten CRM-gestützt |
| Routing Accuracy | 1,0 | erwartet und tatsächlich `combined` |

Die Halluzinationsrate darf nicht als Full-Answer-Metrik ausgegeben werden. Der Wert 0,0 gilt ausschließlich für die tatsächlich erzeugte Partial-Antwort.

Vorhandene deterministische Thesis-Metrik auf Warm-Partial-Antwort plus aktuellem Kontext:

| Metrik | Wert |
|---|---:|
| Retrieval Support Hit | 1,0 |
| Retrieval Context Precision | 0,5556 |
| Answer Exact Match | 0,0 |
| Answer Token F1 | 0,2917 |
| Citation Source Presence | 1,0 |
| Citation Support Rate | 0,625 |

`Citation Source Presence = 1,0` entsteht durch CRM-Sources und ist keine gültige PDF-Citation.

## 10. A/B-Vergleich

Baseline: `reports/combined_crm_rag_glass_damage_test_20260726_203418.json`

| Metrik | Self-check aktiviert | Self-check deaktiviert | Differenz |
|---|---:|---:|---:|
| HTTP-Status | 206 / 206 | 206 / 206 | 0 |
| Complete E2E | FAIL | FAIL | keine |
| Failure Stage | self_check / self_check | answer / answer | Pipeline kommt weiter, bleibt aber unvollständig |
| Wall Latency First | 50.432,080 ms | 36.619,076 ms | -13.813,004 ms (-27,39 %) |
| Wall Latency Warm | 47.679,596 ms | 138.144,851 ms | +90.465,255 ms (+189,74 %) |
| Mittelwert Wall Latency | 49.055,838 ms | 87.381,964 ms | +38.326,126 ms (+78,13 %) |
| Partial Answer Correctness | 0,8333 | 0,8333 | 0 |
| Full Hallucination Rate | n/a | n/a | n/a |
| Partial Hallucination Rate | 0,0 | 0,0 | 0 |
| Citation Validity | FAIL | FAIL | keine |
| Context Relevance | 0,6 | 0,6 | 0 |
| Context Sufficiency | 1,0 | 1,0 | 0 |

Bewertung der acht Vergleichsfragen:

1. **Wurde der Self-check-Timeout beseitigt?** Ja. Der Stage wurde kontrolliert übersprungen und nicht als ausgeführt protokolliert.
2. **Wurde eine vollständige Combined-Antwort erzeugt?** Nein.
3. **Sind CRM- und PDF-Fakten weiterhin korrekt?** CRM-Ausgabe ja; PDF-Ground-Truth ja, aber PDF-Fakten wurden nicht in der Live-Antwort ausgegeben.
4. **Sind gültige PDF-Citations vorhanden?** Nein.
5. **Hat sich die Halluzinationsrate verschlechtert?** Für eine Full Answer nicht auswertbar. Für die Partial-Antwort keine Änderung.
6. **Hat sich die fachliche Antwort verändert?** Die tatsächlich ausgegebene CRM-Partial-Antwort ist unverändert; ein vollständiger Dokumentteil fehlt in beiden Varianten.
7. **Wie änderte sich die Latenz?** Stark variabel: erster Lauf 27,39 % kürzer durch frühen Fehler, Warm-Lauf 189,74 % länger; Paarmittel 78,13 % höher.
8. **Ist der Self-check fachlich redundant?** Nicht nachgewiesen. Dieser Test erzeugte auch ohne ihn keine Full Answer.

## 11. Fehleranalyse

Der kontrollierte Flag funktioniert technisch korrekt:

- `selfCheckEnabled=false`
- `selfCheckSkipped=true`
- Grund `controlled_test_configuration`
- `stageStatus.self_check=skipped`
- kein `selfCheckMs`
- Retrieval und Reranking abgeschlossen
- Answer-Stage tatsächlich gestartet
- Guardrails blieben aktiv

Der erste Answer-Aufruf scheiterte nach einem konfigurierten Retry mit `LLM_UNAVAILABLE`. Der Warm-Aufruf verbrauchte rund 123,3 Sekunden in Answer und endete mit dem unveränderten 120-Sekunden-Stage-Timeout als `LLM_STAGE_TIMEOUT`.

Damit ist der Self-check-Timeout nicht mehr die unmittelbare Fehlerstelle, aber der vollständige E2E bleibt durch die Answer-Runtime blockiert.

Die API-Diagnostics belegen den tatsächlichen Skip. Der Audit-Serializer wurde fokussiert getestet und schreibt dieselben Felder. Da beide Live-RAG-Läufe vor dem normalen erfolgreichen Audit-Abschluss eine Exception auslösten, stieg der Audit-JSONL-Zähler nicht an. Das wird nicht als erfolgreicher Audit-Record dargestellt.

Ein zusätzlicher Kontrolllauf mit `SELF_CHECK_ENABLED=true` wurde nicht ausgeführt, weil der unmittelbar vorherige Bericht zwei reproduzierbare reale Self-check-Timeouts enthält und der Auftrag diesen Lauf ausdrücklich als optional bezeichnet.

## 12. Safety, Tests und Integrität

Fokussierte Abschlussregression:

```text
99 passed, 226 warnings in 76.70s
```

Die Warnungen sind Dependency-Deprecation-Warnings. Getestet wurden:

- Default `true` und Environment-Override `false`
- Ausführung bei `true`
- ausschließlicher Skip bei `false`
- Retrieval und Answer bei `false`
- Diagnostics und Audit-Serialisierung
- aktive Query-, Context- und Answer-Safety-Checks
- CRM-only-Vertrag
- Combined HTTP-206-Partial-Vertrag
- Error-Mappings
- Routing, CRM, Retrieval, Live EspoCRM und RAG Smoke

Integrität:

| Prüfung | Vorher | Nachher | Ergebnis |
|---|---:|---:|---|
| Chroma Count | 8551 | 8551 | PASS |
| CRM Contact | 10 | 10 | PASS |
| CRM MvaPolicy | 15 | 15 | PASS |
| CRM MvaClaim | 8 | 8 | PASS |
| relevante PDF-SHA256 | Baseline | identisch | PASS |
| PDF Git-Status/Diff | sauber | sauber | PASS |
| normaler Backend-Default | `true` | `true` | PASS |

Es gab 0 CRM-Writes, 0 Reindex-Läufe, 0 PDF-Änderungen und keine Secret-Ausgabe.

## 13. Schlussfolgerung

| Bewertung | Ergebnis |
|---|---|
| CRM Data Retrieval | **PASS** |
| RAG Retrieval | **PASS** |
| Combined Routing | **PASS** |
| Complete Combined E2E | **FAIL** |
| Citation Validity | **FAIL** |
| Functional | **FAIL** |
| Quality | **PARTIAL** |
| Performance | **FAIL** |
| Safety and Integrity | **PASS** |
| Hypothesis Supported | **PARTIALLY** |
| Overall | **PARTIAL** |

Die Hypothese wird nur teilweise unterstützt:

- Unterstützt: Der Self-check-Timeout verschwindet, und die Pipeline erreicht die Answer-Stage.
- Nicht nachgewiesen: Ohne Self-check entsteht keine vollständige Antwort; er ist daher nicht als alleiniger E2E-Blocker bewiesen.
- Nicht prüfbar: Eine vollständige fachliche Antwort mit und ohne Self-check liegt nicht vor, daher kann die behauptete Faktenneutralität nicht belegt werden.

## 14. Empfehlung

Der produktive Default sollte `true` bleiben. Als getrennten nächsten Schritt die unveränderte Answer-Stage und die Ollama-Verfügbarkeit profilieren. Sobald Answer-Generation zuverlässig ist, denselben A/B-Test erneut ausführen. Erst danach auf mehrere korrekte, unvollständige, widersprüchliche und irreführende Kontexte erweitern, bevor über eine dauerhafte Self-check-Policy entschieden wird.

## 15. Wesentliche tatsächlich ausgeführte Befehle

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health | ConvertTo-Json -Depth 8

docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

docker exec mva-backend python /app/tmp/check_without_self_check_integrity.py

docker exec mva-backend python scripts/test_mcp_crm_client.py

docker exec mva-backend python scripts/test_combined_crm_rag_glass_damage.py retrieval --output reports/.combined_crm_rag_without_self_check_retrieval_20260726_212746.json

$question = 'Lara Neumann reported windshield glass damage. Is glass damage generally covered under partial coverage motor insurance? Also provide the coverage type, specific deductible, and policy number stored for the motor policy of Lara Neumann. Separate the general document rule from the individual CRM policy data.'
docker exec mva-backend python scripts/test_combined_crm_rag_glass_damage.py retrieval --output reports/.combined_crm_rag_without_self_check_exact_retrieval_20260726_212746.json --query $question

& 'C:\Users\mirae\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe' -f 13 -l 13 -singlefile -png -r 144 'data\raw\pdfs\240_1217_e.pdf' 'tmp\pdfs\without-self-check-240-1217-page-13'

docker exec -d -e SELF_CHECK_ENABLED=false mva-backend sh -c 'echo $$ > /tmp/mva-self-check-disabled.pid; exec python -m uvicorn src.main:app --host 127.0.0.1 --port 8001 > /tmp/mva-self-check-disabled.log 2>&1'

docker exec mva-backend python scripts/test_combined_crm_rag_glass_damage.py live --output reports/.combined_crm_rag_without_self_check_live_20260726_212746.json --backend-url http://127.0.0.1:8001 --timeout-seconds 190

docker exec mva-backend sh -c 'kill 5108'

docker exec mva-backend python /app/tmp/compute_without_self_check_metrics.py

docker exec --env ESPOCRM_PUBLIC_URL=http://espocrm mva-backend python -m pytest -q tests/unit/test_self_check_feature_flag.py tests/unit/test_llm_runtime_stabilization.py tests/unit/test_self_check_decision.py tests/unit/test_insurance_tool_routing.py tests/unit/test_crm_orchestration.py tests/unit/test_retrieval_service.py tests/unit/test_crm_mcp_tools.py tests/unit/test_guardrails_runtime.py tests/unit/test_safety_pii_rules.py tests/integration/test_api_boot.py tests/integration/test_crm_api_routing.py tests/integration/test_espocrm_live.py tests/integration/test_rag_smoke.py

git status --short -- data/raw/pdfs

git diff --name-only -- data/raw/pdfs
```

## 16. Für diesen Auftrag ergänzte Dateien

Produkt-/Konfigurationspfad:

- `.env.llm.example`
- `src/config/models.py`
- `src/core/runtime_diagnostics.py`
- `src/api/rag_service.py`
- `src/main.py`

Testpfad:

- `scripts/test_combined_crm_rag_glass_damage.py`
- `tests/unit/test_self_check_feature_flag.py` (neu)
- `tests/unit/test_llm_runtime_stabilization.py`
- `tests/integration/test_api_boot.py`

Berichte:

- `reports/combined_crm_rag_without_self_check_20260726_212746.md`
- `reports/combined_crm_rag_without_self_check_20260726_212746.json`

Der Worktree enthielt bereits vor diesem Auftrag Änderungen in mehreren dieser produktiven Dateien. Es wurden nur die hier beschriebenen Flag-, Diagnostics- und Testergänzungen vorgenommen; bestehende Änderungen wurden nicht zurückgesetzt.
