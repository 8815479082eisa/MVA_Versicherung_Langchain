# Faktenprüfung für Kapitel 1 „Einleitung“

Stand: 9. August 2026  
Technische Referenz: Branch `Abschluss-Arbeit`, Commit `437812f7be59ff6ae21325056d158b497bf56f2b`

## Gesamtbefund

Das Projekt implementiert einen domänenspezifischen, routing- und werkzeugbasierten RAG-Prototyp für interne Versicherungsanfragen. Er kann strukturierte Daten aus einem read-only angebundenen EspoCRM und unstrukturierte Versicherungsdokumente getrennt oder gemeinsam verarbeiten. Die vier Qualitätsdimensionen der Forschungsfrage sind technisch adressiert und jeweils zumindest teilweise evaluiert. Eine einzige, umfassende Evaluation des aktuellen Gesamtsystems am oben genannten Commit, die alle vier Dimensionen gleichzeitig nachweist, liegt jedoch nicht vor. Der aktuelle Stand darf deshalb nicht als produktionsreif oder als abschließend validiert bezeichnet werden.

## 1. Tatsächliche Problemstellung

Fachliche Anfragen im untersuchten Anwendungsszenario können sowohl individuelle Kunden-, Vertrags- oder Schadendaten als auch allgemeine Bedingungen aus Versicherungsdokumenten benötigen. Ein reiner Dokumenten-RAG-Pfad kennt keine individuellen CRM-Fakten; ein reiner CRM-Pfad darf aus Vertragsstammdaten keine allgemeine Deckung ableiten. Das System muss daher die Anfrage klassifizieren, die zulässigen Quellen auswählen, strukturierte und unstrukturierte Evidenz korrekt zusammenführen und die Antwort mit passenden Quellenbelegen versehen. Gleichzeitig muss es unbelegte oder unvollständige Antworten, Prompt-Injection, unzulässige Datenabfragen und den Umgang mit personenbezogenen Daten kontrollieren. Für den praktischen Einsatz kommen Betriebsaspekte wie Latenz, Readiness, Fehlerbehandlung, read-only Zugriffe und Auditierbarkeit hinzu.

Die Problemstellung ist im aktuellen Projekt besonders sichtbar, weil Groundedness und Vollständigkeit nicht dasselbe messen: Im Noah-Diebstahlszenario waren die tatsächlich formulierten Aussagen belegt, obwohl zwei im Kontext vorhandene Pflichten in der Antwort fehlten. Der Lauf bestand Groundedness, Citations und Safety, wurde fachlich aber als `FAIL` bewertet.

## 2. Umgesetztes Ziel

Umgesetzt wurde ein wissenschaftlicher Proof of Concept für einen internen Versicherungsassistenten. Er umfasst:

- eine FastAPI-Schnittstelle und eine React-Oberfläche,
- deterministisches Routing zwischen `retrieval_only`, `crm_only`, `combined` und `denied`,
- fünf read-only CRM-Werkzeuge für Kunden-, Policen- und Schadendaten,
- hybrides Retrieval aus BM25- und Embedding-Suche auf einer persistenten Chroma-Sammlung,
- Cross-Encoder-Reranking und einen deterministischen Filter für die zum Stichtag aktuelle Police,
- LLM-basierte Antwortgenerierung mit Dokument- und CRM-Citations,
- mehrstufige NeMo-Guardrails und deterministische Sicherheitsregeln,
- eine Groundedness-Nachprüfung sowie Audit- und Diagnosedaten,
- Komponenten-, Integrations-, Benchmark-, Sicherheits- und ausgewählte End-to-End-Tests.

Die aktuelle Konfiguration verwendet `gpt-4o-mini` für die Antwortgenerierung, `BAAI/bge-m3` für normalisierte Embeddings auf CPU und `cross-encoder/ms-marco-MiniLM-L-6-v2` für das Reranking. Self-Check, Query-Rewrite, Context-Compression und die Antwort-Vollständigkeitsprüfung sind derzeit deaktiviert; Retrieval wird erzwungen. Der Ausdruck „agentisch“ bezeichnet im aktuellen System daher vor allem die mehrstufige Orchestrierung und Werkzeugwahl. Das Top-Level-Routing erfolgt regelbasiert und nicht durch einen autonomen LLM-Agenten.

## 3. Nachgewiesene Systemkomponenten

| Komponente | Nachgewiesener aktueller Stand | Primärbeleg |
|---|---|---|
| API und Routing | FastAPI-Endpunkt `/api/ask`; regelbasierte Wahl zwischen CRM-, Dokument-, kombiniertem und gesperrtem Pfad | [`src/main.py`](../../src/main.py), [`src/core/insurance_tool_routing.py`](../../src/core/insurance_tool_routing.py) |
| CRM | EspoCRM 10.0.3/MariaDB; fünf read-only Werkzeuge; keine Schreib-, Export- oder Bulk-List-Funktionen | [`src/mcp_servers/crm_server.py`](../../src/mcp_servers/crm_server.py), [`docs/espocrm_mcp_integration.md`](../espocrm_mcp_integration.md) |
| Synthetische CRM-Daten | 10 Kontakte, 16 Policen, 8 Schäden in den aktuellen CSV-Dateien | [`data/synthetic/crm`](../../data/synthetic/crm) |
| Dokumentkorpus | 11 aktive Helvetia-PDFs, 138 Seiten und 1.585 Chroma-Chunks; Baloise-Dateien aus dem aktiven Korpus entfernt | [`reports/helvetia_reindex_summary_20260801.md`](../../reports/helvetia_reindex_summary_20260801.md) |
| Retrieval | BM25 plus Chroma-Vektorsuche, balancierte Reciprocal-Rank-Fusion, Nachbar-Chunk-Erweiterung | [`src/api/rag_service.py`](../../src/api/rag_service.py) |
| Reranking | Aktuell MiniLM Cross-Encoder auf CPU; deterministische Current-Policy-Selektion vor dem Reranking | [`reports/reranker_minilm_production_activation_20260801.md`](../../reports/reranker_minilm_production_activation_20260801.md) |
| Antwortgenerierung | Dokument- und CRM-Evidenz werden getrennt formatiert; Antworten sollen Dateiname/Seite bzw. CRM-Referenz zitieren und keine finale Schadenentscheidung treffen | [`src/api/rag_service.py`](../../src/api/rag_service.py) |
| Groundedness | `fact_aware_claim_support_v5`; aktuell geladener Schwellenwert 0,7888 aus der Kalibrierungsdatei | [`src/config/models.py`](../../src/config/models.py), [`config/groundedness_calibration.json`](../../config/groundedness_calibration.json) |
| Safety | NeMo-Input-, Kontext- und Output-Rails; Injection-/Secret-Prüfung, PII-Erkennung/Redaktion und Fail-closed-Verhalten | [`src/guardrails`](../../src/guardrails), [`config/nemo_guardrails`](../../config/nemo_guardrails) |
| Nachvollziehbarkeit | Audit-Log und Request-Diagnostik für Route, Quellen, Kandidaten, Scores, Sicherheitsentscheidungen und Laufzeiten | [`src/core/audit.py`](../../src/core/audit.py), [`src/core/request_diagnostics.py`](../../src/core/request_diagnostics.py) |

## 4. Tatsächlich durchgeführte Evaluationen

### Fachliche Korrektheit und Retrieval

Die allgemeine Thesis-Evaluation implementiert acht Metriken. Für Retrieval und Antwortkorrektheit sind dies Retrieval Support Hit Rate, Retrieval Context Precision, Exact Match und Token F1. Exact Match und Token F1 vergleichen generierte Antworten mit InsuranceQA-Referenzantworten; die Retrieval-Metriken beruhen auf lexikalischer Überlappung. Der strengere historische Lauf ohne InsuranceQA-PDFs im aktiven Korpus verarbeitete 200 QA-Fälle und erreichte Retrieval Support Hit Rate 0,9750, Retrieval Context Precision 0,6350, Exact Match 0,0000 und Token F1 0,1730. Er verwendete jedoch `qwen2.5:7b-instruct` und einen früheren Systemstand; er ist kein Ergebnis der aktuellen OpenAI-/CRM-/MiniLM-Konfiguration.

Die Reranker-Auswahl wurde zusätzlich mit kontrollierten Kandidatenmengen bewertet. Nach dem deterministischen Current-Policy-Filter erreichte MiniLM auf 64 Fällen Top-1 Accuracy 0,969, MRR@5 0,982 und nDCG@5 0,986 bei 467,8 ms Median. Das ist ein Komponentenbenchmark, keine Aussage über die Qualität vollständiger Antworten.

Der derzeit aussagekräftigste kombinierte Noah-Lauf verwendete das aktuelle Helvetia-Korpus, CRM, MiniLM, OpenAI, v5-Groundedness und deaktivierte Antwort-Vollständigkeitsprüfung. Die Antwort war bezüglich der formulierten Aussagen korrekt, ließ aber zwei von zehn fachlich erwarteten Bedingungen aus. Der Bericht weist Requirement Recall 0,80, Requirement F1 0,8889, Evidence Recall 1,00 und Context Precision 0,60 aus und bewertet den Lauf als technisch bestanden, fachlich jedoch nicht bestanden.

### Groundedness und Citations

Die allgemeine Thesis-Evaluation misst Source Presence Rate und Citation Support Rate. Citation Support ist dabei eine deterministische, lexikalische Heuristik; Source Presence prüft nur, ob Quellen zurückgegeben wurden. Die in den historischen 200-QA-Läufen gemeldete `inline_citation_coverage` ist keine robuste Satzabdeckung: Der aktuelle Implementierungscode setzt sie anhand terminaler Citation-Marker binär auf 0 oder 1.

Der aktuelle v5-Groundedness-Algorithmus prüft unter anderem Claim-Support sowie Quellen-, Seiten-, Policen- und strukturierte Faktenkonsistenz. Der Schwellenwert 0,7888 stammt aus einer erweiterten, schwach beaufsichtigten Kalibrierung. Die Ergebnisberichte weisen sehr gute technische Kennzahlen aus, zugleich sind die menschlichen Reviewer-Dateien leer und der finale Holdout ist nicht fachlich annotiert. Diese Kalibrierung ist daher nicht als unabhängig humanvalidierter Nachweis zu behandeln. Außerdem unterscheiden sich die Ergebnisdefinitionen zwischen dem experimentellen v5-Bericht und dem späteren Schwellenwertvergleich.

Im aktuellen Noah-Lauf lag Groundedness bei 0,872132 und damit über dem Schwellenwert. Citation Presence und Citation Support bestanden; die Claim-bezogene Citation Coverage lag bei 0,50. Wegen zweier ausgelassener Bedingungen betrug der Requirement Recall dennoch nur 0,80. Dies belegt, dass Groundedness die Quellenstützung vorhandener Aussagen misst, aber keine Antwortvollständigkeit garantiert.

ALCE-orientierte Citation Recall/Precision und optional MAUVE sind im Code implementiert und unit-getestet. Ein persistierter, ausgeführter ALCE-Ergebnislauf wurde im Projekt nicht gefunden.

### Sicherheit gegen missbräuchliche Eingaben

Der Sicherheitsdatensatz `thesis_safety_mix_200.jsonl` umfasst 200 gelabelte Fälle: 70 Angriffe und 130 legitime Anfragen. Gemessen werden Attack Block Rate und Benign Allow Rate. Zwei historische vollständige Läufe erreichten jeweils 1,0000 für beide Metriken, also keine False Negatives und keine False Positives in diesem Datensatz. Diese Ergebnisse belegen die Leistung auf dem festgelegten Testsatz, nicht allgemeine Sicherheit gegen unbekannte Angriffe.

Am 9. August 2026 bestanden 231 von 234 Unit-Tests. Drei Tests des offiziellen NeMo-Output-Pfads konnten wegen einer fehlenden Datei im lokalen FastEmbed-ONNX-Cache nicht erfolgreich ausgeführt werden; die Fehlermeldung entstand vor der fachlichen Assertion. Außerhalb der betroffenen Runtime-Datei bestanden 199 Tests, in ihr weitere 32. Zusätzlich bestanden 19 nicht-live Integrationstests. Der EspoCRM-Live-Smoke-Test wurde mangels in den Testprozess exportierter Zugangsdaten übersprungen.

### Praktische Einsetzbarkeit

Praktische Aspekte wurden durch Readiness-/Healthchecks, Fehler- und Partial-Response-Verhalten, read-only Zugriffskontrollen, Komponenten- und End-to-End-Latenzen, Docker-Betrieb sowie szenariobasierte Live-Tests untersucht. Der CRM-Latenzbenchmark berichtet für warme REST-Aufrufe p50 31,587 ms und p95 37,806 ms, für warme MCP-Aufrufe p50 67,748 ms und p95 102,335 ms. Der aktuelle Noah-End-to-End-Lauf benötigte rund 47,8 s; davon entfielen rund 19,0 s auf das Reranking. Damit ist die technische Durchführbarkeit gezeigt, nicht jedoch eine für Nutzende akzeptable Reaktionszeit.

Das System besitzt eine Weboberfläche, eine HTTP-API, Quellendarstellung und Auditdiagnostik. Eine formale Usability-Studie, eine Nutzerbefragung, Last-/Dauertests, Verfügbarkeits-SLAs oder ein vorab definierter zusammengesetzter Akzeptanzwert für „praktische Einsetzbarkeit“ wurden nicht gefunden. `PIPELINE_AUTHENTICATED` ist in der aktuellen Laufzeitkonfiguration `false`, obwohl der Zielnutzertyp als interner Sachbearbeiter konfiguriert ist. Produktionsreife ist daher nicht nachgewiesen.

## 5. Zuordnung Forschungsfrage → Qualitätsdimension → Evidenz

| Forschungsfrage | Qualitätsdimension | Systemkomponente | Tatsächlicher Test | Metrik | Vorhandenes Ergebnis und Einordnung |
|---|---|---|---|---|---|
| Wie kann die Architektur fachlich korrekt antworten? | Fachliche Korrektheit | Hybrid Retrieval, Current-Policy-Filter, Cross-Encoder, LLM-Generation | 200-QA-Thesis-Evaluation ohne InsuranceQA-PDFs; 64-Fall-Rerankerbenchmark; Noah-End-to-End-Szenario | Retrieval Support Hit Rate, Context Precision, Exact Match, Token F1; Top-1/MRR/nDCG; Requirement Recall/F1 | Historisch: 0,975/0,635/0,000/0,173. Aktuelle Komponente: MiniLM 0,969/0,982/0,986. Aktuelles Szenario: Recall 0,80 und fachlich `FAIL` wegen zwei ausgelassener Pflichten. |
| Wie kann sie quellenbasiert nachvollziehbar sein? | Groundedness/Citations | Citation-Prompting und -Nachbearbeitung, v5-Groundedness, Source-Metadaten | 200-QA-Evaluation; schwach beaufsichtigte v5-Kalibrierung; Noah-Szenario | Source Presence, Citation Support, Groundedness-Score/Threshold, Citation Coverage | Historisch ohne InsuranceQA-PDFs: Source Presence 0,9100, Citation Support 0,5971. Noah: Groundedness 0,872132 > 0,7888, Citation Presence/Support bestanden, Coverage 0,50. Keine abgeschlossene Humanvalidierung. |
| Wie kann sie missbräuchliche Eingaben sicher behandeln? | Sicherheit | NeMo-Rails, deterministische Injection-/Secret-/PII-Regeln, Denied-Route, Fail-closed-Fallbacks | `thesis_safety_mix_200.jsonl`; Guardrail-Unit- und Integrationstests | Attack Block Rate, Benign Allow Rate, False Negatives/Positives | Historische 200-Fall-Läufe: 1,0000/1,0000, 0 FN, 0 FP. Heutiger Unit-Stand: 231/234 bestanden; drei Output-Tests durch lokalen Modellcache blockiert. |
| Wie kann sie praktisch einsetzbar sein? | Praktische Einsetzbarkeit | FastAPI, React-UI, Docker, persistentes MCP, Readiness, Partial Responses, Audit | CRM-Latenzbenchmark; read-only Berechtigungstests; kombinierte Live-Szenarien | p50/p95, End-to-End- und Stufenlatenzen, HTTP-/Route-/Readiness-Status, fachlicher Szenarioerfolg | CRM warm im zweistelligen Millisekundenbereich; Noah E2E ca. 47,8 s und fachlich unvollständig. Technische Machbarkeit ja; Produktionsreife und Usability nicht nachgewiesen. |

## 6. Offene oder unsichere Informationen

1. Eine `literatur.bib` oder andere belastbare wissenschaftliche Literatursammlung ist im Projekt nicht vorhanden. Allgemeine Aussagen zu Versicherungsprozessen, LLM-Halluzinationen, RAG-Grenzen, Prompt-Injection, Datenschutz und Evaluationsmethoden benötigen externe Quellen.
2. „Fachliche Korrektheit“ ist nicht durch eine einzige Metrik operationalisiert. Exact Match und Token F1 erfassen Formulierungsähnlichkeit; die Szenario-Checks erfassen zusätzlich Fakten und Vollständigkeit. Ein fachlich annotierter aktueller Testkorpus und Akzeptanzgrenzen fehlen.
3. Die v5-Groundedness-Schwelle ist aktiv, obwohl die zugehörigen Berichte sie teilweise noch als experimentell beziehungsweise als nicht humanvalidiert beschreiben. Die Freigabeentscheidung und die widersprüchlichen Ergebnisdefinitionen sollten geklärt werden.
4. Citation Presence und die lexikalische Citation-Support-Heuristik reichen nicht als vollständiger Nachweis der Zitierqualität. Ein ALCE-Lauf oder eine menschliche Prüfung fehlt.
5. Für „praktische Einsetzbarkeit“ fehlen vorab definierte Akzeptanzkriterien, insbesondere für Antwortlatenz, fachliche Erfolgsquote, Verfügbarkeit und Nutzerfreundlichkeit.
6. Die aktuelle Konfiguration deaktiviert die Antwort-Vollständigkeitsprüfung, den Self-Check, Query-Rewrite und Context-Compression. Ob dies der endgültige Thesis-Versuchsstand sein soll, muss bestätigt werden.
7. Das aktive Korpus enthält nur Helvetia-Dokumente. Mindestens ein synthetischer Lara-CRM-Fall weist laut Bericht noch Baloise-Produktdaten auf. Dieser Source-/Issuer-Drift begrenzt die Zahl gültiger kombinierter Szenarien.
8. Die Dokumentation nennt an einer Stelle 15 Policen; die aktuellen CSV-Daten enthalten 16. Für die Arbeit sollte der direkt aus den Daten verifizierte Wert verwendet und die Dokumentation bereinigt werden.
9. Die Bezeichnung „agentisch“ sollte methodisch definiert werden. Das aktuelle Top-Level-Routing ist deterministisch und lexikalisch; eine uneingeschränkt autonome Planung ist nicht implementiert.
10. Die Frontend-Paketbezeichnung enthält noch `baloise-dokumenten-assistent`, obwohl der aktive Dokumentbestand Helvetia umfasst. Das ist Dokumentations-/Branding-Drift, keine Systemfunktion.

## 7. Zentrale Belege

- Aktueller Code: [`src/main.py`](../../src/main.py), [`src/api/rag_service.py`](../../src/api/rag_service.py), [`src/config/models.py`](../../src/config/models.py), [`src/core/insurance_tool_routing.py`](../../src/core/insurance_tool_routing.py)
- Evaluationsimplementierung: [`scripts/evaluation/evaluate_thesis.py`](../../scripts/evaluation/evaluate_thesis.py), [`src/evaluation/thesis_metrics.py`](../../src/evaluation/thesis_metrics.py), [`src/evaluation/security_eval.py`](../../src/evaluation/security_eval.py), [`src/evaluation/alce_metrics.py`](../../src/evaluation/alce_metrics.py)
- Historischer 200+200-Lauf: [`artifacts/test-results/test-result-20260525T235203Z-full-qa200-safety200-qwen25-7b-no-insuranceqa-pdfs/summary.md`](../../artifacts/test-results/test-result-20260525T235203Z-full-qa200-safety200-qwen25-7b-no-insuranceqa-pdfs/summary.md)
- Aktueller kombinierter Fachtest: [`reports/combined_crm_rag_noah_theft_without_completeness_20260805.docx`](../../reports/combined_crm_rag_noah_theft_without_completeness_20260805.docx), [`reports/combined_noah_theft_material_conditions_removal_20260805.md`](../../reports/combined_noah_theft_material_conditions_removal_20260805.md)
- Groundedness: [`reports/groundedness_v5_experimental_report_20260802.md`](../../reports/groundedness_v5_experimental_report_20260802.md), [`reports/groundedness_v5_threshold_078_vs_07888_20260802.md`](../../reports/groundedness_v5_threshold_078_vs_07888_20260802.md), [`reports/groundedness_annotation_gap_report.md`](../../reports/groundedness_annotation_gap_report.md)
- Retrieval/Reranking: [`reports/reranker_post_filter_benchmark_20260801.md`](../../reports/reranker_post_filter_benchmark_20260801.md), [`reports/reranker_minilm_production_activation_20260801.md`](../../reports/reranker_minilm_production_activation_20260801.md)
- Korpus und CRM: [`reports/helvetia_reindex_summary_20260801.md`](../../reports/helvetia_reindex_summary_20260801.md), [`docs/espocrm_mcp_integration.md`](../espocrm_mcp_integration.md), [`reports/espocrm_latency_benchmark.json`](../../reports/espocrm_latency_benchmark.json)

## 8. Abschließende Prüfliste für die Weiterarbeit

### Aussagen mit wissenschaftlichem Quellenbedarf

- Heterogenität und Verteilung strukturierter und unstrukturierter Informationen in Versicherungsunternehmen.
- Halluzinationen und fachliche Fehler von LLMs sowie die Möglichkeiten und Grenzen von RAG.
- Anforderungen an Claim-Level-Groundedness, Citation Correctness und Citation Completeness.
- Prompt-Injection, adversariale Eingaben, Datenschutz- und Sicherheitsanforderungen bei LLM-Anwendungen.
- Methodische Begründung der gewählten Retrieval-, Reranking-, Sicherheits- und Evaluationsmetriken.

### Fachlich zu bestätigende Punkte

- Soll der aktuelle Stand mit deaktivierter Antwort-Vollständigkeitsprüfung als finaler Versuchsstand gelten?
- Soll ausschließlich das fachlich und dokumentseitig ausgerichtete Noah-Szenario als aktueller kombinierter Fall verwendet werden, bis der Lara-Source-Drift bereinigt ist?
- Ist die Bezeichnung „interner Versicherungsassistent für Sachbearbeitende“ die beabsichtigte fachliche Zielgruppe?
- Sind die Grenzen „keine Schadenentscheidung“, „kein schreibender CRM-Zugriff“ und „kein Betrieb mit realen Kundendaten“ verbindlich?

### Noch nicht eindeutig operationalisierte Qualitätsdimensionen

- Fachliche Korrektheit: fehlende verbindliche Kombination aus Faktenkorrektheit, Vollständigkeit und Akzeptanzschwellen.
- Groundedness/Citations: fehlende menschliche Annotation und fehlende aktuelle ALCE-/Claim-Level-Gesamtauswertung.
- Sicherheit: keine Aussage über Generalisierung außerhalb der 200 gelabelten Fälle; drei aktuelle Output-Pfad-Tests sind umgebungsbedingt offen.
- Praktische Einsetzbarkeit: keine Grenzwerte für Latenz, Verfügbarkeit, Fehlerrate oder Usability und keine Nutzerstudie.

### Mit der Betreuung abzustimmende Entscheidungen

- Methodische Definition von „agentisch“ für eine Architektur mit deterministischem Top-Level-Routing.
- Einordnung der historischen 200+200-Evaluation gegenüber dem aktuellen, technisch deutlich veränderten Systemstand.
- Zulässigkeit der schwach beaufsichtigten v5-Groundedness-Kalibrierung als Ergebnis und Umfang einer noch notwendigen Humanvalidierung.
- Entscheidung, ob vor Abgabe ein neuer vollständiger Evaluationslauf auf dem finalen Commit erforderlich ist.
- Formale Definition, ab wann der Prototyp als „praktisch einsetzbar“ und nicht nur als technisch funktionsfähig gilt.
