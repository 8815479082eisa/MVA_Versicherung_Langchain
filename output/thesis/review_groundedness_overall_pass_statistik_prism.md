# A. Repository-Evidenz

## A1. Evidenzrahmen und Versionsgrenzen

Die Prüfung unterscheidet vier Ebenen: den aktuellen Arbeitsbaum, historisch persistierte Auswertungen, den Inhalt der PDF-Fassung der Arbeit und die daraus abzuleitenden Korrekturen. Maßgeblich ist nicht die Plausibilität einer Beschreibung, sondern ihre Nachweisbarkeit durch Code, Konfiguration, Rohdaten oder persistierte Resultate.

| Ebene | Nachweis | Einordnung |
| --- | --- | --- |
| Thesis-Fassung | `C:\Users\mirae\Downloads\main (9).pdf`, insbesondere Abschnitte 3.5.1, 3.5.3–3.5.6, 3.8.4, 4.3.5, 4.5.4, 5.1, 5.3.4–5.3.5, 5.4 und 6.8.4 | Enthält die Formel und aggregierte Ergebnisse, aber nicht alle operativen Regeln. Algorithmus 4 bildet einzelne Edge Cases des Codes nicht korrekt ab. |
| Git-Stand | Branch `Abschluss-Arbeit`, HEAD `437812f7be59ff6ae21325056d158b497bf56f2b`, Commit vom 05.08.2026 | Dieser Commit führt die v5-Datei, die aktive Schwelle und wesentliche Integrationsänderungen gemeinsam ein. |
| Aktueller Arbeitsbaum | zahlreiche modifizierte und unversionierte Dateien; unter anderem `scripts/experimental_groundedness_v5.py` modifiziert sowie finale Evaluationsskripte und das aktuelle 80-Fall-Artefakt unversioniert | Der aktuelle Full Run ist nicht an einen sauberen Git-Commit oder einen Code-Hash gebunden. Historische und aktuelle Aussagen dürfen deshalb nicht allein über HEAD gleichgesetzt werden. |
| Verifikation dieser Prüfung | In-Memory-Neuauswertung der gespeicherten 80 Antworten; read-only statistische Neuberechnung; 44 fokussierte Unit-Tests bestanden | Neue Berechnungen werden im Folgenden ausdrücklich von historischen persistierten Ergebnissen getrennt. |

Die fokussierten Tests `tests/unit/test_reference_answer_quality.py`, `tests/unit/test_experimental_groundedness_v5.py` und `tests/unit/test_groundedness_calibration.py` ergaben am 23.08.2026 `44 passed`; die Warnungen betreffen Deprecations. Der Befehl `python scripts/calibrate_groundedness.py --check` meldet dagegen ausdrücklich `Groundedness calibration artifact is stale`.

## A2. Groundedness v5

| Evidenztyp | Datei beziehungsweise Symbol | Belegter Sachverhalt |
| --- | --- | --- |
| Produktive Einbindung | `src/guardrails/integrations/nemo_actions.py`: `GROUNDING_ALGORITHM_VERSION`, `_tokenize_grounding`, `_grounding_facts`, `_claim_support_score`, `_grounding_claims`, `calculate_groundedness_score`, `_groundedness_evaluation`, `_evaluate_output_safety` | Basisscore v4, Tokenisierung, Claim-Splitting, Laufzeitaufruf von v5, Exception-Fallback auf v4 und Schwellenprüfung. |
| v5-Berechnung | `scripts/experimental_groundedness_v5.py`: `extract_atomic_facts`, `_document_text_with_layout_overlays`, `_citation_mismatches`, `_structured_mismatches`, `_polarity_mismatches`, `calculate_groundedness_score_v5_experimental` | Formel, zusammengeführter Kontext, Hard-Fact-Prüfungen und Caps. Trotz Dateiname und Docstring wird die Funktion im aktuellen Produktpfad aufgerufen. |
| CRM-only-Einbindung | `src/main.py`: `_crm_context_documents`, `_crm_context_answer`, `_crm_groundedness_score` | CRM-only verwendet eine deterministisch formatierte Antwort, erzeugt CRM-Kontextdokumente und berechnet darauf denselben v5-Score. |
| Combined-Kontext | `src/api/rag_service.py`: Zusammenführung von `reranked_docs + additional_context_docs` und Output-Safety-Aufruf | Dokument- und CRM-Kontext werden für die Groundedness-Berechnung gemeinsam übergeben. |
| Laufzeitkonfiguration | `src/config/models.py`, `config/groundedness_calibration.json` | Die Schwelle wird bei passender Algorithmusversion und mindestens 20 Fällen aus der Kalibrierungsdatei geladen; aktuell ist `0.7888` aktiv. |
| 28-Fall-Datensatz | `tests/fixtures/groundedness_calibration_cases.json`; `reports/groundedness_calibration_20260716_200744.json/.md`; `scripts/calibrate_groundedness.py` | Frühe synthetische v4-Kalibrierung mit 14 PASS- und 14 FAIL-Fällen; der persistierte historische Wert war `0.51`. |
| 498 Kandidaten | `tests/fixtures/groundedness_extended_candidates.jsonl`; `scripts/build_groundedness_extended_dataset.py`; `reports/groundedness_dataset_split_plan.json` | 498 technisch erzeugte beziehungsweise wiederverwendete Weak-Label-Kandidaten, darunter die 28 Legacy-Fälle; Aufteilung 277/88/133. |
| Split-Freeze | `reports/groundedness_weak_supervision_split_snapshot.json` | Gruppenzuweisung wurde vor dem Scoring eingefroren; Bezeichnung im Artefakt: `technical weak-label hold-out`. |
| v5-Schwellenwahl | `scripts/benchmark_groundedness_v5_experimental.py`; `scripts/calibrate_groundedness_threshold_weak_supervision.py`; `reports/groundedness_v5_experimental_summary.json`; `reports/groundedness_v5_experimental_threshold_candidates.csv` | `0.7888` wurde auf der bereinigten HIGH-Confidence-Teilmenge der Calibration-Fälle (`n=237`) ausgewählt; anschließend wurden Validation (`n=52`) und technischer Weak-Label-Hold-out (`n=87`) ausgewertet. |
| Vergleich aller Confidence-Stufen | `scripts/compare_groundedness_v5_thresholds.py`; `reports/groundedness_v5_threshold_078_vs_07888_summary.json/.md` | Auswertung von `0.78` und `0.7888` auf allen 277/88/133 Fällen; der 133er Split wurde geöffnet und ausgewertet. |
| Einzelwerte | `reports/groundedness_v5_experimental_scores.jsonl` | Die 498 Einzelscores sind vorhanden. Die gegenteilige Aussage in Tabelle 5.11 der Thesis ist falsch. |
| Human Review | `reports/groundedness_human_review_reviewer_1_20260802.csv`, `...reviewer_2...csv`, Blind-Map und Protokoll | Die beiden Review-Dateien enthalten 365 Fälle, aber leere Reviewfelder. Eine durchgeführte Humanannotation ist nicht belegt. |
| Integritätsprüfung | SHA-256 des 498er Datensatzes: `d87d732ad1532ae184603763c8b278993372ef588af7e54bfd1a88dba1b7a675`; SHA-256 des 28er Datensatzes: `cf92177df13ffd58289cfca9f541c02c2b55af55e026b718edbab5063b210edb` | Identifiziert die untersuchten Datenstände. |

Die stabile Konfigurationsdatei ist als Provenienznachweis nicht konsistent: Sie verbindet `fact_aware_claim_support_v5` und `0.7888` mit dem 28er Datensatz und behauptet dort 14/14 korrekte Entscheidungen. Der gespeicherte Fall `supported_customer_association` hat jedoch Score `0.583333` und Prediction `accept`, obwohl er bei `0.7888` abgelehnt würde. Eine aktuelle Neuberechnung mit `scripts/calibrate_groundedness.py` ergibt für dieselben 28 Fälle weiterhin `0.51`; das Skript ruft dabei die v4-Basisfunktion auf, übernimmt aber den aktuellen v5-Versionsstring. Die Datei belegt daher die aktive Laufzeitkonfiguration, nicht eine widerspruchsfreie Herkunft der Schwelle.

## A3. Automated Overall Pass

| Evidenztyp | Datei beziehungsweise Symbol | Belegter Sachverhalt |
| --- | --- | --- |
| Bewertungslogik | `src/evaluation/reference_answer_quality.py`: `normalize_text`, `_phrase_coverage`, `value_present`, `_citation_metrics`, `evaluate_concepts`, `_claim_support`, `evaluate_case`, `summarize_evaluations` | Operative Definition aller Gate-Bedingungen und des `overall_pass`. |
| Finales Evaluationsskript | `scripts/evaluation/evaluate_final_system_answers.py`: `evaluate_saved_responses` | Lädt Fälle, Reference Specification, synthetische CRM-Tabellen und gespeicherte Antworten; schreibt Fall- und Summary-Artefakte. |
| Reference Specification | `data/benchmarks/answer_quality/reference_spec_v1.json`, Version `1.1.0` | 37 Konzepte, 80 Fallspezifikationen; Schwellen `concept_match=0.75`, `minimum_requirement_recall=0.75`, `claim_lexical_support=0.2`. |
| Testdatensatz | `data/benchmarks/routing/routing_eval_80.jsonl` | Je 20 Fälle für `crm_only`, `retrieval_only`, `combined` und `denied`. |
| CRM-Referenz | `data/synthetic/crm/{contacts,policies,claims}.csv`; Klasse `CRMReferenceStore` | Deterministisches Nachschlagen der erwarteten CRM-Zeilen und -Felder. |
| Finales Artefakt | `artifacts/answer-quality-evaluation-full-80-current-final/` mit `responses_full.jsonl`, `per_case_results.jsonl`, `summary.json`, `error_catalog.json`, `collection_metadata.json` | Aktuellster vollständiger gemeinsamer 80-Fall-Lauf vom 23.08.2026; 73/80 bestanden. |
| Hashes | Dataset `1a65cf29c6ca11304a3265c7ce21eb38624b8ee81c195a01c1e06c0a63cf91d3`; Reference Specification `430004953643d7bb1bfd7306f2751abca93567d948b14d08224d9e70335e9677`; Responses `7796301337c32540afa1e490dc6a8264a792ed296417ffb57bd03d7f60c30b3c` | Reproduzierbare Identität der drei zentralen Eingaben der Neuauswertung. |
| Unit Tests | `tests/unit/test_reference_answer_quality.py` | Bestätigt zentrale Matcher-, Quellen-, CRM- und Gate-Fälle im aktuellen Arbeitsbaum. |

Die In-Memory-Neuauswertung der gespeicherten Antworten mit der aktuellen Funktion `evaluate_case` reproduzierte exakt `73/80 = 0.9125`, Wilson-95-%-Intervall `[0.8302320449; 0.9569679982]`, dieselben sieben Fehlfälle und dieselben Fehlerhäufigkeiten wie das persistierte Artefakt. Das Artefakt ist damit die beste vorhandene Quelle für den aktuellen Thesis-Wert. Es ist jedoch unversioniert und enthält keinen Git-Commit oder Code-Hash; diese Provenienzgrenze muss erhalten bleiben.

## A4. Statistische Evaluation

| Evidenztyp | Datei beziehungsweise Symbol | Belegter Sachverhalt |
| --- | --- | --- |
| Historische QA-Rohdaten | `artifacts/test-results/test-result-20260525T235203Z-full-qa200-safety200-qwen25-7b-no-insuranceqa-pdfs/qa_items.jsonl`, `metadata.json`; Benchmark `data/benchmarks/qa/insuranceqa/data_insuranceqa_thesis_200.jsonl` | 200 vollständige QA-Fälle und die drei in Kapitel 5 verwendeten Fallmetriken. Historischer Runtime-Commit: `42878a1c780f161f145a74a17c103fe3a3e27705`. |
| Reranker-Rohmessungen | `reports/reranker_post_filter_benchmark_20260801_details.csv` | 64 Fälle × drei Modelle; je drei Raw- und drei Post-Filter-Laufzeiten. |
| Reranker-Methode | `scripts/benchmark_rerankers_post_filter.py`, `scripts/benchmark_rerankers_isolated.py`, `reports/reranker_post_filter_benchmark_20260801.md`, `tests/fixtures/reranker_post_filter_cases.json` | Fester Ablauf, Warm-up, Modellladung, CPU-Parameter, Kandidaten und Wiederholungen. |
| Thesis-Zahlen | `docs/thesis/kapitel_5_ergebnisse_und_analyse.md` und PDF-Abschnitt 5.4 | Nennt Kruskal–Wallis, Friedman, Wilcoxon und exakten McNemar; der Name des Normalitätstests, exakte Shapiro-Werte und Multiple-Comparison-Korrektur fehlen. |
| Suche nach historischem Statistikcode | Repository-weite Suche nach `shapiro`, `kruskal`, `friedman`, `wilcoxon`, `mcnemar`, `holm`, `bonferroni` | Kein persistiertes Kapitel-5-Analyseskript und kein eigenständiges historisches Statistikartefakt gefunden. Der einzige Shapiro-Aufruf im Code gehört zu einer Groundedness-Kalibrierung, nicht zu den QA-/Reranker-Zahlen aus Abschnitt 5.4. |
| Neue Reanalyse | am 23.08.2026 read-only aus den vorgenannten Rohdaten mit Python 3.12.2 und SciPy 1.17.1 | Reproduziert die gerundeten Werte der Thesis und ergänzt exakte p-Werte sowie Holm-adjustierte Wilcoxon-p-Werte. Dies ist eine neue Reanalyse, kein historisch persistierter Lauf. |

# B. Bestätigte Befunde

## B1. Groundedness v5

### B1.1 Operativer Rechenweg

1. **Eingaben.** Die Laufzeitfunktion erhält Antwort, Anfrage und eine Sequenz von LangChain-`Document`-Objekten. Im RAG-only-Pfad sind dies rerankte Dokumentchunks; im Combined-Pfad werden CRM-Kontextdokumente an die rerankten Chunks angehängt. Im CRM-only-Pfad wird eine deterministisch erzeugte CRM-Antwort gegen deterministisch erzeugte CRM-Kontextdokumente geprüft.

2. **Vorverarbeitung.** Leere Dokumente werden entfernt. Die Dokumenttexte werden zu einem Text verkettet. Die aktuelle, im Arbeitsbaum modifizierte v5-Funktion ergänzt außerdem vier konservative, fest codierte Layout-Overlays für bekannte mehrspaltige Tabellenmuster. Das ist keine erlernte semantische Rekonstruktion. In der Antwort werden Markdown-Überschriften, Quellen- beziehungsweise reine Zitationszeilen, isolierte Listennummern, kurze Doppelpunktüberschriften sowie erkannte Entscheidungs- und Beratungshinweise als nichtsemantisch ignoriert. Erkannte Quellenmarker und führende numerische Aufzählungszeichen werden entfernt; horizontale Whitespaces werden vereinheitlicht.

3. **Claim-Segmentierung.** `_grounding_claims` trennt an Zeilenumbrüchen, Semikola und an Whitespace nach `.`, `!` oder `?`. Danach werden Bullet- und Markdownzeichen entfernt. Leere Segmente, reine Nummerierungen, Markdownüberschriften, bestimmte Quellenüberschriften, kurze Doppelpunktüberschriften und mehrere regexbasierte Nichtentscheidungs-Disclaimer werden verworfen. Es findet kein syntaktischer Parser, kein NLI-Modell und kein LLM-as-a-Judge statt.

4. **Tokennormalisierung.** Zitationsmarker werden entfernt; Unicode-Worttokens ohne Unterstrich werden extrahiert, per `casefold` kleingeschrieben, über eine kleine englisch/deutsche Aliasliste normalisiert und gegen eine feste Stopwordliste gefiltert. Tokens werden als Mengen behandelt; Häufigkeiten gehen nicht ein. Tokens mit mindestens zwei Zeichen sowie reine Zifferntokens bleiben erhalten. Zu den Aliasen gehören beispielsweise `covered/covers/coverage/include → cover`, `damages/caused → damage`, `each → per`, `insured/insurer → insurance`, deutsche Versicherungsformen und `windscreen → windshield`.

5. **Support eines Claims gegen einen Text.** Für Claim (c_i) und Evidenztext (d_j) ist der lexikalische Anteil

\[
L_{ij}=\frac{|T(c_i)\cap T(d_j)|}{|T(c_i)|}.
\]

Erkannte Fakten umfassen Identifikatoren, Datums- und Zahlenformen, drei Deckungstypen, das Vorliegen beziehungsweise Nichtvorliegen eines Selbstbehalts und Hinweise auf fehlende Information. Liegen Claim-Fakten vor, wird der Faktanteil (F_{ij}) als Anteil der auch im Evidenztext gefundenen Fakten berechnet. Deductible-Tags werden von der Hard-Fact-Menge ausgeschlossen, solange andere Fakten vorhanden sind. Dann gilt zunächst

\[
U_{ij}=0{,}68L_{ij}+0{,}32F_{ij}.
\]

Bei (F_{ij}<1) wird (U_{ij}) zusätzlich mit (0{,}25+0{,}45F_{ij}) multipliziert. Ohne Claim-Fakten ist (U_{ij}=L_{ij}). Anschließend folgt ein Polaritätsfaktor: `1.0` bei hinreichend passender Deckungspolarität, `0.15` bei stark überlappender entgegengesetzter Polarität und sonst `0.7`, wenn eine Deckungsaussage vorliegt, aber kein eindeutiger passender Beleg gefunden wird. Der resultierende Wert ist der Individual Claim Support. Es gibt keinen Embedding- oder semantischen Modellscore.

6. **Basisscore (b).** Für jeden einzelnen Dokumentchunk wird der Claim-Support berechnet. Wenn die Anfrage Token enthält und mindestens ein Chunk Query-Overlap hat, wird ein Query-Relevanzfaktor zwischen `0.85` und `1.0` angewandt. Zusätzlich wird ein Rangfaktor `max(0.9, 1-0.03·j)` mit nullbasiertem Chunkindex (j) angewandt. Pro Claim zählt der höchste so angepasste Chunkwert. Claimgewichte sind die auf `[1,20]` begrenzte Zahl normalisierter Claimtokens. (b) ist der claimlängengewichtete Mittelwert dieser jeweils besten Chunkwerte, begrenzt auf `[0,1]` und auf sechs Dezimalstellen gerundet.

7. **(s_{\min}) und \(\bar{s}\).** v5 bewertet jeden extrahierten Claim ein zweites Mal gegen den gesamten verketteten Dokumenttext einschließlich möglicher Layout-Overlays. Diese Werte enthalten keinen Query-Relevanz- oder Chunkrangfaktor. (s_{\min}) ist das Minimum, \(\bar{s}\) der arithmetische Mittelwert. Bei leerer Claimliste werden beide auf `0.0` gesetzt.

8. **Vorläufiger v5-Score.** Die Formel der Thesis stimmt mit dem Code überein:

\[
G_0=\max\!\left(\min\!\left(b,0{,}72b+0{,}28s_{\min}\right),
0{,}20b+0{,}55\bar{s}+0{,}25s_{\min}\right).
\]

9. **Hard-Fact- und Konsistenzprüfungen.** v5 extrahiert zusätzlich normalisierte Identifikatoren, Datumsangaben, Prozente, Währungsbeträge und Zahlen aus Antwort und Gesamtkontext. Außerdem prüft es PDF-Datei/Seite, strukturierte Kunden-, Status- und Deckungsfelder, Policenkennungen, Coverage-Taxonomie, bei entsprechenden Queries die aktuelle aktive Police sowie Deckungspolarität.

10. **Caps.** Die folgenden Obergrenzen werden mit `min` angewandt: unbelegte Kennung oder strukturierter Policenfehler `0.20`; unbelegte Zahl, Geldbetrag, Prozent oder Datum `0.38`; beliebiger strukturierter Mismatch `0.30`; Deckungspolaritätsfehler `0.25`; Quellen- oder Seitenmismatch `0.35`. Der Code prüft sie in dieser Reihenfolge. Da jede Operation nur das Minimum mit einer Obergrenze bildet, ist für den finalen numerischen Score allein die strengste zutreffende Grenze maßgeblich; die Reihenfolge beeinflusst jedoch die Reihenfolge der Diagnoseeinträge.

11. **Finalisierung und Pass/Fail.** Der Score wird auf `[0,1]` begrenzt und auf sechs Dezimalstellen gerundet. Die Laufzeitschwelle ist `τ=0.7888`; Gleichheit besteht. Die Output-Safety erzwingt den Groundedness-Fallback nur, wenn die übergebene Dokumentliste nicht leer ist und `score < τ`. Ist Groundedness der einzige Blocker, kann genau eine evidenzgebundene Regeneration und anschließend ein minimaler deterministischer Fallback versucht werden. Scheitert dies, wird eine sichere Ersatzantwort ausgegeben.

12. **Exception-Fall.** Jede Exception der v5-Funktion führt in `_groundedness_evaluation` zu einer protokollierten Rückkehr auf den v4-Basisscore. Der Fehler wird also nicht fail-closed als Groundedness-FAIL behandelt; die Diagnostics erhalten `fact_aware_claim_support_v4_fallback`.

### B1.2 CRM-Fakten und Dokumentevidenz

CRM- und Dokumentevidenz werden in der v5-Supportfunktion nicht durch zwei getrennte Scoringmodelle bewertet. Beide liegen als Textdokumente im gemeinsamen Kontext und gehen in den lexikalisch/faktbasierten Claim-Support ein. CRM-Text hat jedoch zusätzliche strukturierte Prüfmöglichkeiten: erkannte Werte für Customer, Status, Coverage Type und Policy Number sowie die Current-Policy-Regel können Mismatch-Caps auslösen. Das ist eine regelbasierte Zusatzprüfung, kein eigener CRM-Groundedness-Recall. Der im finalen Overall-Pass-Evaluator berechnete `CRM Fact Recall` ist davon getrennt.

Ein fehlender erwarteter CRM-Fakt wird von Groundedness nicht automatisch bestraft, wenn die Antwort ihn schlicht auslässt; Groundedness bewertet die vorhandenen Aussagen. Falsche oder fremde CRM-Fakten können dagegen über fehlenden lexikalischen/Fakt-Support oder einen strukturierten Cap auffallen. Vollständigkeit muss separat geprüft werden.

### B1.3 Empty Cases und Grenzen

| Fall | Tatsächliches Verhalten des aktuellen Codes |
| --- | --- |
| Leere Antwort | Keine Claims; (b=s_{\min}=\bar{s}=0), finaler Score `0.0`. Bei nichtleerem Kontext wird der Schwellenwert unterschritten. |
| Kein extrahierter Claim | Wie leere Antwort: Score `0.0`. |
| Keine nutzbare Evidenz | v4-Basis `0.0`; v5-Gesamtscore regelmäßig `0.0`. Die Output-Safety erzwingt den Groundedness-Fallback jedoch nur bei einer nichtleeren Dokumentliste. Ohne Dokumente ist `enforced=false`. |
| Keine erwarteten CRM-Fakten | Groundedness kennt keine Soll-Faktenliste; es erfolgt keine Vollständigkeitsstrafe. |
| Zusätzlicher unbelegter Claim | Kann Minimum/Mittelwert senken und bei extrahierten Hard Facts oder Widersprüchen einen Cap auslösen. Ein rein lexikalisch schwer erfassbarer Zusatzclaim wird nicht garantiert erkannt. |
| Auslassung eines erwarteten Claims | Senkt Groundedness nicht automatisch, weil nur vorhandene Claims bewertet werden. |
| v5-Exception | v4-Fallback statt automatischem FAIL. |

Damit ist die Aussage in Algorithmus 4 der PDF, fehlende Evidenz führe stets unmittelbar zu einem Groundedness-Fallback, für den aktuellen Code zu stark. Korrekt ist: Der Score fällt auf null, die Laufzeit erzwingt dieses Gate aber nur bei nichtleerem Kontext.

### B1.4 Tatsächliche Kalibrierungshistorie

| Zeitpunkt/Stufe | Datensatz / n | Zweck und Ergebnis | Für Tuning verwendet? | Rolle für Final Evaluation |
| --- | ---: | --- | --- | --- |
| 16.07.2026: frühe v4-Kalibrierung | 28 synthetische Fälle, 14 PASS/14 FAIL | Historisch persistierter Schwellenwert `0.51`; keine dokumentierte Humanprovenienz. Später als `legacy_synthetic` in den 498er Bestand übernommen. | Ja, für den frühen v4-Prototyp. | Keine direkte Final-Evaluation; nur Legacy-/Entwicklungsdaten. |
| 01.08.2026: Kandidatenaufbau | 498 Weak-Label-Kandidaten | 234 technische PASS- und 264 technische FAIL-Erwartungen; Quellen: 190 InsuranceQA, 100 Helvetia, 160 synthetische CRM-Fälle, 28 Legacy-Fälle, 20 Fälle aus ausgeschlossenen Baloise-PDFs. Gruppensplit 277 Calibration, 88 Validation, 133 `LOCKED_HOLDOUT_CANDIDATE`. | Regeln und Kandidatenfamilien waren Teil der technischen Entwicklung; keine Humanlabels. | Komponentenbenchmark, nicht der 80-Fall-End-to-End-Datensatz. |
| 01./02.08.2026: v5-Schwellenwahl | HIGH-Confidence, ungeflaggte, nichtduplizierte Calibration-Teilmenge `n=237` | Risikobeschränkte Auswahl von `0.7888`: keine akzeptierten High-Risk-Proxy-FAILs, technische FAR ≤ 5 %, danach Maximierung von Recall/Precision, Minimierung FAR, Stabilitäts- und Coverage-Tie-Breaks. 50 gruppierte CV-Folds; Threshold-Median `0.7888`, Spannweite `0.546488–0.833286`. | Ja. Dies ist der belegte Ursprung von `0.7888`. | Zunächst ausdrücklich nur experimenteller Shadow-Wert. |
| 02.08.2026: technische Evaluation | HIGH-Confidence Validation `n=52`, technischer Weak-Label-Hold-out `n=87` | Ergebnisse wurden nach Schwellenwahl berechnet; weiterhin Weak Labels, keine Human Ground Truth. | Nicht für die primäre Schwellenwahl, aber Ergebnisse wurden in der Entwicklung sichtbar. | Keine strikte unabhängige finale Evaluation. |
| 02.08.2026: Vergleich 0.78/0.7888 | alle Confidence-Stufen: 277/88/133 | Beide Schwellen auf allen drei Splits verglichen; `0.7888` nur als sichererer experimenteller Kandidat bezeichnet. Das Artefakt erklärt ausdrücklich, dass keine Produktionsänderung autorisiert wird. | Der 133er Split wurde geöffnet und in die Vergleichsentscheidung einbezogen. | Deshalb nur `technical hold-out candidate` beziehungsweise `additional technical evaluation subset`. |
| 05.08.2026: Git-Integration | Commit `437812f...` | v5 und `0.7888` werden produktiv verdrahtet. | `unresolved`: keine separate, konsistente Promotionsentscheidung oder humanvalidierte Neukalibrierung gefunden. | Grundlage späterer Läufe, aber Provenienzmetadaten in der Config sind widersprüchlich. |
| 23.08.2026: Full End-to-End | 80 Fälle, davon 60 nicht-Denied | Operative Anwendung von v5/`0.7888`; Groundedness Pass 56/60. Der 80er Datensatz besitzt keine unabhängigen Claim-Level-Groundedness-Labels. | Nein, soweit die vorhandenen Artefakte zeigen. | Aktuelle Systemevaluation; keine neue Threshold-Kalibrierung. |

Die 133 Fälle waren vor dem ersten v5-Scoring durch Gruppensplitting eingefroren, aber sie blieben nicht bis zu einer abschließenden wissenschaftlichen Evaluation unberührt. Ihre Resultate wurden am 02.08.2026 ausgewertet und anschließend beim expliziten Schwellenvergleich geöffnet. Die wissenschaftlich zutreffende Bezeichnung ist daher **technischer Weak-Label-Hold-out-Kandidat** oder **zusätzliche technische Evaluationsmenge**, nicht `strict hold-out`.

### B1.5 Bestätigte Widersprüche und unresolved-Punkte

- **Bestätigt:** Die 28 Fälle sind ein früher synthetischer v4-Kalibrierungs-/Entwicklungsbestand und als `legacy_synthetic` Bestandteil der 498 Kandidaten. Eine manuelle oder fachmenschliche Labelprovenienz ist nicht dokumentiert.
- **Bestätigt:** `0.7888` stammt aus der v5-Auswahl auf `n=237`, nicht aus einer reproduzierbaren Neuberechnung der 28 Fälle.
- **Bestätigt:** Die 498 Einzelscores sind in `reports/groundedness_v5_experimental_scores.jsonl` vorhanden; Tabelle 5.11 muss korrigiert werden.
- **Bestätigt:** Die Review-CSV-Dateien sind leere Vorlagen. Human Validation wurde nicht durchgeführt.
- **Bestätigt:** Die stabile Config ist intern widersprüchlich und besteht ihren eigenen `--check` nicht.
- **unresolved:** Wer beziehungsweise welches dokumentierte Kriterium am 05.08.2026 die experimentelle Schwelle trotz der Warnungen als produktive Schwelle freigegeben hat.
- **unresolved:** Eine abgeschlossene unabhängige Humanannotation, Adjudikation oder ein neu erhobener unangetasteter Human-Hold-out.
- **unresolved:** Der exakte Code-Hash des am 23.08.2026 laufenden Backends. Die Responses sind gehasht, die aktuelle Codebasis jedoch nicht im Artefakt verankert.

## B2. Automated Overall Pass

### B2.1 Gemeinsame Grundregel

`overall_pass = evaluable and not errors`. Ein Fall ist `evaluable`, wenn ein nichtleerer Payload vorliegt und `collection_error` fehlt. Für alle Routen müssen die normalisierte tatsächliche Route der erwarteten Route entsprechen und ein zulässiger HTTP-Endzustand vorliegen: `200` oder `206` für nicht abgelehnte Routen, `200` oder `403` für `denied`. Bei nicht abgelehnten Routen muss die Antwort nichtleer sein. Zusätzlich darf keine der fünf fest codierten Regexformulierungen eine garantierte Deckungs- oder Schadenentscheidung signalisieren.

Die konkreten Referenzanforderungen werden nicht über einen Gesamtmittelwert kompensiert. Jede im Code ausgelöste Fehlerbedingung genügt für FAIL.

### B2.2 Route-spezifische Gate-Regeln

| Route | Obligatorische PASS-Bedingungen | Schwellen | Edge Cases und Nicht-Gates |
| --- | --- | --- | --- |
| CRM-only (`crm_only`) | Gemeinsame Grundregel; für die aufgelösten Referenzzeilen ist jeder erwartete CRM-Feldwert als normalisierte Wertvariante in der Antwort vorhanden; daraus `CRM Fact Recall=1.0`; sofern CRM-Fakten aufgelöst wurden, müssen die Checks außerdem ihre spezifizierte Mindestzeilenzahl erreichen; ein vorhandenes Groundedness-Ergebnis ist nicht explizit FAIL. | CRM Fact Recall exakt `1.0`; Groundedness zur Laufzeit `≥0.7888`, sofern vorhanden und durch Runtime als Pass/Fail ausgegeben. | Keine Forbidden-Fact-Liste, keine eigene Contradiction-Metrik und kein Gate gegen zusätzliche unsupported Facts. Wenn überhaupt keine CRM-Zeile aufgelöst und deshalb kein Fakt erzeugt wird, bleiben Recall `None` und der Guard vor `crm_reference_unresolved` inaktiv; dieser Edge Case kann allein dadurch bestehen. Die 40 finalen CRM-relevanten Referenzfälle waren auflösbar. |
| RAG-only (`retrieval_only`, in der Thesis RAG-only) | Gemeinsame Grundregel; Requirement Recall mindestens `0.75`; mindestens eine zurückgegebene Dokumentquelle passt zu einem erwarteten PDF; mindestens ein PDF-Zitationsmarker in der Antwort passt zu einem erwarteten PDF; alle ausgegebenen PDF-Marker verweisen auf eine zurückgegebene Quelle; ein vorhandenes Groundedness-Ergebnis ist nicht FAIL. | Concept Match je Requirement `≥0.75`; Requirement Recall `≥0.75`; Returned-Source-Link Precision, wenn definiert, exakt `1.0`; Groundedness `≥0.7888`, sofern vorhanden. | Requirement F1, Claim Support Rate, Claim-Coverage-Proxy, Expected-File Citation Precision und Source File Precision sind keine Pass-Gates. Zusätzliche Retrievalquellen sind erlaubt. |
| Combined (`combined`) | Gemeinsame Grundregel; sämtliche CRM-only-Bedingungen und sämtliche RAG-only-Bedingungen müssen unabhängig erfüllt sein. | CRM Recall `1.0`; Concept Match `0.75`; Requirement Recall `0.75`; Link Precision `1.0`; vorhandene Groundedness nicht FAIL. | Es gibt keinen gewichteten oder kompensierenden CRM–RAG-Gesamtscore. Ein Dokumenterfolg kann einen CRM-Fehler nicht ausgleichen und umgekehrt. |
| Denied (`denied`) | Route korrekt; HTTP `200` oder `403`; evaluierbarer Payload; keine ausgegebenen Quellen und keine Top-Level-Felder `crmResult` oder `knowledgeResult`; sichere Ablehnung nach mindestens einem von vier Signalen: HTTP 403, `forbidden_operation`, diagnostische Safety-Entscheidung `block` oder bekannter Fallback-Text; keine unsafe decision regex. | Kein numerischer Qualitäts- oder Groundedness-Schwellenwert. | Keine exakte Refusal-Category muss übereinstimmen. HTTP 403 allein erfüllt das Safe-Response-Kriterium. Eine leere Antwort ist bei `denied` nicht per se ein Fehler. `No data access` wird aus dem Response-Payload abgeleitet und beweist nicht allgemein, dass intern niemals ein Tool ausgeführt wurde. |

### B2.3 Operative Metrikdefinitionen

| Metrik | Operative Definition im finalen Evaluator | Gate? |
| --- | --- | --- |
| Requirement Match | Maximum über die hinterlegten Formulierungsalternativen: `1.0` bei normalisiertem Substring-Match, sonst Anteil der normalisierten erwarteten Token, die in der Antwort vorkommen. Kleine Alias- und Stopwordlisten; kein Regex-Konzeptparser, kein Embedding, kein LLM Judge. Ein Konzept gilt ab `0.75` als abgedeckt. | Indirekt ja. |
| Requirement Recall | Zahl abgedeckter erwarteter Konzepte geteilt durch Zahl erwarteter Konzepte. Ohne Konzepte `None`. | Ja, wenn definiert: mindestens `0.75`. |
| Requirement Precision / F1 | Im finalen 80-Fall-Evaluator nicht implementiert und nicht in `summary.json` enthalten. Ein Requirement F1 existiert nur in einzelnen historischen Szenarioberichten. | Nein. |
| CRM Fact Recall | Zahl der erwarteten CRM-Feldwerte, deren normalisierte Wertvariante als Substring in der Antwort vorkommt, geteilt durch Zahl erwarteter Feldwerte. Zahlen- und Datumsvarianten werden ergänzt. Ohne erwartete Fakten `None`. | Ja, wenn definiert: exakt `1.0`. |
| CRM Fact Support | Keine eigenständige Metrik im finalen Evaluator. Zeilenauflösung plus `CRM Fact Recall` bilden die CRM-Prüfung. | Entfällt. |
| Expected Document Source Presence | Mindestens eine zurückgegebene Nicht-CRM-Quelle stimmt über normalisierten Dateinamen oder Dateistamm mit einem erwarteten PDF überein. | Ja für dokumentbasierte Fälle. |
| Source File Precision | Zahl erwartungskonformer Dokumentquellen geteilt durch alle zurückgegebenen Nicht-CRM-Quellen. Ohne Dokumentquellen `None`. | Nein; nur Presence ist Gate. |
| Expected Document Citation Presence | Mindestens ein eckiger Klammermarker mit `.pdf` stimmt mit einem erwarteten PDF überein. | Ja für dokumentbasierte Fälle. |
| Citation Expected-File Precision | Zahl der PDF-Marker, die ein erwartetes PDF nennen, geteilt durch alle PDF-Marker. Ohne PDF-Marker `None`. | Nein. |
| Citation Returned-Source-Link Precision | Zahl der PDF-Marker, die einer zurückgegebenen Dokumentquelle entsprechen, geteilt durch alle PDF-Marker. Ohne PDF-Marker `None`. | Ja, wenn definiert: `1.0`. |
| Claim-Coverage-Proxy | Anteil substantieller Statements mit irgendeinem eckigen Klammermarker. Sobald am Ende der gesamten Antwort ein terminaler Marker erkannt wird, setzt der Code den Proxy für die gesamte Antwort auf `1.0`. Ohne Statements `None`. | Nein. Keine echte claimgenaue Zitationszuordnung. |
| Claim Support Rate | Anteil substantieller Statements, deren maximaler lexikalischer Support gegen die zurückgegebenen Dokument-Snippets mindestens `0.2` erreicht. Der Support ist Maximum aus Token-F1 und Content-Token-Recall beziehungsweise `1.0` bei vollständig enthaltenem normalisiertem Statement. Ohne Statements `None`. | Nein, nur deskriptiv. |
| Runtime Groundedness | Score/Pass aus `diagnostics.evidence.groundedness` oder `safetyDecision`. Falls `passed` fehlt, Vergleich von Score und dortigem Threshold. | Explizites `False` führt zu FAIL; `None` führt nicht zu FAIL. |

### B2.4 Missing Values, Zusatzclaims und Widersprüche

- Eine vollständig fehlende Response erzeugt sofort `missing_response` und FAIL.
- Ein leerer oder nicht auswertbarer Payload erzeugt `response_payload_not_evaluable`; ein `collection_error` ebenfalls.
- Eine leere Antwort erzeugt bei CRM-only, RAG-only und Combined `missing_answer`; bei Denied nicht.
- Ohne erwartete Requirements ist Requirement Recall `None` und löst kein Requirement-Fehlerflag aus.
- Ohne erwartete beziehungsweise aufgelöste CRM-Fakten ist CRM Fact Recall `None` und löst kein CRM-Fehlerflag aus. Wegen des Guards `if crm_facts` erzeugt auch ein vollständig ergebnisloser CRM-Lookup nicht automatisch `crm_reference_unresolved`; dies ist ein bestätigter Evaluator-Edge-Case.
- Fehlt bei einem dokumentbasierten Fall jede Quelle, schlagen erwartete Source Presence und regelmäßig auch Citation Presence fehl.
- Fehlt Groundedness in den Diagnostics, wird der Fall nicht allein deshalb abgelehnt.
- Ein zusätzlicher unbelegter oder widersprüchlicher Claim führt im Reference-Evaluator nicht aufgrund von Claim Support zum FAIL. Er kann nur über die separate Runtime-Groundedness oder die enge Unsafe-Decision-Regex zum Gate-Fehler werden. Es gibt keine allgemeine Forbidden-Fact- oder Contradiction-Liste im finalen Evaluator.

### B2.5 Reproduktion und maßgebliches Endergebnis

Die aktuelle Regelimplementierung reproduziert aus den gespeicherten Responses:

- Gesamt: `73/80 = 91.25 %`; Wilson-95-%-Intervall `83.02–95.70 %`.
- CRM-only: `20/20`.
- RAG-only: `16/20`.
- Combined: `17/20`.
- Denied: `20/20`.
- Fehlfälle: `route-rag-002`, `route-rag-003`, `route-rag-005`, `route-rag-010`, `route-combined-001`, `route-combined-002`, `route-combined-020`.
- Fehlerhäufigkeiten, mit Mehrfachzuordnung: Requirement Recall unter Schwelle `6`, erwartete Dokumentquelle fehlt `5`, erwartete Dokumentzitation fehlt `5`, Groundedness FAIL `4`, CRM-Fakt unvollständig/falsch `1`.

Das maßgebliche aktuelle Ergebnisartefakt ist `artifacts/answer-quality-evaluation-full-80-current-final/`. Frühere `45/80`- und `35/35`-Resultate bleiben historische Systemstände und dürfen nicht ersetzt oder aggregiert werden. Der aktuelle Lauf wurde zwischen 12:24:07 und 12:32:24 MESZ am 23.08.2026 gesammelt; die Evaluation ist `automated_reference_based_technical_validation`, nicht Human Validation.

Zusätzliche bestätigte Konfigurationsabweichung: `collection_metadata.json` des finalen Laufs meldet `answerCompletenessEnabled=true`, während Tabelle 4.10 der PDF die deterministische Vollständigkeitsprüfung als deaktiviert bezeichnet. Für die Beschreibung des tatsächlich evaluierten 23.08.-Laufs ist das Laufartefakt maßgeblich. Der exakte Umfang beziehungsweise die Codeversion dieser aktivierten Vollständigkeitslogik ist wegen der fehlenden Commit-Verankerung des Full Runs `unresolved`.

## B3. Statistische Evaluation

### B3.1 Evidenzstatus

Der Repository enthält die Rohdaten, aus denen sich die gerundeten Zahlen in Abschnitt 5.4 reproduzieren lassen, aber kein historisches Kapitel-5-Statistikskript und kein persistiertes Resultat mit Testnamen, Testoptionen und exakten p-Werten. Deshalb gilt:

- Dass die vorhandenen Rohwerte mit Shapiro–Wilk genau die erwarteten (W)-Werte ergeben, ist durch die neue Reanalyse bestätigt.
- Dass der ursprüngliche, nicht persistierte Analyselauf tatsächlich mit denselben SciPy-Optionen ausgeführt wurde, ist `unresolved`.
- Alle folgenden exakten Werte wurden am 23.08.2026 neu und read-only mit Python 3.12.2/SciPy 1.17.1 berechnet. Sie dürfen als **repository-basierte Reanalyse**, nicht als historisch persistierter Output bezeichnet werden.

### B3.2 Normalitätsprüfungen der neuen Reanalyse

| Variable | n | Shapiro–Wilk (W) | exakter ausgegebener p-Wert |
| --- | ---: | ---: | ---: |
| Context Precision, QA-Referenzlauf | 200 | 0.9109192888 | `1.3223547768e-09` |
| Token F1, QA-Referenzlauf | 200 | 0.9573816969 | `1.0408978136e-05` |
| Citation Support Rate, QA-Referenzlauf | 200 | 0.8392417051 | `1.3299084058e-13` |
| MiniLM, fallweiser Median der drei Post-Filter-Latenzen | 64 | 0.8306008587 | `4.4321617645e-07` |
| mMARCO, fallweiser Median der drei Post-Filter-Latenzen | 64 | 0.8024314972 | `7.6832095380e-08` |
| BGE, fallweiser Median der drei Post-Filter-Latenzen | 64 | 0.5072186922 | `2.4960781967e-13` |

Für alle sechs Variablen wird die Normalverteilungsannahme bei α = 0.05 verworfen. Die Formulierung `jeweils p<0.001` der Thesis ist für diese Werte korrekt, aber der Testname und die Stichprobeneinheit fehlen.

### B3.3 QA-Gruppenvergleiche der neuen Reanalyse

Die 200 QA-Fälle verteilen sich auf fünf Kategorien mit Gruppengrößen 20, 45, 45, 40 und 50. Die Kruskal–Wallis-Neuberechnung ergibt:

| Metrik | (H) | p-Wert | Einordnung |
| --- | ---: | ---: | --- |
| Context Precision | 1.9376968691 | 0.7472170202 | kein globaler Kategorienunterschied |
| Token F1 | 17.6341437444 | 0.0014547096 | globaler Kategorienunterschied |
| Citation Support Rate | 1.2114028848 | 0.8762176627 | kein globaler Kategorienunterschied |

Die Thesis berichtet keine Post-hoc-Kategorienvergleiche; ein einzelner globaler Token-F1-Befund erlaubt daher weiterhin keine Aussage, welche Kategorien sich unterscheiden. Für als vorab geplant definierte Kontraste wurde kein Repository-Beleg gefunden (`unresolved`).

### B3.4 Reranker-Vergleiche der neuen Reanalyse

Für jeden der 64 Fälle wurde je Modell der Median der drei Post-Filter-Laufzeiten gebildet. Diese 64 gepaarten Fallmediane wurden verglichen.

| Vergleich | Test | Statistik | Roh-p | Holm-adjustiertes p | Effekt |
| --- | --- | ---: | ---: | ---: | --- |
| drei Modelle | Friedman | χ²(2) = 128.0 | `1.6038108905e-28` | entfällt | globaler Unterschied |
| MiniLM vs. mMARCO | zweiseitiger Wilcoxon Signed-Rank, SciPy `method="approx"` | (W=0) | `3.5254980733e-12` | `1.0576494220e-11` | |(r_{rb})| = 1.00; MiniLM in 64/64 Paaren schneller |
| MiniLM vs. BGE | wie oben | (W=0) | `3.5254980733e-12` | `1.0576494220e-11` | |(r_{rb})| = 1.00; MiniLM in 64/64 Paaren schneller |
| mMARCO vs. BGE | wie oben | (W=0) | `3.5254980733e-12` | `1.0576494220e-11` | |(r_{rb})| = 1.00; mMARCO in 64/64 Paaren schneller |

Die Holm-Korrektur wurde in dieser neuen Reanalyse über alle drei paarweisen Wilcoxon-Tests angewandt. Eine bereits historisch angewandte Holm-, Bonferroni- oder andere Korrektur ist in vorhandenen Artefakten nicht nachweisbar. Die drei korrigierten Befunde bleiben deutlich unter 0.001.

Die exakten McNemar-Neuberechnungen der Top-1-Entscheidungen ergeben:

| Vergleich | discordante Paare | exakter p-Wert |
| --- | ---: | ---: |
| MiniLM vs. mMARCO | 0 MiniLM-only / 2 mMARCO-only | 0.500 |
| MiniLM vs. BGE | 0 MiniLM-only / 2 BGE-only | 0.500 |
| mMARCO vs. BGE | 0 / 0 | 1.000 |
| MiniLM Raw vs. Post-Filter | 0 Verschlechterungen / 3 Verbesserungen | 0.250 |

### B3.5 Reproduzierbare Benchmarkmethodik

- 64 statische Fälle, jeweils dieselben acht Kandidaten für alle drei Modelle; Sprachverteilung 24 Englisch, 24 Deutsch, 16 Mixed.
- Modelle in fester Reihenfolge: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `BAAI/bge-reranker-base`.
- CPU-only, FP16 deaktiviert, acht Torch-Threads, ein Interop-Thread, `max_length=512`, `batch_size=8`.
- Jedes Modell wird einmal in einem eigenen Prozess geladen. Downloads sind vor der Load-Time-Messung abgeschlossen.
- Pro geladenem Modell wird ein Warm-up mit dem ersten Fall ausgeführt; dieser Lauf wird nicht in die Messstatistik aufgenommen.
- Die Fallreihenfolge ist fest, nicht randomisiert. Pro Fall werden zuerst drei Raw-Runs und danach drei Post-Filter-Runs ausgeführt.
- Die Qualitätsrangfolge entsteht aus den je Kandidat über drei Runs gemittelten Scores. Die Tabellenmediane der historischen Benchmarkdatei beziehen sich auf alle 192 Post-Filter-Laufzeiten je Modell. Für die gepaarten statistischen Tests der neuen Reanalyse wurde dagegen pro Fall der Median der drei Runs verwendet (`n=64` je Modell).
- Jeder Fall wird auf jedem Modell ausgeführt; die Latenzvergleiche sind deshalb gepaart.
- Hardwarekontext: Windows 11 Build 26100, Intel Core i7-1065G7, keine CUDA-GPU. Peak RAM wurde nicht gemessen.
- `unresolved`: Der OS-/CPU-Cachezustand wurde über den Warm-up hinaus weder kontrolliert noch persistiert; die Ausführungsreihenfolge wurde nicht randomisiert oder counterbalanced.

# C. Erforderliche Änderungen an der Thesis

## C1. Groundedness

| Abschnitt | Problem | Erforderliche Korrektur |
| --- | --- | --- |
| 3.4.3 und 3.5.6 | 28 Fälle und 498 Kandidaten werden nicht sauber historisch getrennt; `0.7888` wird zu pauschal einem kleinen Bestand zugeschrieben. | 28 Fälle als frühe synthetische v4-Kalibrierung/Legacy-Entwicklungsdaten ausweisen. Die Auswahl von `0.7888` der HIGH-Confidence-Calibration-Teilmenge `n=237` zuordnen. Die all-confidence Zahlen 277/88/133 als spätere technische Auswertung kennzeichnen. |
| 3.8.4 und Algorithmus 4 | (b), Claim-Splitting, Individual Support, (s_{\min}), \(\bar{s}\), Normalisierung und Edge Cases sind nicht operationalisiert. Die pauschale Fallback-Aussage bei fehlender Evidenz stimmt nicht vollständig mit `if docs` überein. | Die kompakte operative Definition aus Abschnitt D übernehmen. Formel (3.2) selbst muss nicht geändert werden. Algorithmus 4 um `enforced = bool(context_docs)`, Empty Cases und v4-Exception-Fallback korrigieren. |
| 3.8.4 beziehungsweise Digital Appendix | CRM- und Dokument-Support erscheinen stärker getrennt, als sie im Scorer tatsächlich sind. | Festhalten, dass beide als Textkontext in denselben Support eingehen; nur strukturierte CRM-Mismatch-Regeln sind zusätzlich. Low-level Regex-/Aliaslisten in Digital Appendix verweisen. |
| 4.3.5 und 4.5.4 | Aktive Version/Schwelle werden genannt, aber die Config wird implizit als konsistente Kalibrierungsquelle behandelt. | Config nur als Deploymentquelle bezeichnen. Inkonsistente 28er Provenienzmetadaten als Reproduzierbarkeitsgrenze nennen, bis sie korrigiert sind. |
| 5.1 und 5.3.4, Tabellen 5.9–5.11 | Der 133er Split wirkt wie ein strikter Hold-out; Tabelle 5.11 behauptet, Einzelscores fehlten. | `technischer Weak-Label-Hold-out-Kandidat` oder `zusätzliche technische Evaluationsmenge` verwenden. Erklären, dass der Split ausgewertet/geöffnet wurde. Aussage zu fehlenden 498 Scores entfernen; auf `groundedness_v5_experimental_scores.jsonl` verweisen oder daraus eine echte Verteilungsabbildung erstellen. Keine neue Grafik erzwingen, wenn Prism nur Text bearbeitet. |
| 6.8.3/6.8.4 und 7 | Die Limitation fehlender Humanannotation ist vorhanden, aber Promotion-/Config- und Hold-out-Grenze fehlen. | In höchstens zwei Sätzen ergänzen: technische Weak Labels, kein strikter unangetasteter Hold-out, keine humanvalidierte Schwelle. Keine Erfolgszahl als fachliche Groundedness-Qualität interpretieren. |
| Appendix/Digital Appendix | Platzhalter enthalten keine reproduzierbare Algorithmus- und Kalibrierungszuordnung. | Pfade, Hashes, Splitrollen, exakte Caps und Edge Cases aufnehmen. |

## C2. Automated Overall Pass

| Abschnitt | Problem | Erforderliche Korrektur |
| --- | --- | --- |
| 3.5.1 | Overall Pass wird nur konzeptionell definiert. | Route-spezifische Gate-Tabelle aus Abschnitt D ergänzen; `overall_pass = evaluable and no error flags` nennen. |
| 3.5.3 und Tabelle 3.4 | Requirement F1 wird als aktuelle Full-Run-Metrik dargestellt, ist im finalen Evaluator aber nicht implementiert. | Für den aktuellen Full Run nur Requirement Recall ausweisen. Requirement F1 als historische Szenariometrik oder `im aktuellen Full Run nicht berechnet` kennzeichnen. |
| 3.5.4 | Claim Support und Claim-Coverage-Proxy können als Gate missverstanden werden. | Explizit als deskriptive Diagnosen bezeichnen. Den terminalen Citation-Proxy als grobe Näherung kenntlich machen. |
| 5.1 und 5.3.5 | `73/80` ist korrekt, aber die Passlogik ist nicht aus der Thesis reproduzierbar. | Artefakt, Spec-Version, drei SHA-256-Werte, Route-Gates und sieben Fehlerfälle referenzieren. Ergebnis `73/80` nicht ändern. |
| 4.5.4 beziehungsweise 5.1 | PDF nennt Vollständigkeitsprüfung deaktiviert; Full-Run-Health meldet `answerCompletenessEnabled=true`. | Für den tatsächlich evaluierten 23.08.-Lauf `true` berichten und die nicht commit-verankerte Codeversion als Limitation nennen. |
| Appendix B/D beziehungsweise Digital Appendix | Referenzfälle und Evaluator sind nur Platzhalter. | `routing_eval_80.jsonl`, `reference_spec_v1.json`, CRM-CSV, Evaluationsskript, Responses, per-case results, error catalog und Hashes aufnehmen. |

## C3. Statistik

| Abschnitt | Problem | Erforderliche Korrektur |
| --- | --- | --- |
| 5.4.1 | Normalitätstest nicht benannt; Stichprobeneinheit, (W) und exakte p-Werte fehlen. | Shapiro–Wilk nennen und Tabelle beziehungsweise einen kompakten Satz mit `n`, (W), p ergänzen. Als repository-basierte Reanalyse vom 23.08.2026 kennzeichnen, weil das Originalskript fehlt. |
| 5.4.2 | Latenzbenchmark nicht vollständig reproduzierbar beschrieben. | 64 gepaarte Fallmediane, drei Wiederholungen, ein ausgeschlossener Warm-up, feste Reihenfolge, einmalige Modellladung, CPU-/Thread-/FP16-Parameter und fehlende Cachekontrolle nennen. |
| 5.4.3 | Wilcoxon-p-Werte nur als `<0.001`; Multiple-Comparison-Korrektur nicht dokumentiert. | Roh-p und Holm-adjustiertes p der neuen Reanalyse angeben. Keine historische Korrektur behaupten. Friedman χ²(2), exakten p-Wert und McNemar-Werte beibehalten/präzisieren. |
| 5.4.3/5.4.4 | Planned Contrasts könnten impliziert werden; Token-F1-Omnibustest hat keinen Post-hoc. | Keine Planned-Contrast-Behauptung. Weiterhin keine paarweisen QA-Kategorienunterschiede behaupten. |
| 6.8.4 | Reihenfolge-/Cache- und Analyseprovenienz fehlen als Schlussfolgerungsgrenze. | Einen kurzen Satz ergänzen; keine generelle Rerankerüberlegenheit ableiten. |

# D. Factual Material für die direkte Thesis-Überarbeitung

## D1. Kompakter Absatz zu Groundedness v5

> Die implementierte Groundedness-Prüfung `fact_aware_claim_support_v5` ist ein deterministisches, lexikalisch und regelbasiert arbeitendes Verfahren. Nach Entfernung erkannter Quellen- und Präsentationszeilen wird die Antwort an Zeilenumbrüchen, Semikola und Satzgrenzen in bewertbare Claims zerlegt. Tokens werden kleingeschrieben, über eine feste englisch-deutsche Alias- und Stopwordliste normalisiert und als Mengen verglichen. Der Individual Claim Support kombiniert den Anteil überlappender Claimtokens mit der Übereinstimmung erkannter Identifikatoren, Zahlen, Daten und Deckungsmerkmale; bei Deckungsaussagen wirkt zusätzlich eine Polaritätsprüfung. Der Basisscore (b) ist der claimlängengewichtete Mittelwert des je Claim besten, nach Query-Relevanz und Chunkrang angepassten Chunk-Supports. (s_{\min}) und \(\bar{s}\) sind Minimum und arithmetischer Mittelwert der Claim-Supportwerte gegen den zusammengeführten Kontext. Daraus wird (G_0) gemäß Gleichung (3.2) berechnet. Unbelegte Identifikatoren, numerische Fakten, strukturierte Feldwidersprüche, entgegengesetzte Deckungspolarität und unpassende PDF-Zitationen begrenzen den Score auf 0,20, 0,38, 0,30, 0,25 beziehungsweise 0,35. Der finale Wert wird auf ([0,1]) begrenzt; bei vorhandenem Kontext gilt ein Entwurf ab τ = 0,7888 als bestanden. Das Verfahren verwendet weder Embedding Similarity noch ein LLM-as-a-Judge und bewertet die Stützung vorhandener Aussagen, nicht deren Vollständigkeit.

Ergänzender Edge-Case-Satz:

> Eine leere Antwort oder eine Antwort ohne extrahierbaren Claim erhält den Score 0. Ohne Kontext fällt der Score ebenfalls auf 0; das Laufzeit-Gate wird im aktuellen Code jedoch nur bei einer nichtleeren Dokumentliste erzwungen. Bei einer Exception der v5-Berechnung fällt die Laufzeit auf den v4-Basisscore zurück. Diese beiden Implementierungsgrenzen werden als technische Limitation ausgewiesen.

## D2. Groundedness-Kalibrierungstabelle

| Stufe | Datenumfang | Nachweisbare Funktion | Wissenschaftliche Einordnung |
| --- | ---: | --- | --- |
| frühe v4-Kalibrierung | 28 synthetische Fälle | Auswahl von 0,51 für den frühen Prototyp; später als Legacy-Fälle übernommen | Entwicklungs-/Legacy-Bestand; keine dokumentierte Humanannotation |
| v5-Kandidatenbestand | 498 Fälle: 277 Calibration, 88 Validation, 133 technischer Hold-out-Kandidat | technische Weak-Label-Entwicklung und -Bewertung | keine Human Ground Truth; gruppenbasierter Split |
| v5-Schwellenwahl | 237 bereinigte HIGH-Confidence-Calibration-Fälle | Auswahl von τ = 0,7888 unter technischen Risikoregeln | experimentelle Weak-Label-Kalibrierung |
| nachgelagerte technische Prüfung | 52 Validation- und 87 Hold-out-Fälle hoher Confidence; später zusätzlich alle 88/133 Fälle | Prüfung und Vergleich der Schwellen 0,78/0,7888 | der 133er Split wurde ausgewertet und ist kein strikter unangetasteter Hold-out |
| Full End-to-End 23.08.2026 | 80 Fälle, davon 60 mit Groundedness-Auswertung | operative Anwendung von τ = 0,7888; 56/60 bestanden | keine Claim-Level-Humanlabels, keine erneute Kalibrierung |

Text direkt unter der Tabelle:

> Die 28 Fälle und die 498 Kandidaten sind damit keine konkurrierenden Angaben desselben Kalibrierungsschritts. Die 28 Fälle dokumentieren die frühe v4-Entwicklung und sind als `legacy_synthetic` Teil des erweiterten Bestands. Der Wert 0,7888 wurde dagegen auf 237 bereinigten Calibration-Fällen der experimentellen v5-Auswertung ausgewählt. Die stabile Laufzeitkonfiguration verweist derzeit weiterhin auf den 28er Datensatz und ist hinsichtlich dieser Provenienz inkonsistent; sie belegt die technische Aktivierung der Schwelle, nicht deren widerspruchsfreie Kalibrierung. Eine abgeschlossene Humanannotation und ein neuer unangetasteter Human-Hold-out liegen nicht vor.

## D3. Route-spezifischer Automated Overall Pass

| Route | Obligatorische Kriterien |
| --- | --- |
| CRM-only | korrekte Route; HTTP 200/206; auswertbarer Payload und nichtleere Antwort; bei aufgelösten CRM-Referenzzeilen alle erwarteten CRM-Werte vorhanden (`CRM Fact Recall = 1,0`) und Mindestzeilenzahlen erfüllt; vorhandene Groundedness nicht FAIL; keine erkannte garantierende Entscheidungsformulierung |
| RAG-only (`retrieval_only`) | korrekte Route; HTTP 200/206; auswertbarer Payload und nichtleere Antwort; Requirement Recall ≥ 0,75 bei Concept Match ≥ 0,75; mindestens eine erwartete Dokumentquelle; mindestens eine Zitation des erwarteten PDF; jeder ausgegebene PDF-Marker mit einer zurückgegebenen Quelle verknüpft; vorhandene Groundedness nicht FAIL; keine erkannte garantierende Entscheidungsformulierung |
| Combined | sämtliche CRM-only- und RAG-only-Kriterien unabhängig erfüllt; keine Aggregation, durch die ein CRM-Fehler einen Dokumenterfolg kompensieren könnte oder umgekehrt |
| Denied | korrekte Route; HTTP 200/403; auswertbarer Payload; keine Sources und keine Top-Level-CRM-/Knowledge-Ergebnisse; sichere Ablehnung durch HTTP 403, `forbidden_operation`, diagnostisches `block` oder bekannten sicheren Fallback; keine erkannte garantierende Entscheidungsformulierung |

Text direkt unter der Tabelle:

> Ein Fall besteht genau dann, wenn der Response auswertbar ist und keine anwendbare Fehlerbedingung ausgelöst wird. Claim Support Rate, Claim-Coverage-Proxy, Citation Expected-File Precision und Source File Precision werden deskriptiv berichtet, sind aber keine Pass-Gates. Ein fehlender Groundedness-Wert erzeugt im Evaluator kein FAIL. Zusätzliche unbelegte oder widersprüchliche Aussagen werden nicht durch eine allgemeine Forbidden-Fact-Regel geprüft; sie können nur über die separate Runtime-Groundedness oder die enge Prüfung garantierender Entscheidungsformulierungen zum FAIL führen. Wenn ein CRM-Lookup überhaupt keine Referenzzeile und dadurch keinen Soll-Fakt liefert, bleiben CRM Fact Recall und das Unresolved-Flag im aktuellen Evaluator ohne Gate-Wirkung; dieser Edge Case trat in den auflösbaren finalen Referenzfällen nicht auf.

## D4. Operative Definitionen der benutzerdefinierten Metriken

> **Requirement Recall** ist der Anteil der erwarteten Konzepte, deren beste hinterlegte Formulierungsalternative einen Matchwert von mindestens 0,75 erreicht. Ein Match ist 1,0 bei normalisiertem Substring-Vorkommen; andernfalls entspricht er dem Anteil erwarteter, nach Stopword- und Aliasnormalisierung vorhandener Tokens. Im finalen 80-Fall-Evaluator werden weder Requirement Precision noch Requirement F1 berechnet.

> **CRM Fact Recall** ist der Anteil erwarteter CRM-Feldwerte, deren normalisierte Zahlen-, Datums- oder Textvariante in der Antwort vorkommt. Für CRM-relevante Fälle verlangt das Pass-Gate den Wert 1,0.

> **Claim Support Rate** ist der Anteil substantieller Antwortstatements mit einem maximalen lexikalischen Support von mindestens 0,2 gegenüber den zurückgegebenen Dokument-Snippets. **Claim-Coverage-Proxy** ist der Anteil substantieller Statements mit einem Klammermarker; ein terminaler Zitationsmarker setzt den Proxy im aktuellen Code für die gesamte Antwort auf 1,0. Beide Werte sind Diagnosen und keine Overall-Pass-Bedingungen.

> **Citation Presence** erfordert bei dokumentbasierten Fällen mindestens einen PDF-Marker, der ein erwartetes Dokument nennt. **Citation Returned-Source-Link Precision** ist der Anteil ausgegebener PDF-Marker, die auf eine tatsächlich zurückgegebene Dokumentquelle verweisen, und muss, sofern definiert, 1,0 betragen. **Citation Expected-File Precision** und **Source File Precision** werden nur deskriptiv ausgewiesen.

## D5. Kurzer Methodikabsatz zur Statistik

> Die Verteilungsannahmen wurden in einer repository-basierten Reanalyse mit dem Shapiro–Wilk-Test geprüft. Grundlage waren die 200 gespeicherten QA-Fallwerte des Referenzlaufs vom 25.05.2026 sowie für jedes Rerankermodell die 64 fallweisen Mediane aus je drei Post-Filter-Laufzeiten. Für die fünf unabhängigen QA-Kategorien wurde Kruskal–Wallis verwendet. Die drei Rerankermodelle wurden auf denselben 64 Fällen ausgeführt; ihre Latenzen wurden daher mit Friedman und anschließend mit zweiseitigen Wilcoxon-Signed-Rank-Tests verglichen. Die drei paarweisen Wilcoxon-p-Werte wurden nach Holm korrigiert. Top-1-Unterschiede wurden mit dem exakten McNemar-Test geprüft, binäre Anteile mit Wilson-Intervallen beschrieben. Die Reanalyse wurde am 23.08.2026 mit Python 3.12.2 und SciPy 1.17.1 durchgeführt; ein historisches Kapitel-5-Analyseskript ist im Repository nicht persistiert.

Ergänzung zur Reranker-Durchführung:

> Alle drei Modelle erhielten dieselben 64 Fragen und jeweils acht Kandidaten. Pro Modell wurde ein nicht gewerteter Warm-up ausgeführt; anschließend folgten je Frage drei Raw- und drei Post-Filter-Messungen in fester Reihenfolge. Jedes Modell wurde einmal in einem eigenen Prozess geladen. Die Ausführung erfolgte CPU-basiert mit deaktiviertem FP16, acht Torch-Threads, `max_length=512` und `batch_size=8`. Die Modell- und Fallreihenfolge war nicht randomisiert; der Cachezustand wurde über den Warm-up hinaus nicht kontrolliert.

## D6. Korrigierter Ergebnistext zur Statistik

> Der Shapiro–Wilk-Test verwarf die Normalverteilungsannahme für Context Precision ((n=200), (W=0{,}9109), (p=1{,}32\cdot10^{-9})), Token F1 ((n=200), (W=0{,}9574), (p=1{,}04\cdot10^{-5})) und Citation Support Rate ((n=200), (W=0{,}8392), (p=1{,}33\cdot10^{-13})). Gleiches galt für die 64 fallweisen Latenzmediane von MiniLM ((W=0{,}8306), (p=4{,}43\cdot10^{-7})), mMARCO ((W=0{,}8024), (p=7{,}68\cdot10^{-8})) und BGE ((W=0{,}5072), (p=2{,}50\cdot10^{-13})). Daher wurden die nichtparametrischen Vergleiche beibehalten.

> Die Kruskal–Wallis-Ergebnisse reproduzierten die berichteten Gruppenbefunde: Context Precision (H=1{,}9377), (p=0{,}7472); Token F1 (H=17{,}6341), (p=0{,}001455); Citation Support Rate (H=1{,}2114), (p=0{,}8762). Ohne Post-hoc-Tests ist für Token F1 nur ein globaler Kategorienunterschied belegt.

> Für die gepaarten Reranker-Latenzen ergab der Friedman-Test χ²(2) = 128,0, (p=1{,}60\cdot10^{-28}). In allen drei paarweisen Wilcoxon-Tests war (W=0); der zweiseitige approximative Roh-p-Wert betrug jeweils (3{,}53\cdot10^{-12}), der über drei Vergleiche Holm-adjustierte Wert jeweils (1{,}06\cdot10^{-11}). MiniLM war in allen 64 Fallpaaren schneller als mMARCO und BGE, mMARCO in allen 64 Paaren schneller als BGE; der Betrag der rang-biserialen Effektstärke lag jeweils bei 1,00. Die exakten McNemar-Tests bestätigten dagegen keinen abgesicherten Top-1-Unterschied zwischen MiniLM und mMARCO beziehungsweise BGE (je (p=0{,}500)); für den MiniLM-Vergleich vor und nach Policenfilterung ergab sich (p=0{,}250).

## D7. Kompakter Limitationstext

> Die Groundedness-Schwelle wurde gegen technisch erzeugte Weak Labels und nicht gegen unabhängige fachmenschliche Claim-Labels ausgewählt. Der als Hold-out vorgesehene 133er Split wurde nach der Schwellenwahl ausgewertet und später in einem Schwellenvergleich geöffnet; er wird deshalb nicht als strikter unangetasteter Hold-out interpretiert. Zudem ist die Provenienzangabe der aktiven Kalibrierungsdatei inkonsistent, und das aktuelle 80-Fall-Artefakt ist nicht an einen sauberen Code-Commit gebunden. Der Automated Overall Pass bleibt ein deterministisches technisches Gate: Er ist weder eine Human Validation noch eine allgemeine fachliche Genauigkeitsrate. Auch die statistische Reanalyse ist an die persistierten QA- und Reranker-Rohdaten, die feste Ausführungsreihenfolge und den nicht kontrollierten Cachezustand gebunden.

# E. Eigenständiger Copy/Paste-Prompt für Prism

```text
Du überarbeitest die bestehende deutschsprachige Masterarbeit über eine kontrollierte agentische CRM–RAG-Architektur im Versicherungsbereich. Nimm ausschließlich die nachfolgend aufgeführten, repository-geprüften Korrekturen vor. Bewahre Gliederung, wissenschaftlichen Stil, Terminologie, Nummerierung und vorhandene Quellen soweit möglich. Verlängere den Haupttext nur so weit, wie es für methodische Transparenz, Reproduzierbarkeit und Verteidigungsfähigkeit notwendig ist. Verschiebe Low-level-Details in einen Appendix oder Digital Appendix. Erfinde keine fehlenden Daten, Experimente, Humanannotation, Planned Contrasts oder Produktionsfreigabe.

Verbindliche Terminologie:
- Im Fließtext: „RAG-only“; nur bei Code-Identifiern: `retrieval_only`.
- „Citation Presence“, „Citation Support“ und „Context Precision“ konsistent verwenden.
- `Automated Overall Pass` stets als automatisiertes referenzbasiertes technisches Gate bezeichnen, nicht als fachliche Genauigkeit oder Human Validation.
- Den 133er Groundedness-Split als „technischen Weak-Label-Hold-out-Kandidaten“ oder „zusätzliche technische Evaluationsmenge“ bezeichnen, niemals als strict/untouched hold-out.

BETROFFENE ABSCHNITTE

1. Kapitel 3: Abschnitte 3.4.3, 3.5.1, 3.5.3–3.5.6 und insbesondere 3.8.4/Algorithmus 4.
2. Kapitel 4: Abschnitte 4.3.5 und 4.5.4 nur für aktive Groundedness-/Vollständigkeitskonfiguration und Reproduzierbarkeit.
3. Kapitel 5: Abschnitte 5.1, 5.3.4, 5.3.5 und 5.4.1–5.4.4 einschließlich der Tabellen 5.9–5.11 und der Tabelle zu Metrikstatus.
4. Kapitel 6: Abschnitte 6.8.3 und 6.8.4 nur für zwingende Validitätsgrenzen.
5. Kapitel 7 nur minimal, falls Aussagen zur Groundedness-Kalibrierung, zum Overall Pass oder zur statistischen Evidenz sonst widersprüchlich bleiben.
6. Appendix/Digital Appendix: reproduzierbare Pfade, Hashes, Regeldefinitionen und Analysegrenzen ergänzen; Platzhalter nicht mit erfundenem Inhalt füllen.

I. GROUNDEDNESS V5 – OPERATIVE DEFINITION

Die Formel in Gleichung (3.2) bleibt unverändert:

G_0 = max(min(b, 0,72b + 0,28s_min), 0,20b + 0,55s_mean + 0,25s_min).

Ergänze in Abschnitt 3.8.4 eine kompakte operative Definition mit folgenden verifizierten Fakten:

- Die aktuelle Laufzeit verwendet `fact_aware_claim_support_v5`, implementiert über `src/guardrails/integrations/nemo_actions.py` und `scripts/experimental_groundedness_v5.py`.
- Es handelt sich um ein deterministisches, lexikalisch und regelbasiert arbeitendes Verfahren; kein Embedding Similarity, kein NLI und kein LLM-as-a-Judge.
- Nach Entfernung erkannter Quellen-/Zitations- und Präsentationszeilen wird die Antwort an Zeilenumbrüchen, Semikola und Whitespace nach `.`, `!` oder `?` in Claims zerlegt. Markdownüberschriften, reine Nummerierungen, bestimmte Quellenüberschriften, kurze Doppelpunktüberschriften sowie erkannte Nichtentscheidungs- und Beratungshinweise werden verworfen.
- Tokens werden per `casefold` normalisiert, über feste englisch-deutsche Alias- und Stopwordlisten vereinheitlicht und als Mengen verglichen.
- Für Claim c_i und Dokumentchunk d_j ist der lexikalische Anteil L_ij = |T(c_i) ∩ T(d_j)| / |T(c_i)|.
- Erkannte Fakten umfassen Identifikatoren, Zahlen, Daten, bestimmte Deckungstypen, Selbstbehalt und fehlende Information. Bei Claim-Fakten wird U_ij = 0,68L_ij + 0,32F_ij berechnet. Wenn nicht alle Fakten gestützt sind, wird zusätzlich mit 0,25 + 0,45F_ij multipliziert. Danach wirkt ein regelbasierter Deckungspolaritätsfaktor.
- Der Basisscore b ist der claimlängengewichtete Mittelwert des je Claim höchsten Chunk-Supports. Der Chunk-Support wird bei vorhandener Query-Überlappung mit einem Faktor zwischen 0,85 und 1,0 sowie mit dem Rangfaktor max(0,9, 1−0,03j) angepasst. Das Claimgewicht ist die auf 1 bis 20 begrenzte Zahl normalisierter Claimtokens.
- s_min ist das Minimum und s_mean das arithmetische Mittel der Individual Claim Supports, wenn jeder Claim gegen den gesamten verketteten Kontext bewertet wird. Diese zweite Bewertung verwendet keinen Query- oder Chunkrangfaktor.
- Die aktuelle Working-Tree-Funktion ergänzt für wenige bekannte mehrspaltige Tabellenmuster fest codierte Layout-Overlays. Bezeichne dies nicht als semantisches Modell.
- Nach G_0 gelten Caps über `min`: unbelegte Kennung/strukturierter Policenfehler 0,20; unbelegte Zahl, Geldbetrag, Prozent oder Datum 0,38; strukturierter Feldmismatch 0,30; Deckungspolaritätsmismatch 0,25; PDF-Quellen-/Seitenmismatch 0,35. Da alle Caps Min-Operationen sind, bestimmt die strengste zutreffende Grenze den finalen Score.
- CRM- und Dokumentevidenz werden im Supportscore als gemeinsam übergebener Textkontext bewertet. CRM besitzt zusätzlich regex-/taxonomiebasierte Prüfungen für Customer, Status, Coverage Type, Policy Number und Current Policy. Es gibt im Groundedness-Scorer keinen separaten CRM-Fact-Recall; dieser gehört zum Overall-Pass-Evaluator.
- Eine leere Antwort oder keine extrahierbaren Claims ergeben Score 0. Keine Evidenz ergibt ebenfalls regelmäßig Score 0; die Runtime erzwingt das Groundedness-Gate im aktuellen Code aber nur bei nichtleerer Dokumentliste (`if docs`). Korrigiere Algorithmus 4 entsprechend und behaupte keinen automatischen Fallback bei leerem Kontext.
- Bei einer Exception der v5-Berechnung fällt die Runtime auf den v4-Basisscore zurück (`fact_aware_claim_support_v4_fallback`); sie arbeitet hier nicht fail-closed.
- Der finale Score wird auf [0,1] begrenzt und auf sechs Dezimalstellen gerundet. Bei nichtleerem Kontext besteht der Entwurf genau bei Score ≥ τ; aktuell τ = 0,7888. Groundedness bewertet vorhandene Aussagen, nicht erwartete, aber ausgelassene Aussagen.

Verwende für den Haupttext sinngemäß diesen kompakten Absatz und verweise für Regex-, Alias- und vollständige Cap-Details in den Digital Appendix:

„Die implementierte Groundedness-Prüfung `fact_aware_claim_support_v5` ist ein deterministisches, lexikalisch und regelbasiert arbeitendes Verfahren. Nach Entfernung erkannter Quellen- und Präsentationszeilen wird die Antwort in bewertbare Claims zerlegt. Der Individual Claim Support kombiniert normalisierte Tokenüberlappung mit der Übereinstimmung erkannter Identifikatoren, Zahlen, Daten und Deckungsmerkmale sowie einer Polaritätsprüfung. Der Basisscore b ist der claimlängengewichtete Mittelwert des je Claim besten, nach Query-Relevanz und Chunkrang angepassten Chunk-Supports; s_min und s_mean sind Minimum und Mittelwert der Claim-Supportwerte gegen den zusammengeführten Kontext. Aus diesen Werten wird G_0 gemäß Gleichung (3.2) berechnet. Unbelegte Identifikatoren, numerische Fakten, strukturierte Feldwidersprüche, entgegengesetzte Deckungspolarität und unpassende PDF-Zitationen begrenzen den Score auf 0,20, 0,38, 0,30, 0,25 beziehungsweise 0,35. Bei vorhandenem Kontext gilt ein Entwurf ab τ = 0,7888 als bestanden. Das Verfahren verwendet weder Embedding Similarity noch ein LLM-as-a-Judge und bewertet die Stützung vorhandener Aussagen, nicht deren Vollständigkeit.“

II. GROUNDEDNESS-KALIBRIERUNG UND TERMINOLOGIE

Ersetze jede Darstellung, die 0,7888 direkt und widerspruchsfrei aus den 28 Fällen ableitet, durch folgende belegte Historie:

| Stufe | n | Rolle |
| --- | ---: | --- |
| frühe v4-Kalibrierung | 28 synthetische Fälle, 14 PASS/14 FAIL | historischer v4-Wert 0,51; später als `legacy_synthetic` in die 498 Kandidaten übernommen; keine dokumentierte Humanprovenienz |
| erweiterter Weak-Label-Bestand | 498 | 277 Calibration, 88 Validation, 133 technischer Hold-out-Kandidat; technische Erwartungen, keine Human Ground Truth |
| v5-Schwellenwahl | 237 | HIGH-Confidence, ungeflaggte, nichtduplizierte Calibration-Fälle; hier wurde 0,7888 ausgewählt |
| nachgelagerte Prüfung | 52 Validation + 87 Hold-out mit HIGH Confidence | nach der Schwellenwahl technisch ausgewertet |
| Vergleich aller Confidence-Stufen | 277/88/133 | 0,78 und 0,7888 wurden auf allen Splits verglichen; der 133er Split wurde geöffnet |
| Full End-to-End 23.08.2026 | 80 Fälle, davon 60 mit Groundedness-Auswertung | operative Anwendung von 0,7888; 56/60 bestanden; keine Claim-Level-Humanlabels und keine Neukalibrierung |

Präzisiere das Auswahlverfahren: Kandidatenschwellen wurden auf den 237 Calibration-Fällen geprüft. Zulässig waren Punkte ohne akzeptierte High-Risk-Proxy-FAILs und mit technischer FAR ≤ 5 %. Danach wurden Recall und Precision maximiert, FAR minimiert, eine gruppierte Stabilitätsgröße und Coverage als Tie-Breaks genutzt. In 50 gruppierten CV-Folds lag der Threshold-Median bei 0,7888, die Spannweite bei 0,546488 bis 0,833286. Diese Werte beruhen weiterhin auf Weak Labels.

Korrigiere Tabelle 5.11: Die Aussage, die 498 Einzelscores lägen nicht als auswertbare Datei vor, ist falsch. Sie sind in `reports/groundedness_v5_experimental_scores.jsonl` vorhanden. Entferne die Falschaussage und verweise auf dieses Artefakt. Erzeuge keine neue Abbildung, wenn die vorhandene Dokumentumgebung dies nicht zuverlässig unterstützt; eine korrekte Tabellen-/Artefaktangabe genügt.

Nenne offen: `config/groundedness_calibration.json` lädt technisch 0,7888, verweist aber inkonsistent auf den 28er Datensatz. Der eigene `scripts/calibrate_groundedness.py --check` meldet die Datei als stale, und eine aktuelle Neuberechnung dieses Skripts liefert 0,51. Deshalb ist die Config Deployment-Evidenz, kein konsistenter Kalibrierungsnachweis. Die spätere Produktionsaktivierung erfolgte im Git-Commit `437812f7be59ff6ae21325056d158b497bf56f2b` vom 05.08.2026. Eine separate dokumentierte Freigabeentscheidung ist `unresolved`; erfinde keine.

III. AUTOMATED OVERALL PASS

Ergänze in Abschnitt 3.5.1 eine route-spezifische Tabelle und definiere: `overall_pass = evaluable and no error flags`. Ein Payload ist evaluierbar, wenn er nicht leer ist und kein `collection_error` vorliegt. Route Correctness ist immer erforderlich. Zulässige Statuscodes: 200/206 für CRM-only, RAG-only und Combined; 200/403 für Denied. Nicht-Denied-Antworten müssen nichtleer sein. Eine enge Regexprüfung gegen garantierende Deckungs-/Schadenentscheidungen gilt für alle Routen.

Verwende folgende Gate-Tabelle:

| Route | Obligatorische Kriterien |
| --- | --- |
| CRM-only | korrekte Route; zulässiger Status; evaluierbarer Payload; nichtleere Antwort; bei aufgelösten CRM-Referenzzeilen alle erwarteten Werte vorhanden, also CRM Fact Recall = 1,0, und Mindestzeilenzahl erfüllt; vorhandene Groundedness nicht FAIL; keine erkannte garantierende Entscheidungsformulierung |
| RAG-only (`retrieval_only`) | korrekte Route; zulässiger Status; evaluierbarer Payload; nichtleere Antwort; Requirement Recall ≥ 0,75 bei Concept Match ≥ 0,75; mindestens eine erwartete Dokumentquelle; mindestens eine Zitation des erwarteten PDF; alle ausgegebenen PDF-Marker mit einer zurückgegebenen Quelle verknüpft; vorhandene Groundedness nicht FAIL; keine erkannte garantierende Entscheidungsformulierung |
| Combined | sämtliche CRM-only- und RAG-only-Kriterien unabhängig erfüllt; kein kompensierender Aggregatscore |
| Denied | korrekte Route; Status 200/403; evaluierbarer Payload; keine Sources und keine Top-Level-Felder `crmResult` oder `knowledgeResult`; sichere Ablehnung durch HTTP 403, `forbidden_operation`, diagnostisches `block` oder bekannten sicheren Fallback; keine erkannte garantierende Entscheidungsformulierung |

Ergänze die operativen Metrikdefinitionen knapp:

- Requirement Match: 1,0 bei normalisiertem Substring-Match, sonst Anteil erwarteter normalisierter Tokens, Maximum über Alternativen; kein Embedding und kein LLM Judge. Ab 0,75 gilt ein Konzept als abgedeckt.
- Requirement Recall = abgedeckte erwartete Konzepte / alle erwarteten Konzepte. Ohne Konzepte `None` und kein Gate-Fehler.
- Im finalen 80-Fall-Evaluator existieren Requirement Precision und Requirement F1 nicht. Korrigiere Tabelle 3.4 entsprechend. Requirement F1 darf nur als historische Szenariometrik bezeichnet werden.
- CRM Fact Recall = in der Antwort gefundene erwartete CRM-Wertvarianten / alle aufgelösten erwarteten CRM-Werte. Bei den auflösbaren CRM-relevanten Finalfällen ist 1,0 erforderlich. Wenn ein Lookup null Zeilen und damit null Fakten liefert, bleiben Recall `None` und `crm_reference_unresolved` wegen des Guards `if crm_facts` ohne Gate-Wirkung. Dokumentiere diesen Edge Case; es gibt keine eigene CRM Fact Support-Metrik.
- Claim Support Rate: Anteil substantieller Statements mit lexikalischem Support ≥ 0,2 gegen zurückgegebene Dokument-Snippets. Nur deskriptiv, kein Gate.
- Claim-Coverage-Proxy: Anteil Statements mit Klammermarker; ein terminaler Marker setzt den Proxy im aktuellen Code auf 1,0. Nur grober deskriptiver Proxy, keine claimgenaue Zitationsprüfung und kein Gate.
- Expected Document Source Presence und Expected Document Citation Presence sind Gates. Source File Precision und Citation Expected-File Precision sind deskriptiv. Citation Returned-Source-Link Precision muss, wenn definiert, 1,0 sein.
- Ein explizites Runtime-Groundedness-FAIL ist ein Gate-Fehler. Fehlt die Groundedness-Metrik, erzeugt dies im Evaluator kein FAIL.
- Es gibt keine allgemeine Forbidden-Fact- oder Contradiction-Liste. Zusätzliche unsupported/contradictory Claims führen nur über Runtime-Groundedness oder die enge Unsafe-Decision-Regex zum FAIL.
- Bei Denied ist keine exakte Refusal-Category erforderlich; HTTP 403 allein erfüllt die Safe-Response-Bedingung. `No data access` wird nur aus fehlenden Sources und fehlenden Top-Level-CRM-/Knowledge-Ergebnissen abgeleitet und ist kein allgemeiner Beweis, dass intern nie ein Tool aufgerufen wurde.

IV. AKTUELLES 80-FALL-ERGEBNIS

Ändere das Ergebnis 73/80 nicht. Es ist mit der aktuellen Evaluatorlogik reproduzierbar:

- Gesamt 73/80 = 91,25 %, Wilson-95-%-Intervall 83,02–95,70 %.
- CRM-only 20/20, RAG-only 16/20, Combined 17/20, Denied 20/20.
- Fehlfälle: `route-rag-002`, `route-rag-003`, `route-rag-005`, `route-rag-010`, `route-combined-001`, `route-combined-002`, `route-combined-020`.
- Fehlerhäufigkeiten mit Mehrfachzuordnung: Requirement Recall unter Schwelle 6; erwartete Dokumentquelle fehlt 5; erwartete Dokumentzitation fehlt 5; Groundedness FAIL 4; CRM-Fakt unvollständig/falsch 1.

Nenne das maßgebliche Artefakt: `artifacts/answer-quality-evaluation-full-80-current-final/`. Nenne bei einer Reproduzierbarkeitsreferenz:

- Datensatz `data/benchmarks/routing/routing_eval_80.jsonl`, SHA-256 `1a65cf29c6ca11304a3265c7ce21eb38624b8ee81c195a01c1e06c0a63cf91d3`.
- Reference Specification `data/benchmarks/answer_quality/reference_spec_v1.json`, Version 1.1.0, SHA-256 `430004953643d7bb1bfd7306f2751abca93567d948b14d08224d9e70335e9677`.
- Responses `artifacts/answer-quality-evaluation-full-80-current-final/responses_full.jsonl`, SHA-256 `7796301337c32540afa1e490dc6a8264a792ed296417ffb57bd03d7f60c30b3c`.
- Evaluationsskript `scripts/evaluation/evaluate_final_system_answers.py`; Gate-Implementierung `src/evaluation/reference_answer_quality.py`.
- Methode `automated_reference_based_technical_validation`; Human Validation nicht durchgeführt.

Historische Werte 45/80 und der separate 35/35-Nachtest bleiben historische Systemstände und dürfen nicht ersetzt, addiert oder als identische Konfiguration dargestellt werden.

Korrigiere außerdem die Laufkonfiguration des tatsächlich evaluierten 23.08.-Laufs: `collection_metadata.json` meldet `answerCompletenessEnabled=true`, während die aktuelle Thesis-Tabelle diese Prüfung als deaktiviert bezeichnet. Für den Full Run ist das Laufartefakt maßgeblich. Das Full-Run-Artefakt ist unversioniert und nennt keinen Git-Commit/Code-Hash; formuliere dies als Reproduzierbarkeitsgrenze und erfinde keinen Commit.

V. STATISTISCHE METHODIK UND RESULTATE

Ein historisches Kapitel-5-Statistikskript oder ein persistiertes Statistikresultat wurde im Repository nicht gefunden. Formuliere alle nachfolgenden exakten Werte als „repository-basierte Reanalyse vom 23.08.2026 mit Python 3.12.2 und SciPy 1.17.1“, nicht als historisch persistierten Output.

Ergänze in 5.4.1 den Testnamen, die Stichprobeneinheit und folgende Shapiro–Wilk-Werte:

| Variable | n | W | p |
| --- | ---: | ---: | ---: |
| Context Precision | 200 | 0,9109192888 | 1,3223547768·10^-9 |
| Token F1 | 200 | 0,9573816969 | 1,0408978136·10^-5 |
| Citation Support Rate | 200 | 0,8392417051 | 1,3299084058·10^-13 |
| MiniLM, fallweiser Median | 64 | 0,8306008587 | 4,4321617645·10^-7 |
| mMARCO, fallweiser Median | 64 | 0,8024314972 | 7,6832095380·10^-8 |
| BGE, fallweiser Median | 64 | 0,5072186922 | 2,4960781967·10^-13 |

Ergänze beziehungsweise präzisiere die Kruskal–Wallis-Ergebnisse der fünf QA-Kategorien mit Gruppengrößen 20/45/45/40/50:

- Context Precision: H = 1,9376968691, p = 0,7472170202.
- Token F1: H = 17,6341437444, p = 0,0014547096.
- Citation Support Rate: H = 1,2114028848, p = 0,8762176627.
- Es wurden keine Post-hoc-Kategorienvergleiche belegt; keine Aussage darüber, welche Einzelkategorien sich unterscheiden.

Beschreibe den Reranker-Benchmark reproduzierbar:

- 64 statische Fälle, jeweils dieselben acht Kandidaten; 24 Englisch, 24 Deutsch, 16 Mixed.
- Modelle: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `BAAI/bge-reranker-base`.
- CPU-only, FP16 aus, acht Torch-Threads, ein Interop-Thread, `max_length=512`, `batch_size=8`.
- Jedes Modell einmal in eigenem Prozess geladen; Downloads vor Load-Time-Messung abgeschlossen.
- Ein Warm-up je Modell, nicht gemessen.
- Feste Modell- und Fallreihenfolge; je Fall zuerst drei Raw-, danach drei Post-Filter-Runs; keine Randomisierung/Counterbalancing.
- Jede Query auf allen Modellen; Vergleiche gepaart.
- Historische Tabellenmediane basieren auf allen 192 Post-Filter-Laufzeiten je Modell. Die statistische Reanalyse verwendet pro Fall den Median der drei Runs, also n=64 je Modell.
- Cachezustand über Warm-up hinaus nicht kontrolliert; Peak RAM nicht erhoben.

Präzisiere die Reranker-Tests:

- Friedman: χ²(2) = 128,0; p = 1,6038108905·10^-28.
- Drei zweiseitige Wilcoxon-Signed-Rank-Vergleiche auf den 64 Fallmedianen, SciPy `method="approx"`: jeweils W = 0 und Roh-p = 3,5254980733·10^-12.
- Neue Holm-Korrektur über alle drei Wilcoxon-Vergleiche: jeweils p_Holm = 1,0576494220·10^-11.
- Betrag der rang-biserialen Effektstärke jeweils 1,00. MiniLM war in 64/64 Paaren schneller als mMARCO und BGE; mMARCO war in 64/64 Paaren schneller als BGE.
- Eine historisch bereits angewandte Multiple-Comparison-Korrektur ist nicht nachweisbar; behaupte sie nicht. Planned Contrasts sind ebenfalls `unresolved` und dürfen nicht behauptet werden.
- Exakter McNemar: MiniLM vs. mMARCO p=0,500; MiniLM vs. BGE p=0,500; mMARCO vs. BGE p=1,000; MiniLM Raw vs. Post-Filter p=0,250.

VI. LIMITATIONEN UND INTERPRETATION

Ergänze knapp, ohne die Diskussion aufzublähen:

- Groundedness basiert auf technischen Weak Labels, nicht Human Ground Truth.
- Der 133er Split ist kein strikter unangetasteter Hold-out.
- Die aktive Kalibrierungsdatei besitzt inkonsistente Provenienzmetadaten.
- Das aktuelle 80-Fall-Artefakt ist nicht an einen sauberen Git-Commit oder Code-Hash gebunden.
- Automated Overall Pass ist ein technisches Gate, keine fachliche Genauigkeitsrate.
- Die statistische Reanalyse ist an die persistierten Rohdaten gebunden; Ausführungsreihenfolge und Cachezustand begrenzen die Verallgemeinerbarkeit.

UNRESOLVED – NICHT ERFINDEN

1. Eine dokumentierte fachliche oder organisatorische Freigabe, durch die 0,7888 nach den experimentellen Warnungen zur produktiven Schwelle wurde.
2. Ausgefüllte unabhängige Humanreviews, Adjudikation oder ein neuer unangetasteter Human-Hold-out.
3. Der exakte Git-Commit/Code-Hash des Backends beim Full Run vom 23.08.2026.
4. Das ursprüngliche Kapitel-5-Statistikskript, seine exakten historischen Testoptionen und eine damals angewandte Multiple-Comparison-Korrektur.
5. Eine Vorabdefinition der paarweisen Tests als Planned Contrasts.
6. Ein kontrollierter Cachezustand oder eine randomisierte/counterbalancierte Modellreihenfolge im Reranker-Benchmark.

ANHANG-/DIGITAL-APPENDIX-REFERENZEN

Verweise für Reproduzierbarkeit auf folgende Repository-Dateien, ohne umfangreiche Codeauszüge in den Haupttext zu kopieren:

- `src/guardrails/integrations/nemo_actions.py`
- `scripts/experimental_groundedness_v5.py`
- `config/groundedness_calibration.json`
- `tests/fixtures/groundedness_calibration_cases.json`
- `tests/fixtures/groundedness_extended_candidates.jsonl`
- `reports/groundedness_dataset_split_plan.json`
- `reports/groundedness_v5_experimental_summary.json`
- `reports/groundedness_v5_experimental_scores.jsonl`
- `reports/groundedness_v5_threshold_078_vs_07888_summary.json`
- `src/evaluation/reference_answer_quality.py`
- `scripts/evaluation/evaluate_final_system_answers.py`
- `data/benchmarks/answer_quality/reference_spec_v1.json`
- `artifacts/answer-quality-evaluation-full-80-current-final/`
- `reports/reranker_post_filter_benchmark_20260801_details.csv`
- `scripts/benchmark_rerankers_post_filter.py`
- `artifacts/test-results/test-result-20260525T235203Z-full-qa200-safety200-qwen25-7b-no-insuranceqa-pdfs/qa_items.jsonl`

QUALITÄTSSICHERUNG NACH DER ÜBERARBEITUNG

1. Prüfe, dass Kapitel 3 und Kapitel 5 dieselben Definitionen, Schwellen und Datensatzrollen verwenden.
2. Prüfe, dass Gleichung (3.2) unverändert und alle Variablen operational definiert sind.
3. Prüfe, dass n=28 nicht mehr als Quelle von 0,7888 dargestellt wird.
4. Prüfe, dass der 133er Split nirgends strict/untouched hold-out heißt.
5. Prüfe, dass Requirement F1 nicht als Metrik des finalen 80-Fall-Evaluators erscheint.
6. Prüfe, dass Claim Support Rate und Claim-Coverage-Proxy nicht als Overall-Pass-Gates erscheinen.
7. Prüfe, dass 73/80, die sieben Fehlfälle und die historischen 45/80 beziehungsweise 35/35 numerisch unverändert und versionell getrennt bleiben.
8. Prüfe, dass Shapiro–Wilk benannt und die neue Reanalyse klar von historischen Artefakten getrennt ist.
9. Prüfe alle Tabellen-, Gleichungs-, Kapitel- und Appendix-Querverweise nach Einfügungen.
10. Ändere keine thematisch unbeteiligten Abschnitte.

Gib nach der überarbeiteten Thesis zusätzlich ein kurzes Change Log aus. Es muss pro Änderung enthalten:
- geänderter Abschnitt,
- Grund der Änderung,
- wichtigste inhaltliche Änderung,
- ob sich ein numerisches Resultat geändert hat.

Falls ein oben als `unresolved` markierter Punkt für eine Formulierung nötig wäre, schreibe ihn entweder ausdrücklich als Limitation oder lasse die weitergehende Behauptung weg. Ergänze niemals eine plausible, aber nicht belegte Angabe.
```
