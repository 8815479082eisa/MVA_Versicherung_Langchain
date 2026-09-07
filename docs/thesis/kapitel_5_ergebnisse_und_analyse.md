# 5 Ergebnisse und Analyse

Dieses Kapitel fasst die empirischen Ergebnisse des Proof of Concept zusammen. Im Mittelpunkt stehen Retrieval- und Antwortqualität, Quellenbezug, Sicherheit sowie die praktische Ausführbarkeit der drei Verarbeitungspfade. Da die gespeicherten Tests zu unterschiedlichen Entwicklungsständen gehören, werden ihre Ergebnisse getrennt ausgewiesen und nicht zu einer gemeinsamen Erfolgsquote zusammengeführt.

## 5.1 Auswertungsstrategie und Ergebnisdarstellung

Berücksichtigt wurden ausschließlich im Repository nachvollziehbare Ergebnisdateien und Testberichte. Ein vollständig ausgeführter Test kann dabei fachlich bestanden, fehlgeschlagen oder in einen Fallback gewechselt sein. Technisch abgebrochene Läufe werden nicht als fachliche Systemergebnisse gewertet.

**Tabelle 5.1: Abgrenzung der zentralen Ergebnisquellen**

| Ergebnisquelle | Umfang | Einordnung |
|---|---:|---|
| Referenzlauf vom 25.05.2026 | 200 QA- und 200 Safety-Fälle | damaliger RAG-Stand mit qwen2.5:7b-instruct, ohne heutige CRM–RAG-Integration |
| RAG-only-End-to-End-Tests vom 16.07.2026 | zwei Läufe derselben Frage | kontrollierter Einzeltest in einem isolierten und einem größeren Korpus |
| CRM-only- und Combined-Tests vom 26.07. bis 05.08.2026 | Live-, Wiederholungs- und Regressionstests | Nachweise für einzelne Verarbeitungspfade und dokumentierte Grenzfälle |
| Reranker-Benchmark vom 01.08.2026 | 64 statische Fälle, drei Wiederholungen je Modell | isolierter Vergleich von Rankingqualität und CPU-Latenz |
| Groundedness-v5-Auswertung | 498 technische Weak-Label-Fälle | technische Prüfung der Schwelle 0,7888, keine abgeschlossene Humanannotation |
| Separate GPT-/OpenAI-Tests | frühe Diagnostik, fünf Smoke-Fälle und gezielte Live-Fälle | ergänzende Ergebnisse zur Antwortgenerierung, nicht direkt vergleichbar |

Der aktuelle Prototypstand verwendet gpt-4o-mini, den MiniLM-Reranker und Groundedness v5 mit der Schwelle 0,7888. Für diesen Gesamtstand liegt kein neuer Test mit 200 QA- und 200 Safety-Fällen vor. Kleine Live-Testreihen werden deshalb als Fallstudien berichtet. Für ausgewählte binäre Anteile werden Wilson-Intervalle verwendet \cite{wilson1927}; die statistischen Vergleiche beschränken sich auf ausreichend große und passend strukturierte Datensätze.

## 5.2 Deskriptive Analyse

### 5.2.1 Lage- und Streuungsmaße

Tabelle 5.2 enthält die zentralen QA-Kennzahlen des Referenzlaufs vom 25.05.2026.

**Tabelle 5.2: Zentrale QA-Ergebnisse des Referenzlaufs (n = 200)**

| Metrik | Ergebnis |
|---|---:|
| Retrieval Support Hit Rate | 97,5 % |
| Context Precision | 0,635 |
| Answer Token F1 | 0,173 |
| Citation Presence | 91,0 % |
| Citation Support | 0,597 |

Das Retrieval fand bei fast allen Fragen mindestens eine stützende Passage. Die niedrigeren Werte für Context Precision, Token F1 und Citation Support zeigen jedoch, dass gute Retrieval-Abdeckung nicht automatisch zu einer präzisen, vollständigen und vollständig belegten Antwort führte.

### 5.2.2 Varianz und Standardabweichung

Die Streuung war bei Context Precision und Citation Support stärker als beim Token F1. Die Standardabweichungen betrugen 0,256 für Context Precision, 0,271 für Citation Support und 0,090 für Token F1. Separate Varianzwerte werden nicht berichtet, da ihre quadrierte Skala keine zusätzliche inhaltliche Aussage für die Forschungsfrage liefert.

### 5.2.3 Konfidenzintervalle

Für drei zentrale Kennzahlen wurden 95-%-Intervalle bestimmt.

**Tabelle 5.3: Ausgewählte 95-%-Intervalle des Referenzlaufs**

| Metrik | Punktschätzung | 95-%-Intervall |
|---|---:|---:|
| Retrieval Support Hit Rate | 97,5 % | 94,3–98,9 % |
| Answer Token F1 | 0,173 | 0,161–0,185 |
| Citation Presence | 91,0 % | 86,2–94,2 % |

Die Intervalle beschreiben die Unsicherheit innerhalb des damaligen Testbestands. Sie sind nicht auf den aktuellen Prototyp oder den realen Versicherungsbetrieb übertragbar.

### 5.2.4 Verteilung der Messwerte

Die begrenzten und teilweise diskreten Qualitätsmetriken waren nicht symmetrisch verteilt. Beim Token F1 lagen Median und Interquartilsabstand bei 0,159 beziehungsweise 0,101–0,230. Für die Gruppenvergleiche wurden daher nichtparametrische Verfahren eingesetzt; eine zusätzliche Verteilungsabbildung ist für diesen Abschnitt nicht erforderlich.

## 5.3 Ergebnisse der Systemkomponenten

### 5.3.1 Retrieval-Ergebnisse

Im Referenzlauf fand das Retrieval bei 195 von 200 Fragen mindestens eine nach der verwendeten Heuristik stützende Passage. Die Retrieval Support Hit Rate betrug damit 97,5 %. Die Context Precision von 0,635 zeigt zugleich, dass der finale Kontext häufig zusätzliche oder nur teilweise relevante Passagen enthielt.

Die RAG-only-End-to-End-Prüfung vom 16.07.2026 wurde mit derselben synthetischen Windschutzscheibenfrage in zwei Umgebungen durchgeführt. Sowohl im isolierten Drei-Chunk-Korpus als auch in einer temporären Vollkorpuskopie wurde die relevante Seite auf Rang 1 eingeordnet. Beide Läufe enthielten alle sechs erwarteten Fakten und bestanden die damaligen Zitations-, Groundedness- und Safety-Prüfungen. Da dieselbe kontrollierte Frage zweimal verwendet wurde, handelt es sich nicht um zwei unabhängige Qualitätsfälle.

Der aktuelle Helvetia-Index umfasst 1.585 Chunks aus elf PDF-Dokumenten. In drei Smoke-Tests zu Glasschäden, Hausrat und Annullierungskosten erschienen die jeweils einschlägigen Dokumente unter den ersten fünf Treffern. Dies bestätigt die technische Funktionsfähigkeit des Index, ersetzt jedoch keine umfassende Recall- oder Precision-Evaluation. Im Noah-Diebstahlfall erreichten die erforderlichen Dokumentseiten den finalen Kontext. Die später fehlenden Inhalte waren daher kein Retrieval-Fehler.

### 5.3.2 Ergebnisse des Reranker-Vergleichs

Der isolierte Benchmark verglich drei Cross-Encoder auf 64 statischen Fällen mit jeweils acht Kandidaten und drei Wiederholungen. Tabelle 5.4 zeigt die Ergebnisse nach der deterministischen Policenfilterung.

**Tabelle 5.4: Rankingqualität und CPU-Latenz der Reranker**

| Modell | Top-1 | MRR@5 | nDCG@5 | Medianlatenz |
|---|---:|---:|---:|---:|
| ms-marco-MiniLM-L-6-v2 | 0,969 | 0,982 | 0,986 | 467,8 ms |
| mmarco-mMiniLMv2-L12-H384-v1 | 1,000 | 1,000 | 1,000 | 843,5 ms |
| bge-reranker-base | 1,000 | 1,000 | 1,000 | 4.889,3 ms |

MiniLM ordnete in 62 von 64 Fällen den korrekten Kandidaten auf Rang 1 ein. Die beiden größeren Modelle erreichten 64 von 64 Treffern, benötigten auf der CPU jedoch deutlich mehr Zeit. Im englischen Teilbestand erreichten alle drei Modelle Top-1 = 1,0.

Die deterministische Policenfilterung erhöhte MiniLM von 59 auf 62 Top-1-Treffer. Im Lara-Fall zur aktuellen Police bevorzugten alle drei semantischen Reranker zunächst eine ältere Police. Erst die strukturierte Prüfung von Status und Laufzeit wählte die tatsächlich aktuelle Police aus. Semantisches Ranking allein bildete die zeitliche Vertragslogik somit nicht zuverlässig ab.

**Abbildung 5.1 (Darstellungsvorschlag):** Streudiagramm mit der Medianlatenz auf einer logarithmischen x-Achse und nDCG@5 auf der y-Achse. Die drei beschrifteten Modellpunkte machen den Qualitäts-Latenz-Kompromiss unmittelbar sichtbar.

### 5.3.3 Ergebnisse der Antwortgenerierung

Die gespeicherten Antworttests stammen aus unterschiedlichen Systemständen. Tabelle 5.5 stellt deshalb nur ihre jeweils beobachteten Ergebnisse dar; ein direkter Modellvergleich ist daraus nicht möglich.

**Tabelle 5.5: Zentrale Ergebnisse der Antwortgenerierung**

| Teststand | Umfang | Beobachtetes Ergebnis | Begrenzung |
|---|---:|---|---|
| Referenzlauf, qwen2.5:7b-instruct, 25.05.2026 | 200 Fragen | Exact Match 0; mittlerer Token F1 0,173; 18 Fallback-Antworten | damaliger RAG-Stand ohne CRM-Integration |
| Frühes automatisches GPT-Diagnostikum, 26.02.2026 | 8 Einträge | das Artefakt nennt gpt-4o als LLM-Metadatum; automatisch bewerteter Stützungsscore 0,988 | Antworttexte fehlen; vier Einträge betreffen dieselbe Frage; gpt-4o-mini war der Evaluator |
| OpenAI-InsuranceQA-Smoke-Test, 04.03.2026 | 5 Fragen | Exact Match 0; mittlerer Token F1 0,245; zwei Antworten lehnten wegen fehlender Evidenz ab | genaue Modellversion nicht gespeichert |
| Lara-Windschutzscheibe mit gpt-4o-mini, 29.07.2026 | drei Wiederholungen derselben Frage | Gesamtpipeline dreimal semantisch bestanden; mittlere Generierungszeit 4,279 s | damaliger Baloise-Korpus, Schwelle 0,51 und deterministische Nachbearbeitung |
| Noah-Diebstahl mit gpt-4o-mini, 05.08.2026 | gezielte aktuelle Läufe | vorhandene Evidenz teilweise ausgelassen; Regeneration beseitigte eine Quellenlücke nicht | Fallstudie, keine repräsentative Stichprobe |

Im OpenAI-Smoke-Test enthielten die drei inhaltlichen Antworten noch den nicht aufgelösten Platzhalter [Doc-ID:page]. Somit lag trotz des höheren numerischen Token F1 keine verwendbare konkrete Dokumentzitation vor. Die kleine Stichprobe und die unbekannte Modellversion schließen eine belastbare Verbesserungsaussage gegenüber dem Referenzlauf aus. Auch das frühe GPT-Diagnostikum eignet sich wegen fehlender Antworttexte und wiederholter Fragen nur als Entwicklungsbefund.

Die Lara-Wiederholungen zeigen, dass die damalige Gesamtpipeline vollständige und verständliche Antworten erzeugen konnte. Die drei Läufe sind jedoch keine unabhängigen Fragen und dürfen nicht allein dem Antwortmodell zugeschrieben werden. Im aktuellen Noah-Fall ließ die Generierung trotz vorhandenem Kontext zwei Pflichten aus. In einem weiteren Lauf erfüllte die erste Antwort neun von zehn Anforderungen; die Regeneration erzeugte denselben Text. Für den aktuellen gpt-4o-mini-Stand liegt daher kein umfangreicher Antwortqualitätsbenchmark vor.

### 5.3.4 Groundedness- und Zitationsergebnisse

Im Referenzlauf enthielten 91,0 % der Antworten mindestens eine sichtbare Quelle. Der mittlere Citation Support von 0,597 zeigt, dass eine vorhandene Zitation nicht automatisch alle Aussagen der Antwort stützte.

Groundedness v5 wurde auf 498 konstruierten technischen Fällen geprüft. Die zunächst als experimenteller Vergleich erstellte Auswertung umfasst alle Weak-Label-Konfidenzstufen; im Prototypstand vom 05.08.2026 ist das Verfahren mit der Schwelle 0,7888 technisch aktiv.

**Tabelle 5.6: Groundedness v5 bei Schwelle 0,7888 gegen technische Weak Labels**

| Split | n | TP | FP | TN | FN | Precision | Recall | False Acceptance Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Kalibrierung | 277 | 110 | 4 | 146 | 17 | 96,5 % | 86,6 % | 2,7 % |
| Validierung | 88 | 41 | 0 | 45 | 2 | 100,0 % | 95,3 % | 0,0 % |
| technischer Hold-out-Kandidat | 133 | 62 | 0 | 69 | 2 | 100,0 % | 96,9 % | 0,0 % |
| Gesamt | 498 | 213 | 4 | 260 | 21 | 98,2 % | 91,0 % | 1,5 % |

Die Medianwerte der technischen PASS- und FAIL-Fälle lagen bei 1,000 beziehungsweise 0,282. Die Schwelle trennte die konstruierten Fälle damit deutlich, ohne alle PASS-Fälle anzunehmen. Da die Labels aus technischen Regeln stammen und keine abgeschlossene unabhängige Humanannotation vorliegt, beschreiben die Kennzahlen keine humanvalidierte fachliche Groundedness.

Die Noah-Tests verdeutlichen die Grenzen der einzelnen Kontrollgrößen. Eine unvollständige Antwort bestand mit einem Groundedness-Score von 0,856, weil die tatsächlich genannten Aussagen belegt waren. In einem anderen Lauf wurde eine nummerierte Überschrift als unbelegte Zahl interpretiert und der Score auf 0,38 begrenzt. Groundedness bewertet somit die Stützung vorhandener Aussagen, aber nicht zuverlässig deren Vollständigkeit. Citation Presence bestätigt nur, dass eine Quelle genannt wurde; Citation Support bewertet dagegen, ob die Aussagen durch diese Evidenz getragen werden.

**Abbildung 5.2 (Darstellungsvorschlag):** Zwei Box- oder Violinplots für die Groundedness-Scores der technischen PASS- und FAIL-Fälle mit einer eingezeichneten Schwelle bei 0,7888. Die Beschriftung muss ausdrücklich auf Weak Labels und die fehlende abgeschlossene Humanannotation hinweisen.

### 5.3.5 Ergebnisse der CRM-only-, RAG-only- und Combined-Tests

Die drei Verarbeitungspfade wurden zu unterschiedlichen Entwicklungsständen geprüft. Tabelle 5.7 fasst den jeweils nachgewiesenen Funktionsumfang zusammen; eine gemeinsame Erfolgsquote wird nicht berechnet.

**Tabelle 5.7: Ergebnisse der drei Verarbeitungspfade**

| Pfad und Teststand | Umfang | Zentrales Ergebnis |
|---|---:|---|
| CRM-only, 26.07.2026 | eine Live-Abfrage; fünf MCP-Werkzeuge; je 50 REST- und MCP-Aufrufe | HTTP 200 und korrekte Vertragsdaten; alle Werkzeug- und Benchmarkaufrufe erfolgreich |
| RAG-only, Referenzlauf 25.05.2026 | 200 Fragen | 200 von 200 verarbeitet; Retrieval Support Hit Rate 97,5 %, Token F1 0,173, Citation Presence 91,0 % |
| RAG-only, 16.07.2026 | zwei End-to-End-Läufe derselben Frage | im isolierten Korpus und in der Vollkorpuskopie jeweils PASS; relevante Seite auf Rang 1 |
| Combined, 27. und 29.07.2026 | ein Marderbiss- und drei Windschutzscheibenläufe | vier vollständig bestandene Antworten mit getrennten CRM- und Dokumentbelegen |
| Combined mit MiniLM, 01.08.2026 | zwei End-to-End-Läufe | zweimal HTTP 200; aktuelle Police und zentrale Glasdeckungsinformation korrekt, aber sprachliche Ungenauigkeit zu GlassPlus |
| Combined, Noah-Fall 05.08.2026 | gezielte aktuelle Läufe | CRM-Auswahl und Retrieval korrekt; finale Antwort wegen Vollständigkeit beziehungsweise Groundedness nicht erfolgreich |

Der CRM-only-Test bestätigte den lesenden Zugriff auf Kunden-, Policen- und Schadendaten. Alle fünf freigegebenen MCP-Werkzeuge funktionierten. Je 50 direkte REST- und persistente MCP-Aufrufe wurden ohne Fehler abgeschlossen. Weitere 22 Tests bestätigten das Routing und die Auswahl der aktuellen Police.

Für RAG-only bleibt der Referenzlauf ein vollständiger quantitativer Nachweis des damaligen Systems. Die beiden End-to-End-Läufe vom 16.07. ergänzen diesen Befund um einen kontrollierten erfolgreichen Einzelfall. Sie bilden weder das aktuelle Antwortmodell noch die spätere CRM-Integration vollständig ab.

Für Combined liegen vier ausdrücklich bestandene Entwicklungsantworten mit dem damaligen Baloise-Korpus vor. Die zwei späteren MiniLM-Läufe bestätigten zusätzlich die aktuelle Policenauswahl und die Nutzung des Helvetia-Korpus, enthielten jedoch eine sprachliche Ungenauigkeit bei GlassPlus. Der vollständig ausgeführte Noah-Test war fachlich nicht erfolgreich: Die benötigten CRM- und Dokumentinformationen waren vorhanden, die finale Antwort erfüllte aber nicht alle Anforderungen an Vollständigkeit und Groundedness. Erfolgreiche Teilkomponenten garantierten somit keine erfolgreiche Combined-Endantwort.

Technisch abgebrochene Versuche, etwa aufgrund eines Self-Check-Timeouts oder einer nicht verfügbaren lokalen Modelllaufzeit, werden nicht als fachliche Ergebnisse dieser Pfade gewertet.

### 5.3.6 Sicherheits- und Robustheitsergebnisse

Im Safety-Teil des Referenzlaufs wurden alle 70 Angriffsfälle blockiert und alle 130 legitimen Anfragen zugelassen. Im vorgegebenen Testsatz traten damit weder False Negatives noch False Positives auf. Das Ergebnis gilt für den Systemstand vom 25.05.2026; eine unabhängige Humanannotation der festen Testlabels ist nicht dokumentiert.

Die read-only-Beschränkung der CRM-Anbindung wurde mit fünf unerlaubten Operationen geprüft. Das Anlegen, Ändern, Löschen und Exportieren von Daten wurde jeweils mit HTTP 403 abgewiesen, und die Datenbestände blieben unverändert. Der MCP-Server stellte ausschließlich fünf lesende Werkzeuge bereit.

Im Noah-Fall wurde außerdem die CRM-E-Mail-Adresse vor der Antwortgenerierung redigiert, während die erforderlichen Policenangaben erhalten blieben. Der spätere Fallback wurde durch die Groundedness-Prüfung und nicht durch die PII-Erkennung ausgelöst. Dies bestätigt die getrennte Wirkung beider Kontrollen im geprüften Fall.

### 5.3.7 Laufzeit- und Ressourcenergebnisse

Tabelle 5.8 stellt nur zentrale Laufzeiten dar. Die Werte stammen aus unterschiedlichen Konfigurationen und sind daher nicht als einheitlicher Performance-Benchmark zu lesen.

**Tabelle 5.8: Ausgewählte Komponenten- und End-to-End-Laufzeiten**

| Messung | Ergebnis | Einordnung |
|---|---:|---|
| EspoCRM REST, n = 50 | Median 31,6 ms; p95 37,8 ms | warme Komponentenmessung |
| Persistentes CRM-MCP, n = 50 | Median 67,7 ms; p95 102,3 ms | warme Komponentenmessung |
| RAG-only, 16.07.2026 | 358,0 s isoliert; 588,4 s Vollkorpuskopie | zwei erfolgreiche Einzelfälle; nicht interaktiv geeignet |
| Combined mit MiniLM, 01.08.2026 | 24,124 s kalt; 11,498 s warm | zwei einzelne End-to-End-Läufe |
| Noah-Combined, 05.08.2026 | 35,721 s | vollständig ausgeführter fachlicher FAIL |
| QA-Referenzlauf, n = 200 | Median 21,180 s; p95 23,641 s | damaliger Systemstand |

Die CRM-Aufrufe waren gegenüber der Gesamtpipeline kurz. Die Reranker-Latenzen in Tabelle 5.4 zeigen außerdem, dass die Modellwahl die lokale Verarbeitungszeit deutlich beeinflusste. Der Noah-Lauf blieb unter dem Backend-Gesamttimeout von 180 s, endete jedoch fachlich nicht erfolgreich. Laufzeit und Antwortqualität sind daher getrennte Bewertungsdimensionen.

Die gespeicherten Ressourcenwerte sind Momentaufnahmen. Das Backend beanspruchte etwa 1,93 bis 2,16 GiB RAM, EspoCRM rund 128 MiB und MariaDB rund 159 MiB. Im geprüften Zustand lag der größte Speicherbedarf somit beim Backend. Aussagen über Spitzenverbrauch oder Skalierbarkeit sind aus diesen Einzelmessungen nicht ableitbar.

## 5.4 Statistische Analyse

### 5.4.1 Prüfung der Verteilungsannahmen

Die Normalverteilungsannahme wurde für Context Precision, Token F1, Citation Support und die fallweisen Reranker-Latenzen verworfen (jeweils p < 0,001). Aufgrund der begrenzten beziehungsweise diskreten Qualitätsmetriken und der gepaarten Reranker-Messungen wurden nichtparametrische Tests verwendet.

### 5.4.2 Auswahl der statistischen Testverfahren

Der Kruskal-Wallis-Test wurde für Unterschiede zwischen den fünf QA-Kategorien eingesetzt \cite{kruskal1952}. Die gepaarten Reranker-Latenzen wurden mit dem Friedman- und anschließend dem Wilcoxon-Test verglichen \cite{friedman1937,wilcoxon1945}. Der exakte McNemar-Test prüfte Unterschiede der Top-1-Entscheidungen \cite{mcnemar1947}; Wilson-Intervalle beschrieben ausgewählte binäre Anteile. Für kleine Live-Fallstudien wurden keine Signifikanztests berechnet.

### 5.4.3 Parametrische und nichtparametrische Vergleiche

Aufgrund der Verteilungsbefunde werden keine parametrischen Gruppenvergleiche als Hauptergebnis berichtet. Die wesentlichen nichtparametrischen Ergebnisse sind in Tabelle 5.9 zusammengefasst.

**Tabelle 5.9: Zentrale nichtparametrische Vergleiche**

| Vergleich | Teststatistik | p-Wert | Ergebnis |
|---|---:|---:|---|
| Context Precision zwischen QA-Kategorien | H = 1,938 | 0,747 | kein nachweisbarer Unterschied |
| Token F1 zwischen QA-Kategorien | H = 17,634 | 0,0015 | globaler Kategorienunterschied |
| Citation Support zwischen QA-Kategorien | H = 1,211 | 0,876 | kein nachweisbarer Unterschied |
| Reranker-Latenz über drei Modelle | χ²(2) = 128,0 | < 0,001 | deutlicher Modellunterschied |
| MiniLM gegenüber mMARCO, Latenz | W = 0 | < 0,001 | MiniLM in allen 64 Paaren schneller |
| MiniLM gegenüber BGE, Latenz | W = 0 | < 0,001 | MiniLM in allen 64 Paaren schneller |
| MiniLM gegenüber mMARCO beziehungsweise BGE, Top-1 | exakter McNemar-Test | jeweils 0,500 | zwei zusätzliche Treffer nicht statistisch abgesichert |

Der globale Token-F1-Unterschied erlaubt ohne Post-hoc-Test keine Aussage darüber, welche einzelnen Kategorien sich signifikant unterscheiden. Deskriptiv hatten allgemeine Definitionsfragen den höchsten Mittelwert. Die Policenfilterung erhöhte MiniLM von 59 auf 62 Top-1-Treffer; mit p = 0,250 war auch dieser Unterschied statistisch nicht abgesichert, löste jedoch den fachlich wichtigen Lara-Fall zur aktuellen Police.

### 5.4.4 Effektstärken und praktische Relevanz

Der Kategorienunterschied beim Token F1 hatte mit ε² = 0,070 einen kleinen bis mittleren Effekt. Die Fragekategorie erklärte damit nur einen begrenzten Teil der Unterschiede im Referenzlauf.

Praktisch bedeutsamer war der Reranker-Vergleich. MiniLM war in allen 64 gepaarten Fallmedianen schneller als mMARCO und BGE; der Betrag der rang-biserialen Effektstärke lag jeweils bei |r_rb| = 1,00. Die Medianlatenz von mMARCO betrug das 1,80-Fache und die von BGE das 10,45-Fache der MiniLM-Latenz. Dem standen nur zwei zusätzliche Top-1-Treffer beziehungsweise 3,1 Prozentpunkte gegenüber. Im untersuchten CPU-Setup bot MiniLM daher den günstigsten Qualitäts-Latenz-Kompromiss. Daraus folgt keine allgemeine Überlegenheit außerhalb dieses kleinen, teilweise synthetischen Benchmarks.

Zusammenfassend arbeiteten Retrieval und CRM-Anbindung in den jeweils dokumentierten Tests zuverlässig. MiniLM erreichte eine hohe Rankingqualität bei deutlich geringerer Latenz als die größeren Reranker. Groundedness und Zitation lieferten wichtige Kontrollen für die Quellenstützung, garantierten aber keine vollständige Antwort. Der Noah-Fall zeigte, dass erfolgreiche CRM-, Retrieval- und Reranking-Schritte nicht automatisch zu einer erfolgreichen Combined-Endantwort führen. Die Aussagekraft bleibt durch unterschiedliche Systemstände, synthetische Testdaten und die begrenzte Humanannotation eingeschränkt.
