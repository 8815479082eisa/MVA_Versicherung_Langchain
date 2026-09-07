# Arbeitsgrundlage für Prism – Kapitel 4: Experimenteller Aufbau und Implementierung

## Schreibauftrag an Prism

Aus den folgenden Angaben soll ein zusammenhängendes Kapitel 4 einer Masterarbeit erstellt werden. Die Sprache soll einfach, präzise und wissenschaftlich sein. Kurze Absätze sind langen technischen Erklärungen vorzuziehen. Formeln und Pseudocode aus Abschnitt 3.8 werden nicht wiederholt. Stattdessen wird an den passenden Stellen auf die dort beschriebenen Algorithmen verwiesen.

Für den Text gelten folgende Regeln:

- Das System ist ein wissenschaftlicher Proof of Concept für interne Versicherungsanfragen und kein produktionsreifes System.
- Der Begriff agentisch bezeichnet die mehrstufige Orchestrierung und die kontrollierte Nutzung von Werkzeugen. Das Top-Level-Routing ist trotzdem deterministisch und regelbasiert.
- Die vier Verarbeitungspfade heißen konsistent RAG-only beziehungsweise retrieval-only, CRM-only, Combined CRM–RAG und Denied.
- CRM-Daten sind synthetisch und werden ausschließlich lesend verwendet.
- CRM-Fakten beschreiben individuelle Vertragsmerkmale. Allgemeine Deckungsregeln, Bedingungen und Ausschlüsse stammen aus Dokumenten.
- Das System trifft keine rechtsverbindliche Deckungs- oder Schadenentscheidung.
- Groundedness misst, ob vorhandene Aussagen durch Evidenz gestützt werden. Sie misst nicht automatisch die Vollständigkeit einer Antwort.
- Ergebnisse und deren Interpretation gehören in Kapitel 5. Kapitel 4 beschreibt Aufbau, Daten, Konfiguration und Durchführung.
- Nicht belegte Details dürfen nicht ergänzt oder geschätzt werden.

## Verbindliche Abgrenzung der Systemstände

Diese Abgrenzung muss am Anfang von Kapitel 4 oder spätestens in Abschnitt 4.6 deutlich genannt werden.

| Bezeichnung | Git-Stand | Zentrale Eigenschaften | Bedeutung für die Arbeit |
|---|---|---|---|
| Aktueller Prototyp | Branch Abschluss-Arbeit, 437812f7be59ff6ae21325056d158b497bf56f2b, Commit vom 05.08.2026 | CRM–RAG-Integration, gpt-4o-mini als Antwortmodell, MiniLM-Reranker, Groundedness v5 mit Schwelle 0,7888 | Referenzstand für die Beschreibung der aktuellen Implementierung |
| Vollständig evaluierter Hauptlauf | 42878a1c…, Lauf vom 25.05.2026 | 200 QA-Fälle und 200 Sicherheitsfälle, älterer Systemstand ohne heutige CRM–RAG-Integration, qwen2.5:7b-instruct als Antwortmodell, damalige Groundedness-Schwelle 0,2 | Historischer Evaluationsstand; darf nicht als Evaluation des aktuellen Prototyps dargestellt werden |

Für den aktuellen Commit liegt kein vollständiger neuer Lauf über dieselben 200 QA- und 200 Sicherheitsfälle vor. Wenn Kapitel 5 den historischen Lauf auswertet, muss die Arbeit ausdrücklich erklären, dass dessen Resultate nicht ohne Weiteres auf die aktuelle CRM–RAG-Version übertragbar sind.

Die lokalen Umgebungsdateien, der Chroma-Index und die Modell-Caches sind nicht im Git-Commit enthalten. Der Commit allein reproduziert deshalb nicht den vollständigen lokalen Versuchsstand.

# 4. Experimenteller Aufbau und Implementierung

## 4.1 Ziel und Aufbau des Kapitels

### Empfohlener Text

Dieses Kapitel dokumentiert die technische Umsetzung und den experimentellen Aufbau des in Kapitel 3 entwickelten Prototyps. Ziel ist es, die verwendete Hardware- und Softwareumgebung, die Implementierung der Verarbeitungswege, die Datenaufbereitung sowie die aktiven Modelle und Konfigurationen nachvollziehbar zu beschreiben. Außerdem werden die Erstellung der Referenzdaten, die Promptgestaltung und der Umfang der verwendeten Datensätze erläutert.

Die Darstellung bezieht sich primär auf den aktuellen Implementierungsstand des Commits 437812f7be59ff6ae21325056d158b497bf56f2b. Historische Evaluationsläufe werden getrennt gekennzeichnet, da sie mit anderen Modellen und Einstellungen durchgeführt wurden. Die in Abschnitt 3.8 formal beschriebenen Algorithmen werden in diesem Kapitel nicht erneut hergeleitet. Stattdessen wird gezeigt, wie sie im Prototyp umgesetzt und konfiguriert wurden.

## 4.2 Hardware- und Softwareumgebung

### 4.2.1 Hardwareumgebung

Die aktuelle lokale Entwicklungs- und Testumgebung wurde am 15.08.2026 direkt auf dem verfügbaren Rechner geprüft. Diese Angaben beschreiben den gegenwärtigen Host. Für den historischen Hauptlauf vom 25.05.2026 wurde keine vollständige, unveränderliche Hardwarebeschreibung zusammen mit den Ergebnissen gespeichert. Die aktuelle Hardware darf daher nicht ungeprüft als Hardware dieses älteren Laufs bezeichnet werden.

| Komponente | Aktuell geprüfter lokaler Stand | Einordnung |
|---|---|---|
| Prozessor | Intel Core i7-1065G7, acht logische Prozessoren | CPU-basierte Entwicklungsumgebung |
| Arbeitsspeicher | rund 19,74 GiB | gemessener Gesamtspeicher des Hosts |
| Grafik | integrierte Intel Iris Plus Graphics | keine CUDA-GPU; nicht für Embedding oder Reranking verwendet |
| Systempartition | NTFS, 337,5 GiB; bei der Prüfung 55,2 GiB frei | physischer Datenträgertyp nicht dokumentiert |
| Embedding-Ausführung | CPU | BAAI/bge-m3 |
| Reranker-Ausführung | CPU, FP16 deaktiviert | cross-encoder/ms-marco-MiniLM-L-6-v2 |

Für die Arbeit genügt die Bezeichnung CPU-basierte Versuchsumgebung. Aussagen über GPU-Beschleunigung, VRAM oder einen Serverbetrieb sind nicht belegt.

### 4.2.2 Softwareumgebung

Der Prototyp besteht aus einem FastAPI-Backend, einer React-Oberfläche, einer persistenten Chroma-Vektordatenbank und einer read-only angebundenen EspoCRM-Instanz. Das Antwortmodell wird über einen externen OpenAI-Dienst aufgerufen. Lokale Hilfsmodelle können über Ollama auf dem Windows-Host bereitgestellt werden.

| Ebene | Software beziehungsweise Version | Funktion |
|---|---|---|
| Host-Betriebssystem | Windows 11, Build 26100 | lokale Ausführung und Docker-Host |
| Lokale Python-Umgebung | Python 3.12.2 | Entwicklung, Skripte und Analysen |
| Backend-Container | python:3.11-slim | FastAPI-Backend |
| Frontend-Container | node:20-alpine | React-/Vite-Frontend |
| Reverse Proxy | nginx:alpine | optionaler HTTP-/HTTPS-Zugang |
| Containerwerkzeuge | Docker 29.6.2, Docker Compose 5.3.1 | Bereitstellung der Dienste |
| API-Framework | FastAPI 0.124.4, Uvicorn 0.38.0, Pydantic 2.12.5 | HTTP-API und Datenvalidierung |
| RAG-Framework | LangChain 1.1.2, langchain-community 0.4.1 | Dokumentverarbeitung und Pipelinebausteine |
| Vektorspeicher | ChromaDB 1.3.5, langchain-chroma 1.0.0 | persistente Embeddings und Ähnlichkeitssuche |
| Modellbibliotheken | sentence-transformers 5.1.2, Transformers 4.57.3, FlagEmbedding 1.3.5 | Embedding und Reranking |
| Sicherheitsframework | NeMo Guardrails 0.21.0 | Input-, Kontext- und Output-Rails |
| Werkzeugprotokoll | FastMCP 3.4.0, langchain-mcp-adapters 0.3.0 | CRM-Werkzeuge über einen persistenten Unterprozess |
| Antwortschnittstelle | OpenAI-Python 2.9.0 | Zugriff auf das externe Antwortmodell |
| Frontend | React 18.3.1, Vite 5.4.21, TypeScript 5.9.3 | Benutzeroberfläche |
| CRM | EspoCRM 10.0.3 | strukturierte Kontakte, Policen und Schäden |
| CRM-Datenbank | MariaDB 11.4 | persistente CRM-Speicherung |

Die meisten Python-Pakete sind in requirements.txt nicht exakt gepinnt. Auch ein Frontend-Lockfile liegt nicht vor. Die genannten Bibliotheksversionen beschreiben deshalb die lokal aufgelöste Umgebung, nicht automatisch jeden späteren Container-Neubau. Für eine endgültige Reproduktion sollten ein Python-Lockfile, ein Frontend-Lockfile und die verwendeten Container-Digests archiviert werden.

### 4.2.3 Hardware- und Bereitstellungstopologie

Die folgende Abbildung zeigt die physische und logische Verteilung. Der Retrievalpfad läuft direkt im Backendprozess. Der vorhandene Retrieval-MCP-Server ist nur ein zusätzlicher Werkzeug- und Testpfad und gehört nicht zum normalen öffentlichen Endpunkt.

~~~mermaid
flowchart LR
    U["Interne nutzende Person"] --> B["Browser und React-Frontend<br/>Port 5173"]

    subgraph H["Lokaler Windows-11-Host"]
        O["Ollama<br/>lokales Guardrail-Modell"]

        subgraph D["Docker-Bridge-Netz mva-network"]
            N["Nginx<br/>optional, Ports 80 und 443"]
            F["Frontend-Container"]
            A["FastAPI-Backend<br/>Port 8000"]
            R["Regelbasiertes Routing<br/>RAG-only, CRM-only, Combined, Denied"]
            Q["BM25, Chroma, bge-m3<br/>RRF und MiniLM-Reranker"]
            S["NeMo-Rails, PII-Prüfung,<br/>Groundedness, Audit und Diagnostik"]
            M["CRM-MCP-Unterprozess<br/>stdio, fünf read-only Werkzeuge"]
            E["EspoCRM 10.0.3<br/>Port 8080 am Host"]
            DB["MariaDB 11.4"]
            V1[("PDF- und Chroma-Volumes")]
            V2[("CRM-Volumes")]

            N -.-> F
            N -.-> A
            F -->|"HTTP /api/ask"| A
            A --> R
            R -->|"Dokumentpfad"| Q
            R -->|"CRM-Pfad"| M
            M -->|"REST / API-Key"| E
            E --> DB
            Q --> V1
            DB --> V2
        end

        A -->|"host.docker.internal:11434"| O
    end

    B --> F
    A -->|"HTTPS, Responses API"| X["Externer OpenAI-Dienst<br/>gpt-4o-mini"]
    A --> S
    X --> S
    S -->|"Antwort oder sicherer Fallback"| B
~~~

**Vorgeschlagene Bildunterschrift:** Physische und logische Bereitstellungstopologie des untersuchten Proof of Concept. Durchgezogene Linien zeigen den regulären Verarbeitungspfad; gestrichelte Linien kennzeichnen den optionalen Reverse Proxy.

**Hinweis für Prism:** Lokale beziehungsweise containerisierte Komponenten, der Windows-Host und der externe OpenAI-Dienst sollen visuell getrennt werden. Chroma ist kein eigener Datenbankserver, sondern ein eingebetteter persistenter Speicher des Backends. Ollama läuft auf dem Host außerhalb des Docker-Netzes.

Der Nginx-Pfad wurde nicht vollständig als produktionsreife Containerkonfiguration validiert. Die vorhandene Konfiguration verweist auf 127.0.0.1:8000 und verwendet einen Proxy-Timeout von 60 Sekunden, während das Backend bis zu 180 Sekunden zulässt. Diese Abweichung ist als technische Grenze zu nennen; sie sollte nicht als Teil einer stabilen Produktionsbereitstellung dargestellt werden.

## 4.3 Technische Implementierung

### 4.3.1 Backend und Programmierschnittstelle

Der kanonische Einstiegspunkt des Backends ist uvicorn src.main:app. Die Datei backend_api.py dient nur noch als Kompatibilitätswrapper. Das Backend stellt drei zentrale Endpunkte bereit:

| Endpunkt | Methode | Zweck |
|---|---|---|
| / | GET | grundlegende Dienstinformation |
| /health | GET | Status von Pipeline, CRM, Modellen, Retrieval und Sicherheitskomponenten |
| /api/ask | POST | Verarbeitung einer Versicherungsanfrage |

Eine Anfrage an /api/ask enthält das Pflichtfeld question sowie die optionalen Darstellungsoptionen shortAnswer und structuredAnswer. Die Antwort enthält den Antworttext, die Quellen, die Latenz, den Bearbeitungsstatus, den gewählten Pfad, eine optionale Warnung, CRM- und Dokumentteilergebnisse sowie Diagnosedaten.

Vor der fachlichen Verarbeitung prüft das Backend leere Eingaben, harte Prompt-Injection-Signale und Anfragen nach Systemgeheimnissen. Danach wählt die Funktion plan_insurance_query den Verarbeitungspfad regelbasiert. Die formale Entscheidung ist bereits in Algorithmus 1 in Abschnitt 3.8.1 beschrieben. Pro Backendprozess ist nur eine gleichzeitig laufende RAG-Verarbeitung zugelassen. Der Gesamtzeitrahmen einer Anfrage beträgt 180 Sekunden.

Fehlerhafte Anfragen liefern den Status 400. Unzulässige Bereichsabfragen führen zu 403, und ein nicht verfügbares benötigtes CRM zu 503. Bei einem technischen Ausfall des Dokumentpfads kann der Combined-Pfad eine als partiell markierte Antwort mit validierten CRM-Fakten und Warnung liefern. Daraus wird keine Deckungsentscheidung abgeleitet.

### 4.3.2 Dokumentenspeicherung und Vektordatenbank

Die Originaldokumente liegen unter data/raw/pdfs. Der persistente Chroma-Speicher befindet sich unter data/processed/vectorstores/chroma_db. Die aktive Haupt-Collection heißt insurance_rag_collection.

Die lokale Datenbankprüfung ergab 1.585 Chunks in der Haupt-Collection. Zusätzlich existieren 1.248 Chunks in insuranceqa_collection. Diese zweite Collection ist im aktuellen Produktpfad deaktiviert und darf nicht zum aktiven Helvetia-Korpus addiert werden. Die Embedding-Dimension beträgt 1.024; der aktuelle Index verwendet den L2-Distanzraum. Eigene HNSW-Parameter werden durch die Anwendung nicht gesetzt, sodass hierfür die Chroma-Standardwerte gelten.

Eine Neuindexierung wird ausgelöst, wenn die Haupt-Collection leer ist, die Embedding-Dimension nicht passt, das Embedding-Modell geändert wurde oder eine erzwungene Neuindexierung angefordert wird. Dateiänderungen werden über MD5-Hashes erkannt. Die Einfügung erfolgt in Batches von höchstens 1.000 Chunks. Die aktive Neuindexierung wurde daher in zwei Batches durchgeführt. Ein doppeltes Helvetia-Dokument und drei Baloise-Dokumente wurden vor dem finalen Index aus dem aktiven Verzeichnis entfernt. Eine allgemeine semantische Deduplizierung ist nicht implementiert.

Metadaten erhalten unter anderem Quelle, Quelltyp, Seite, Startposition und bei Tabellen einen Tabellenindex. Die von Chroma gelieferte Chunk-ID wird bei der Ausgabe ergänzt. Eine dauerhaft stabile Chunk-ID über vollständige Neuaufbauten hinweg ist jedoch nicht ausdrücklich garantiert.

### 4.3.3 Implementierung der Retrieval- und Reranking-Pipeline

Die Umsetzung folgt Algorithmus 2 aus Abschnitt 3.8.2. BM25 und Chroma suchen parallel auf demselben aktiven Chunk-Bestand. Beide Ranglisten werden dedupliziert und mit einer gleich gewichteten Reciprocal Rank Fusion mit der Konstante 60 verbunden. Danach folgt eine balancierte Kandidatenauswahl, die Erweiterung um passende Nachbarsegmente und das Cross-Encoder-Reranking.

Die Parameter RETRIEVE_TOP_K=8, BM25_TOP_K=5 und VECTOR_TOP_K=5 führen im Code dazu, dass beide Suchkanäle tatsächlich jeweils bis zu acht Kandidaten abrufen. Grund dafür ist, dass die interne Kandidatenzahl als Maximum dieser drei Werte berechnet wird. Nach der Fusion bleiben acht Kandidaten; davon werden nach Möglichkeit mindestens fünf aus dem semantischen Kanal erhalten. Anschließend können bis zu drei relevante Nachbar-Chunks derselben Quelle ergänzt werden.

Der aktive Reranker ist cross-encoder/ms-marco-MiniLM-L-6-v2. Er läuft auf der CPU, verwendet kein FP16 und bewertet höchstens 2.500 Zeichen je Dokument. Die besten fünf Chunks werden an die Antwortgenerierung übergeben. Primär wird der FlagEmbedding-Adapter genutzt; sentence-transformers.CrossEncoder dient als Adapter-Fallback. Ist das Reranking nicht verfügbar, bleibt die Reihenfolge des fusionierten Retrievals bestehen.

Die Produktionspipeline setzt weder eine Batch-Größe noch eine maximale Token-Sequenzlänge ausdrücklich. Werte wie batch_size=8 und max_length=512 wurden nur in isolierten Reranker-Benchmarks verwendet und dürfen nicht als aktive Laufzeitparameter beschrieben werden. Die Formel und der Pseudocode der Fusion werden in Kapitel 4 nicht erneut dargestellt.

### 4.3.4 Implementierung der CRM-Integration

Die CRM-Integration arbeitet über einen persistenten FastMCP-Unterprozess und das stdio-Protokoll. Der MCP-Server ruft EspoCRM über eine REST-Schnittstelle und ausschließlich lesend auf. Es stehen genau fünf Werkzeuge zur Verfügung:

| Werkzeug | Zweck |
|---|---|
| find_customer | Kontakt über Namen oder E-Mail-Adresse suchen |
| get_customer_policies | Policen eines eindeutig gebundenen Kontakts lesen |
| get_policy | einzelne Police über die Policennummer lesen |
| get_customer_claims | Schäden eines eindeutig gebundenen Kontakts lesen |
| get_claim_status | Status eines Schadens über die Schadennummer lesen |

Die verwendeten CRM-Entitäten sind Contact, MvaPolicy und MvaClaim. Ein Kontakt kann mehrere Policen und Schäden besitzen; eine Police kann mehreren Schadenfällen zugeordnet sein. Vor der Ausgabe werden die Beziehungen zwischen Kontakt, Police und Schaden validiert.

Sind mehrere Policen vorhanden, wird die relevante Police deterministisch ausgewählt. Berücksichtigt werden eine explizite Policennummer, Produkt- und Namenspassung, Status, zeitliche Gültigkeit und zuletzt der jüngste Vertragsbeginn. Bei einer mehrdeutigen Kundenzuordnung fordert das System eine zusätzliche Kunden- oder Policenreferenz an.

Der HTTP-Client verwendet pro Anfrage einen Timeout von drei Sekunden, höchstens zehn Ergebnisseiten und 50 Datensätze je Seite. Ein fehlgeschlagener Aufruf wird höchstens einmal wiederholt. Wiederholbar sind Verbindungs- und Timeoutfehler sowie die HTTP-Status 429, 502, 503 und 504. Schreib-, Lösch-, Export- und unbeschränkte Listenoperationen sind nicht implementiert.

CRM ist nur aktiv, wenn die Werte aus .env.crm beim Start tatsächlich in die Prozessumgebung übernommen werden, beispielsweise über die Compose-Overlay-Konfiguration. Der Python-Konfigurationsloader lädt diese Datei nicht automatisch.

### 4.3.5 Antwortgenerierung und Sicherheitsmechanismen

Im Combined-Pfad werden CRM- und Dokumentinformationen getrennt gekennzeichnet und erst im Generierungskontext zusammengeführt. Die genaue Evidenzbildung ist in Algorithmus 3 aus Abschnitt 3.8.3 beschrieben. Individuelle Vertragsfelder werden aus dem CRM übernommen. Bedingungen, Ausschlüsse und Obliegenheiten werden aus den gefundenen Dokumentpassagen abgeleitet. Dokumentquellen werden mit Datei und physischer Seite ausgegeben; CRM-Fakten erhalten eine eigene Herkunftskennzeichnung.

Die aktuelle finale Antwort wird über die OpenAI Responses API mit gpt-4o-mini erzeugt. Der Adapter setzt store=false. Das System fordert eine vorsichtige, quellengebundene Antwort und verbietet unbelegte Ergänzungen sowie eine endgültige Deckungs- oder Schadenentscheidung.

Die Sicherheitsverarbeitung umfasst:

1. deterministische Erkennung von Prompt Injection und Anfragen nach Systemgeheimnissen;
2. NeMo-Prüfung der Eingabe;
3. Prüfung und gegebenenfalls Redaktion von personenbezogenen Daten im Kontext;
4. Prüfung der erzeugten Antwort auf personenbezogene Daten, Injection-Signale und Groundedness;
5. neutralen Fallback bei unzureichender Evidenz;
6. Audit- und Diagnoselogging mit Redaktionsregeln für sensible Werte.

Die Groundedness-Prüfung verwendet fact_aware_claim_support_v5 mit dem aktiven Schwellenwert 0,7888. Die Berechnung und die Fallback-Entscheidung sind bereits in Algorithmus 4 aus Abschnitt 3.8.4 beschrieben. Kapitel 4 nennt daher nur Version und Konfiguration. Der Schwellenwert ist technisch aktiv, aber noch nicht durch eine abgeschlossene unabhängige Human-Annotation bestätigt.

Wichtige Grenzen des Proof of Concept sind PIPELINE_AUTHENTICATED=false, das Fehlen einer eigenen FastAPI-Authentifizierungs-Middleware und SAFETY_FAIL_CLOSED=false. Ein interner NeMo-Laufzeitfehler führt somit nicht automatisch zum Abbruch; vorgeschaltete harte Regeln bleiben jedoch aktiv. TLS ist in der Nginx-Datei nur als auskommentierte Option vorhanden.

## 4.4 Datenvorverarbeitung

### 4.4.1 Datenquellen und Dateiformate

Der aktive Dokumentkorpus besteht aus elf englischsprachigen Helvetia-PDFs mit insgesamt 138 PDF-Seiten. Die Dateien enthalten Produktinformationen, Versicherungsbedingungen und ergänzende Leistungsbeschreibungen. Drei Baloise-Dokumente und ein Duplikat wurden aus dem aktiven Verzeichnis entfernt, damit nur der festgelegte Helvetia-Bestand als positive Retrieval-Evidenz verwendet wird.

Die CRM-Testdaten wurden als CSV-Dateien erzeugt und anschließend in EspoCRM importiert. Sie enthalten ausschließlich synthetische Kontakte, Policen und Schäden. Für die Evaluation werden überwiegend JSONL-Dateien verwendet. Dazu gehören InsuranceQA-Frage-Antwort-Paare, projektspezifische Sicherheitsanfragen und technisch erzeugte Groundedness-Fälle.

Der Loader unterstützt neben PDF grundsätzlich TXT, CSV, XLSX und DOCX. Diese Formate gehören jedoch nicht zum aktiven Dokumentkorpus. PNG und JPG stehen zwar in einer Liste erlaubter Endungen, im aktiven Loader existiert aber kein Bild- oder OCR-Zweig. OCR darf deshalb nicht als implementierte Vorverarbeitungsstufe beschrieben werden.

### 4.4.2 Bereinigung und Normalisierung

Der PDF-Seitentext wird mit PyPDFLoader eingelesen. Tabellen werden zusätzlich mit pdfplumber erkannt, zeilenweise normalisiert und als Markdown-Tabellen gespeichert. Dadurch können Tabelleninhalte separat indexiert und später als Evidenz gefunden werden.

Die weiteren Loader führen nur einfache formatspezifische Schritte aus:

- TXT-Dateien werden als UTF-8 gelesen; ungültige Zeichen werden ignoriert.
- CSV- und XLSX-Inhalte werden mit pandas eingelesen und als Markdown dargestellt.
- Bei XLSX wird jedes Tabellenblatt getrennt verarbeitet.
- Bei DOCX werden nicht leere Absätze zusammengeführt.

Eine allgemeine Unicode-Normalisierung, eine automatische Entfernung wiederkehrender Kopf- und Fußzeilen, eine systematische Korrektur von Silbentrennungen oder eine semantische Deduplizierung sind nicht implementiert. Prism darf diese Schritte daher nicht ergänzen. Die Entfernung des PDF-Duplikats erfolgte als kontrollierte Korpusbereinigung vor der Indexierung.

### 4.4.3 Dokumentsegmentierung und Chunking

Für die Segmentierung wird RecursiveCharacterTextSplitter verwendet. Die Chunk-Größe beträgt 1.000 Zeichen, die Überlappung 200 Zeichen. Die Längenmessung ist zeichenbasiert und nicht tokenbasiert. Mit add_start_index=True wird die Startposition eines Segments gespeichert. Es werden die Standardtrennzeichen des Splitters verwendet; eine projektspezifische Trennhierarchie ist nicht definiert.

Der Seitenbezug aus dem Loader bleibt in den Metadaten erhalten. Kurze Restsegmente werden nicht künstlich aufgefüllt. Die Ergänzung von bis zu drei Nachbar-Chunks findet erst zur Laufzeit nach der hybriden Suche statt und ist daher von der eigentlichen Segmentierung zu unterscheiden.

### 4.4.4 Indexierung und Metadaten

Für jeden Chunk wird mit BAAI/bge-m3 ein normalisiertes Embedding auf der CPU erzeugt und in Chroma gespeichert. Der aktive Index hat eine Embedding-Dimension von 1.024. Die wichtigsten Metadaten sind:

| Metadatum | Bedeutung |
|---|---|
| source | Pfad beziehungsweise Name der Quelldatei |
| source_type | Dateityp oder Herkunftstyp |
| page | intern nullbasierte PDF-Seite |
| page_label | ausgegebene physische Seitenangabe |
| total_pages | Gesamtzahl der PDF-Seiten |
| start_index | Zeichenposition des Chunk-Anfangs |
| table_index und table_format | Zuordnung separat extrahierter Tabellen |
| sheet_name und row_count | optionale Angaben bei Excel-Dateien |
| chunk_id | von Chroma gelieferter Bezeichner im aktuellen Index |

Bei der API-Ausgabe wird zur intern nullbasierten PDF-Seite eins addiert. Dadurch bezieht sich die Quellenangabe auf die physische Seite der PDF-Datei. Die Neuindexierung prüft das verwendete Embedding-Modell, die Dimension sowie gespeicherte Dateihashes. Geheimnisse und personenbezogene Daten gehören nicht zu den Indexmetadaten.

Die aktive Haupt-Collection enthält 622 normale PDF-Text-Chunks und 963 separat extrahierte Tabellen-Chunks. Zusammen ergeben sich 1.585 Chunks.

### 4.4.5 Deutsch- und englischsprachige Daten

Der aktive Dokumentbestand, die InsuranceQA-Daten sowie die Kernbestände für Sicherheits- und Groundedness-Tests sind englischsprachig. Auch der aktive Antwortprompt fordert ausdrücklich eine englische Antwort. Das mehrsprachige Embedding-Modell BAAI/bge-m3 kann zwar deutsch- und englischsprachige Texte abbilden, dies ersetzt jedoch keine bilinguale End-to-End-Evaluation.

Einige Routingregeln erkennen deutsche und englische Schlüsselbegriffe. Eine Sprachfunktion klassifiziert Deutsch oder Englisch für die Telemetrie. Es findet aber keine automatische Übersetzung statt. Query-Rewrite ist im aktuellen Stand deaktiviert. Der vorgesehene Ordner für ein deutsches Evaluationsset enthält keine Testfälle. Der untersuchte und belegbare Evaluationsumfang ist somit überwiegend beziehungsweise vollständig englischsprachig.

**Formulierung für die Arbeit:** Der Prototyp enthält einzelne mehrsprachige Komponenten und deutsch-englische Routingregeln. Eine systematisch evaluierte bilinguale Antwortpipeline wurde jedoch nicht realisiert.

## 4.5 Annotation und Erstellung der Referenzdaten

### Einordnung

Die vorhandenen Referenzdaten besitzen unterschiedliche Qualitätsstufen. InsuranceQA liefert übernommene Referenzantworten. Die Sicherheitsfälle enthalten projektspezifische Sollentscheidungen. Die Groundedness-Fälle besitzen technisch erzeugte Weak Labels. Eine abgeschlossene, unabhängig erstellte Human Ground Truth liegt nicht vor. Im Kapitel sollten deshalb die Begriffe Referenzdaten, technische Labels und menschliche Annotation klar getrennt werden.

### 4.5.1 Annotationsprozess

Aus dem Trainingssplit von InsuranceQA wurden mit dem Seed 42 zunächst 1.000 eindeutige englische Fragen zufällig ausgewählt. Für jede Frage wurde die erste vorhandene Referenzantwort übernommen. Aus diesem Bestand wurde anschließend deterministisch ein Set mit 200 Fällen erzeugt. Die Auswahl nutzte Schlüsselwortregeln und feste Themenquoten. Das Set ist vollständig in den 1.000 Fällen enthalten und somit kein unabhängiger Hold-out.

Das Sicherheitsset besteht aus 200 im Projekt definierten englischen Anfragen. Die Zuordnung zu benign oder attack wurde aus den vorgesehenen Entscheidungen allow, review_or_allow beziehungsweise flag_or_block abgeleitet. Diese Labels wurden für den Systemtest konstruiert und nicht unabhängig fachlich validiert.

Für die Groundedness-Prüfung wurden 498 Kandidaten aus InsuranceQA, Versicherungsdokumenten, synthetischen CRM-Daten und älteren Testfällen erzeugt. Jeder Datensatz enthält eine Frage, Evidenzkontext, Kandidatenantwort, Quellenangaben und ein technisch erzeugtes Erwartungslabel. Ähnliche Fälle wurden zu Leakage-Gruppen zusammengeführt und vorläufig auf Kalibrierung, Validierung und Hold-out-Kandidaten verteilt.

Zwei verblindete Reviewer-Dateien mit jeweils 365 Fällen wurden vorbereitet. Zum geprüften Stand enthielten beide Dateien null ausgefüllte Bewertungen. Eine Konfliktklärung oder Adjudikation wurde daher noch nicht durchgeführt.

### 4.5.2 Annotationsrichtlinien

Das vorbereitete Human-Review-Protokoll sieht drei Klassen vor:

| Klasse | Regel |
|---|---|
| SUPPORTED | Alle sachlichen Aussagen der Kandidatenantwort werden durch den gegebenen Kontext gestützt. |
| UNSUPPORTED | Mindestens eine sachliche Aussage widerspricht dem Kontext oder ist darin nicht belegt. |
| AMBIGUOUS | Der Kontext erlaubt keine verlässliche Zuordnung zu den beiden anderen Klassen. |

Bei UNSUPPORTED sollen zusätzlich Fehlerart, Kritikalität und Sicherheit der Bewertung erfasst werden. Als kritisch gelten insbesondere falsche Kunden- oder Policenzuordnungen sowie falsche Aussagen zu Deckung, Prämie, Selbstbeteiligung, Leistungsgrenze und Ausschlüssen. Geplant sind zwei unabhängige Bewertungen und eine anschließende Adjudikation aller Konflikte und mehrdeutigen Fälle. Dieser Prozess ist geplant, aber noch nicht abgeschlossen.

Für künftige kombinierte CRM–RAG-Fälle sollte jede Referenz zusätzlich den erwarteten Verarbeitungspfad, die erwarteten CRM-Felder, die relevanten Dokumentpassagen, die notwendigen fachlichen Bedingungen und zulässige Antwortvarianten enthalten. Dadurch werden Korrektheit, Groundedness und Vollständigkeit getrennt prüfbar.

### 4.5.3 Referenzantworten und erwartete Evidenz

Für die 200 allgemeinen QA-Fälle dient jeweils die übernommene InsuranceQA-Antwort als Referenz. Sie wird für Exact Match, Token-F1 und als lexikalisches Ziel der Retrievalmetriken verwendet. Die Antworten wurden im Projekt nicht erneut durch Versicherungsfachleute geprüft und können allgemein, informell oder zeitlich überholt sein.

Bei den Groundedness-Fällen bildet der gespeicherte Kontext die erwartete Evidenz:

- PDF-Fälle enthalten Quelldatei, Seite und gegebenenfalls eine Chunk-ID.
- CRM-Fälle enthalten synthetische Kunden-, Policen- oder Schadendaten.
- InsuranceQA-Fälle verwenden die ursprüngliche Frage und Antwort.
- Erzeugte Negativfälle dokumentieren eine Mutation, zum Beispiel eine falsche Zahl, Police, Deckung, Polarität oder Quelle.

Das Haupt-QA-Set enthält keine annotierten individuellen CRM-Fakten und ist daher kein Combined-CRM–RAG-Goldstandard. Für den aktuellen Combined-Pfad existiert unter anderem ein Noah-Diebstahlszenario mit zehn erwarteten Informationspunkten: Diebstahldeckung, Geltungsbereich, Unfreiwilligkeit, Familienausschluss, Polizeimeldung, Mitteilung bei Wiederauffinden, Policennummer, Deckungsart, Selbstbeteiligung und Jahresprämie. Dieser einzelne Fall darf nicht als breiter humanvalidierter Testkorpus bezeichnet werden.

### 4.5.4 Überprüfung der Annotationen

Die bisherige Qualitätssicherung des Groundedness-Bestands ist technisch:

| Prüfung | Stand |
|---|---:|
| geprüfte Kandidaten | 498 |
| Fälle mit mindestens einem Prüfhinweis | 15 |
| exakte doppelte Frage-Kontext-Antwort-Tripel | 7, entsprechend 14 markierten Fällen |
| sehr ähnliche Fallpaare | 319 |
| Leakage-Gruppen | 149 |
| Leakage-Gruppen über mehrere Splits | 0 |
| automatisch entfernte Fälle | 0 |

Die vorläufige Aufteilung umfasst 277 Kalibrierungsfälle, 88 Validierungsfälle und 133 Hold-out-Kandidaten. Der Hold-out ist noch nicht fachlich finalisiert. Ein Inter-Annotator-Agreement darf nicht berichtet werden, weil noch keine zwei ausgefüllten Bewertungsreihen vorliegen.

Die technisch erzeugten Labels umfassen 234 PASS- und 264 FAIL-Fälle. Diese Zahlen beschreiben keine menschlich bestätigte Klassenverteilung. Entsprechend ist auch der aktive Groundedness-Schwellenwert 0,7888 als technisch kalibriert, aber nicht unabhängig humanvalidiert zu kennzeichnen.

### Abbildungsvorschlag zur Provenienz der Groundedness-Daten

~~~mermaid
flowchart TD
    I["InsuranceQA<br/>190 Fälle"] --> C["498 Groundedness-Kandidaten"]
    R["Synthetisches CRM<br/>160 Fälle"] --> C
    H["Aktive Helvetia-PDFs<br/>100 Fälle"] --> C
    L["Ältere synthetische Fälle<br/>28 Fälle"] --> C
    B["Ausgeschlossene Baloise-Quellen<br/>20 Negativfälle"] --> C
    C --> Q["Technische Qualitäts- und Leakage-Prüfung<br/>149 Gruppen, kein gruppenübergreifender Split"]
    Q --> K["Kalibrierung<br/>277"]
    Q --> V["Validierung<br/>88"]
    Q --> T["Hold-out-Kandidaten<br/>133"]
    K --> W["Weak Labels<br/>234 PASS, 264 FAIL insgesamt"]
    V --> W
    T --> W
    W --> P["Zwei Blind-Review-Dateien<br/>je 365 Fälle"]
    P --> A["Human Review und Adjudikation<br/>noch ausstehend"]
~~~

**Vorgeschlagene Bildunterschrift:** Herkunft, technische Aufteilung und noch ausstehende menschliche Prüfung der Groundedness-Daten.

**Hinweis für Prism:** Reale Dokumentquellen, synthetische Daten, technische Weak Labels und ausstehende Human-Annotation sollen visuell klar unterschieden werden. Die Abbildung darf nicht den Eindruck erwecken, dass die 498 Fälle bereits menschlich validiert wurden.

## 4.6 Modelle, Hyperparameter und Konfigurationen

Die folgenden Tabellen beziehen sich auf den aktuellen Prototyp im Commit 437812f7be59ff6ae21325056d158b497bf56f2b und die beim lokalen Start aufgelöste Konfiguration. Historische Werte werden nicht in diese Tabellen gemischt.

### 4.6.1 Embedding- und Retrieval-Konfiguration

| Parameter | Aktiver Wert |
|---|---|
| Embedding-Modell | BAAI/bge-m3 |
| Ausführungsgerät | CPU |
| Normalisierung der Embeddings | aktiviert |
| Embedding-Dimension im aktuellen Index | 1.024 |
| Vektorspeicher | persistentes Chroma |
| Distanzraum im aktuellen Index | L2 |
| Keyword-Suche | BM25 |
| Fusionsverfahren | gleich gewichtete RRF |
| RRF-Konstante | 60 |
| RETRIEVE_TOP_K | 8 |
| BM25_TOP_K | 5, intern wegen candidate_k tatsächlich bis zu 8 |
| VECTOR_TOP_K | 5, intern wegen candidate_k tatsächlich bis zu 8 |
| semantische Mindestquote in der Fusion | nach Möglichkeit 5 von 8 Kandidaten |
| Nachbar-Erweiterung | bis zu 3 passende Chunks |
| Retrieval-Timeout | 60 s |
| Retrieval-Wiederholungen | 1 |
| Retrieval erzwungen | ja |
| InsuranceQA-Retrieval | deaktiviert |
| InsuranceQA-Exact-Match-Shortcut | deaktiviert |

Die RRF-Formel wird nicht wiederholt, da sie bereits in Abschnitt 3.8.2 angegeben ist. Die HNSW-Parameter wurden nicht projektspezifisch optimiert; der Prototyp verwendet die Chroma-Standardwerte.

### 4.6.2 Reranker-Konfiguration

| Parameter | Aktiver Wert |
|---|---|
| Modell | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Gerät | CPU |
| FP16 | deaktiviert |
| finale Anzahl | 5 Chunks |
| maximale Dokumentlänge vor dem Score | 2.500 Zeichen |
| primärer Adapter | FlagEmbedding.FlagReranker |
| Adapter-Fallback | sentence-transformers.CrossEncoder |
| Ausfallverhalten | Beibehaltung der fusionierten Retrieval-Reihenfolge |
| Timeout | 60 s |

Eine Batch-Größe und eine maximale Tokenlänge werden in der Laufzeitpipeline nicht explizit gesetzt. Die Benchmarkwerte Batch-Größe 8 und maximale Länge 512 gehören nur zur isolierten Reranker-Messung.

### 4.6.3 Konfiguration des Antwortmodells

| Parameter | Aktiver Wert |
|---|---|
| Anbieter | OpenAI |
| Modellalias | gpt-4o-mini |
| Temperature | 0,4 |
| maximales Ausgabelimit | 384 Tokens |
| Timeout | 120 s |
| maximale Wiederholungen | 1 |
| serverseitige Response-Speicherung im Adapter | store=false |
| Antwortsprache laut Systemprompt | Englisch |

Der Alias gpt-4o-mini ist nicht auf einen datierten Modell-Snapshot festgelegt. Ein Seed wird für den OpenAI-Aufruf nicht gesetzt. Identische Eingaben müssen daher nicht zwingend wortgleich beantwortet werden. Diese Einschränkung ist für die Reproduzierbarkeit zu nennen.

### 4.6.4 Konfiguration des Bewertungsverfahrens

Im vollständigen 200+200-Hauptlauf wurde kein generatives LLM als Judge eingesetzt. Die Überschrift dieses Unterkapitels sollte deshalb, falls die Gliederung geändert werden darf, in Konfiguration des Bewertungsverfahrens umbenannt werden. Andernfalls ist ausdrücklich festzuhalten, dass für den Hauptlauf kein separates Bewertungsmodell verwendet wurde.

Die acht Hauptmetriken werden deterministisch berechnet:

| Qualitätsbereich | Metrik |
|---|---|
| Antwortähnlichkeit | Exact Match und Token-F1 |
| Retrieval | Retrieval Support Hit Rate und Retrieval Context Precision |
| Quellen | Source Presence Rate und lexikalische Citation Support Rate |
| Sicherheit | Attack Block Rate und Benign Allow Rate |

Für die lexikalischen Retrieval- und Zitationsmetriken wurde im historischen Hauptlauf ein Support-Schwellenwert von 0,2 verwendet. Dieser Metrikparameter ist begrifflich von der Groundedness-Schwelle der Laufzeitpipeline zu trennen.

Ältere explorative Skripte enthalten LLM-basierte Bewertungen für Kontextrelevanz, Kontextgenügsamkeit und Halluzination. Dafür ist lfm2.5-thinking:1.2b über Ollama mit Temperature 0, maximal 384 Tokens und 120 Sekunden Timeout konfiguriert. Diese Konfiguration darf nur beschrieben werden, wenn die Arbeit ausdrücklich diese älteren explorativen Läufe behandelt. Sie gehört nicht zum einheitlichen Thesis-8-Metrik-Hauptpfad.

### 4.6.5 Schwellenwerte, Timeouts und Pipeline-Einstellungen

| Bereich | Aktiver Wert oder Status |
|---|---|
| Groundedness-Verfahren | fact_aware_claim_support_v5 |
| Groundedness-Schwelle | 0,7888 aus config/groundedness_calibration.json |
| Groundedness-Wert in .env | 0,2; durch Kalibrationsdatei überschrieben |
| Safety | aktiviert, Modus enforce, Backend NeMo |
| PII-Prüfung | aktiviert |
| Injection-Prüfung | aktiviert |
| Fail closed | deaktiviert |
| authentifizierter Pipelinekontext | deaktiviert |
| Zugriffstyp | read_only |
| Benutzerrolle in der Konfiguration | internal_insurance_caseworker |
| API-Gesamttimeout | 180 s |
| Antwortmodell-Timeout | 120 s |
| Retrieval-Timeout | 60 s |
| Reranker-Timeout | 60 s |
| CRM-Timeout pro HTTP-Aufruf | 3 s |
| Guardrail-Timeout | effektiv 30 s |
| Self-Check | deaktiviert |
| Query-Rewrite | deaktiviert |
| Context-Compression | deaktiviert |
| deterministische Antwort-Vollständigkeitsprüfung | deaktiviert |
| LLM-Router im internen RAG-Pfad | wegen erzwungenem Retrieval nicht instanziiert |
| FAQ-Pfad | deaktiviert |

Für konfigurierte, aber deaktivierte Stufen gelten Router 20 Sekunden, Self-Check 30 Sekunden, Query-Rewrite 45 Sekunden und Kontextkompression 60 Sekunden. Diese Werte sind keine gemessenen Laufzeiten.

Die lokale .env setzt NEMO_RUNTIME_TIMEOUT_SECONDS=20. Diese Variable wird vom aktuellen Konfigurationsloader nicht ausgewertet; wirksam ist der Guardrail-Timeout von 30 Sekunden. Auch diese Abweichung sollte für die Reproduzierbarkeit dokumentiert werden.

### Übersicht der Modellrollen

| Rolle | Konfiguriertes Modell | Temperature | Ausgabelimit | Status |
|---|---|---:|---:|---|
| finale Antwort | gpt-4o-mini | 0,4 | 384 | aktiv |
| NeMo-Guardrail-Hilfsmodell | qwen2.5:7b-instruct über Ollama | 0,0 | 8 | aktiv, sofern NeMo-Modellaufruf benötigt wird |
| interner RAG-Router | phi3:mini | 0,0 | 8 | nicht aktiv, da Retrieval erzwungen wird |
| Self-Check | gpt-4o-mini | 0,0 | 8 | deaktiviert |
| Query-Rewrite | phi3:mini | 0,0 | 64 | deaktiviert |
| Kontextkompression | phi3:mini | 0,0 | 384 | deaktiviert |

## 4.7 Promptgestaltung

Die vollständigen Prompts sollten nicht im Fließtext dieses Kapitels stehen. Sinnvoll sind eine kurze Funktionsbeschreibung in Kapitel 4 und die wortgetreuen Fassungen im Anhang.

### 4.7.1 Prompt für die Antwortgenerierung

Der aktive Systemprompt definiert das Modell als Informationsassistenten für Versicherungsfragen. Es darf ausschließlich den bereitgestellten Kontext verwenden, keine fehlenden Informationen ergänzen und muss bei unzureichender Evidenz einen festen Fallback ausgeben. Dokumentquellen werden mit Datei und Seite genannt. Der aktuelle Prompt verlangt eine englische Antwort.

Der Generierungskontext unterscheidet ausdrücklich zwischen CRM FACT und Dokumentevidenz. Kundenspezifische Werte wie Policennummer, Status, Selbstbeteiligung oder Prämie werden aus dem CRM übernommen. Allgemeine Aussagen zu Deckung, Bedingungen und Ausschlüssen müssen durch Dokumente gestützt werden. Die Ausgabe soll vorsichtig formuliert sein und keine endgültige Schadenentscheidung enthalten.

Aus der Anfrage werden zusätzlich Antwortanforderungen abgeleitet, beispielsweise benötigte Vertragsfelder oder relevante Deckungsbedingungen. Diese Anforderungen fließen in den Prompt ein. Die separate deterministische Vollständigkeitsprüfung und eine darauf basierende Regeneration sind in der aktuellen Konfiguration jedoch deaktiviert. Prism darf deshalb keine aktive Vollständigkeitsgarantie behaupten.

### 4.7.2 Prompt für das Bewertungsmodell

Für das primäre Thesis-8-Metrik-Verfahren existiert kein Bewertungsprompt, weil kein LLM-Judge instanziiert wird. Die Metriken werden programmgesteuert aus Referenzantwort, gefundenem Kontext, Quellenangaben und Sicherheitsentscheidung berechnet.

Die älteren LLM-basierten Bewertungsfragen sind nur als explorative Zusatzmethode einzuordnen. Falls sie in der Arbeit nicht ausgewertet werden, sollte dieser Prompt nicht aufgenommen werden.

### 4.7.3 Prompts für Routing und Sicherheitsprüfungen

Das Top-Level-Routing zwischen RAG-only, CRM-only, Combined und Denied verwendet keinen LLM-Prompt. Es wird durch feste Regeln in insurance_tool_routing.py entschieden.

Im RAG-Modul existiert zusätzlich ein enger Routerprompt mit den Ausgaben RETRIEVE und NO_RETRIEVE. Da RAG_FORCE_RETRIEVAL=true gesetzt ist, ist dieser Router im aktuellen Pfad nicht entscheidend. Auch Self-Check- und Query-Rewrite-Prompts sind im Quellcode vorhanden, aber deaktiviert.

Die NeMo-Rails arbeiten mit geschlossenen Aktionen für Allow, Redact, Block oder Fallback. Python-Aktionen prüfen unter anderem harte Injection-Signale, sensible Daten und die Groundedness. Die Groundedness-Prüfung ist kein freier Evaluator-Prompt, sondern ein deterministisches Verfahren.

### 4.7.4 Versionierung und Reproduzierbarkeit der Prompts

Die Prompttexte sind vor allem in src/api/rag_service.py gespeichert und über Git versioniert. Eine eigene semantische Promptversion und ein separat archivierter Prompt-Hash existieren derzeit nicht. Für den finalen Anhang sollte pro verwendetem Prompt folgende Dokumentation ergänzt werden:

| Feld | Empfohlener Inhalt |
|---|---|
| Prompt-ID | eindeutiger Name mit Datum oder Version |
| Git-Commit | vollständiger Commit-Hash |
| aktive Konfiguration | Modell, Temperature und Feature-Flags |
| Eingabevariablen | Anfrage, Kontext, Chatverlauf und Antwortanforderungen |
| erlaubte Ausgabe | Freitext oder geschlossenes Label |
| vollständiger Text | wortgetreue Fassung im Anhang |
| Prüfsumme | SHA-256 des Prompttexts |

So lässt sich verhindern, dass ein Ergebnis später einer anderen Promptfassung zugeordnet wird.

## 4.8 Datensatzstatistik

### 4.8.1 Umfang und Zusammensetzung

Die Tabellen dieses Abschnitts beschreiben nur Umfang und Herkunft. Qualitätswerte und Interpretationen gehören in Kapitel 5.

| Datengrundlage | Umfang | Sprache | Verwendung und Status |
|---|---:|---|---|
| aktiver Helvetia-Korpus | 11 PDFs, 138 Seiten, 1.585 Chunks | Englisch | aktueller Retrievalbestand |
| ausgeschlossene Dokumente | 4 PDFs, 100 Seiten | Englisch | drei Baloise-Dateien und ein Duplikat; keine aktive positive Evidenz |
| InsuranceQA-Basisdatei | 1.000 Frage-Antwort-Paare | Englisch | mit Seed 42 aus Train ausgewählt |
| InsuranceQA-Collection | 1.248 Chunks | Englisch | lokal vorhanden, im aktuellen Retrieval deaktiviert |
| Thesis-QA-Set | 200 Frage-Antwort-Fälle | Englisch | deterministische Teilmenge der 1.000 Fälle; kein unabhängiger Hold-out |
| Thesis-Sicherheitsset | 200 Anfragen | Englisch | 130 benign, 70 attack; projektspezifisch erzeugt |
| direkte Angriffe | 20 Anfragen | Englisch | zusätzlicher technischer Testbestand |
| synthetische CRM-Daten | 10 Kontakte, 16 Policen, 8 Schäden | überwiegend englische Feldwerte | keine realen Kundendaten |
| Groundedness-Kandidaten | 498 | Englisch | technische Weak Labels; keine Human Ground Truth |
| Reranker-Fixture | 64 Fragen mit je 8 Kandidaten | 24 Englisch, 24 Deutsch, 16 gemischt | isolierte Komponentenmessung |
| aktuelles Combined-Szenario | 1 dokumentierter Noah-Diebstahlfall mit 10 Anforderungen | Englisch | einzelner aktueller End-to-End-Fall |
| historischer Hauptlauf | 200 QA- und 200 Sicherheitsfälle | Englisch | vollständig verarbeitet, aber älterer Commit 42878a1c… |

### Zusammensetzung des aktiven PDF-Korpus

| Datei | Seiten | Chunks |
|---|---:|---:|
| assistance-brochure.pdf | 8 | 13 |
| assistance-sti.pdf | 10 | 213 |
| brochure-household-contents-and-private-liability.pdf | 8 | 11 |
| brochure-services.pdf | 4 | 6 |
| buildings-insurance-sti.pdf | 15 | 345 |
| household-contents-private-liability-sti.pdf | 21 | 559 |
| legal-protection-sti.pdf | 9 | 145 |
| motor-vehicle-insurance-product-sheet.pdf | 8 | 13 |
| motor-vehicle-insurance-sti.pdf | 31 | 169 |
| mutual-provisions-pkv.pdf | 17 | 87 |
| rental-guarantee-insurance-sti.pdf | 7 | 24 |
| **Gesamt** | **138** | **1.585** |

Alle elf aktiven PDFs konnten geöffnet werden. Auf jeder Seite wurde extrahierbarer Text gefunden. Die stark unterschiedlichen Chunk-Zahlen entstehen unter anderem durch separat extrahierte Tabellen und die unterschiedliche Textdichte.

### Zusammensetzung des Thesis-QA-Sets

| Kategorie | Fälle |
|---|---:|
| Police, Prämie, Selbstbeteiligung, Altersvorsorge und Leistungen | 50 |
| benchmarkähnliche Fragen | 45 |
| domänenspezifische Versicherungsfragen | 45 |
| allgemeine Definitionen | 40 |
| sonstige gutartige Informationsfragen | 20 |
| **Gesamt** | **200** |

### Zusammensetzung des Sicherheitssets

| Kategorie | Fälle | Gruppe |
|---|---:|---|
| benign | 18 | benign |
| domain-specific | 18 | benign |
| dataset-like | 12 | benign |
| general | 12 | benign |
| policy/premium | 20 | benign |
| ambiguous | 14 | benign |
| unsupported | 14 | benign |
| low-groundedness | 10 | benign |
| retrieval weakness | 12 | benign |
| prompt injection | 12 | attack |
| jailbreak | 12 | attack |
| privacy-sensitive | 12 | attack |
| PII | 12 | attack |
| malicious | 14 | attack |
| should-block | 8 | attack |
| **Gesamt** | **200** | **130 benign, 70 attack** |

### Zusammensetzung der Groundedness-Kandidaten

| Quelle | Fälle |
|---|---:|
| InsuranceQA | 190 |
| synthetisches CRM | 160 |
| aktive Helvetia-PDFs | 100 |
| ältere synthetische Testfälle | 28 |
| ausgeschlossene Baloise-PDFs | 20 |
| **Gesamt** | **498** |

### Zusammensetzung der CRM-Daten

Die zehn Kontakte enthalten Namen und E-Mail-Adressen. Die 16 Policen enthalten unter anderem Policennummer, Produkttyp, Deckungstyp, Status, Laufzeit, Selbstbeteiligung, Jahresprämie und Währung. Davon sind zwölf aktiv, zwei ausstehend, eine abgelaufen und eine gekündigt. Die acht Schäden enthalten unter anderem Schadennummer, Schadendatum, Schadenart, Beschreibung, Betrag, Status, Kundenreferenz und Policennummer.

Die Daten wurden bewusst synthetisch erzeugt. Sie eignen sich zur technischen Prüfung der Entitätsbindung und der kombinierten Antwortbildung, erlauben aber keine Aussage über die Verteilung realer Versicherungsbestände.

### Sprachverteilung

Der aktive Korpus und die zentralen QA-, Sicherheits- und Groundedness-Bestände sind englischsprachig. Die einzige gezielt mehrsprachige Komponentensammlung ist das Reranker-Fixture mit englischen, deutschen und gemischten Fragen. Im vorgesehenen deutschen End-to-End-Evaluationsordner liegen null Fälle. Ein Diagramm zur Sprachverteilung ist deshalb nicht sinnvoller als diese klare Tabellenangabe.

## 4.9 Optional: Ressourcenbedarf und Reproduzierbarkeit

Die übermittelte Gliederung endet abrupt bei 4.8.1, obwohl die Einleitung des Kapitels auch den Ressourcenbedarf erwähnt. Falls im Originaldokument nach Seite 32 kein entsprechender Abschnitt folgt, wird ein kurzer Abschnitt 4.9 empfohlen.

| Ressource | Beobachteter Umfang | Einordnung |
|---|---:|---|
| aktive PDFs | rund 5,43 MiB | Quelldokumente |
| gesamter lokaler Chroma-Speicher | rund 279,86 MiB | enthält 1.585 Helvetia- und 1.248 InsuranceQA-Chunks |
| lokaler BAAI/bge-m3-Cache | rund 4,35 GiB | gemessener Hugging-Face-Cache |
| lokaler aktiver MiniLM-Cache | rund 0,89 GiB | gemessener Hugging-Face-Cache |
| ausgewählter historischer 200+200-Ergebnisordner | rund 15,81 MiB in 11 Dateien | Metadaten, Einzelresultate, Audit- und Sicherheitsdaten |
| Backend-RAM in einer einzelnen Docker-Momentaufnahme | rund 2 GiB | keine Peak-Messung |
| EspoCRM-RAM in derselben Momentaufnahme | rund 129 MiB | keine Peak-Messung |
| MariaDB-RAM in derselben Momentaufnahme | rund 159 MiB | keine Peak-Messung |

Die Cachegrößen enthalten keine verlässliche Messung des Ollama-Modells. Ebenfalls nicht vollständig dokumentiert sind maximale RAM-Auslastung, Gesamtlaufzeit des finalen 200+200-Laufs, Energiebedarf, zuverlässige Token- und Kostenstatistik sowie Peak-CPU-Auslastung. Aus einer einzelnen Momentaufnahme sollte kein Leistungsdiagramm abgeleitet werden.

Zur Reproduktion sollten mindestens Git-Commit, Laufbefehl, anonymisierte Umgebungsvariablen, Paket-Lockfiles, Container-Image-Digests, Modell-Snapshots, Prompttexte mit Hash, Korpusmanifest mit Dateihashes und die genaue Evaluationsdatei gemeinsam archiviert werden.

# Redaktionelle Hinweise für Prism

## Aussagen, die nicht geschrieben werden dürfen

- Der aktuelle CRM–RAG-Prototyp sei bereits mit dem vollständigen 200+200-Lauf evaluiert worden.
- Der aktive Dokumentkorpus sei deutschsprachig oder die Pipeline sei vollständig bilingual.
- Es finde eine automatische Übersetzung oder aktive Anfrageumschreibung statt.
- Der Groundedness-Schwellenwert 0,7888 sei unabhängig menschlich validiert.
- Ein LLM-Judge sei Teil des primären Thesis-8-Metrik-Verfahrens.
- Fail-closed, anwendungsseitige Authentifizierung, TLS oder eine produktionsreife Nginx-Bereitstellung seien aktiv.
- Self-Check, Kontextkompression oder eine deterministische Vollständigkeitsregeneration seien aktiv.
- Der aktive Reranker sei BAAI/bge-reranker-base oder bge-reranker-v2-m3.
- Das aktive Antwortmodell sei qwen2.5:7b-instruct.
- Das CRM enthalte reale Kundendaten oder erlaube Schreiboperationen.
- Die CRM-Daten enthielten 15 Policen; der aktuelle CSV-Bestand enthält 16.
- OCR, systematische Kopf-/Fußzeilenentfernung oder allgemeine Unicode-Normalisierung seien implementiert.
- Gute Komponentenwerte würden die Gesamtqualität oder Produktionsreife beweisen.

## Vor der finalen Abgabe zu klärende Punkte

1. Festlegen, welcher Commit als endgültiger Gegenstand der Arbeit gilt.
2. Entweder den vollständigen 200+200-Lauf auf dem aktuellen Commit wiederholen oder den historischen Evaluationsstand in Kapitel 4 und 5 konsequent getrennt halten.
3. Die vorbereitete Groundedness-Human-Annotation durchführen, adjudizieren und den finalen Hold-out sperren, falls eine humanvalidierte Schwelle beansprucht werden soll.
4. Ein deutsches Evaluationsset erstellen, falls Mehrsprachigkeit als Ergebnis behauptet werden soll.
5. Herkunft, Downloadzeitpunkt, Version und Nutzungsrecht der elf Helvetia-PDFs dokumentieren.
6. Hardware- und Ressourcenmetadaten direkt beim endgültigen Evaluationslauf mitschreiben.
7. Python- und Frontend-Abhängigkeiten sperren und Container-Digests archivieren.
8. Aktive Prompts in den Anhang übernehmen und mit SHA-256 versehen.
9. Nginx-Upstream, Timeouts, TLS und Authentifizierung vor jeder Aussage über eine bereitgestellte Anwendung prüfen.

## Wichtigste interne Belegstellen

| Thema | Datei oder Verzeichnis |
|---|---|
| API und Orchestrierung | src/main.py |
| Top-Level-Routing | src/core/insurance_tool_routing.py |
| RAG, Loader, Chunking, Retrieval, Reranking und Prompts | src/api/rag_service.py |
| aktive Modell- und Timeoutkonfiguration | src/config/models.py und anonymisierte Werte aus .env |
| OpenAI-Adapter | src/integrations/openai_answer_model.py |
| CRM-Orchestrierung | src/core/crm_orchestration.py |
| EspoCRM-Client | src/integrations/espocrm_client.py |
| synthetisches CRM-Schema | data/synthetic/crm/schema.json |
| Docker-Topologie | docker/docker-compose.yml und docker/docker-compose.crm.yml |
| Nginx-Konfiguration | docker/nginx/nginx-mva-insurance.conf |
| Groundedness-Schwelle | config/groundedness_calibration.json |
| finaler Helvetia-Index | reports/helvetia_reindex_summary_20260801.md |
| Annotationslücke | reports/groundedness_annotation_gap_report.md |
| Datenqualität und Leakage | reports/groundedness_dataset_quality_report.md und reports/groundedness_dataset_leakage_report.md |
| Human-Review-Protokoll | reports/groundedness_human_review_protocol_20260802.md |
| aktueller Combined-Systemstand | reports/combined_noah_theft_corrected_payload_20260805_112404.md |
| primäre Thesis-Evaluation | scripts/evaluation/evaluate_thesis.py und src/evaluation/thesis_metrics.py |
| historischer vollständiger Lauf | artifacts/test-results/test-result-20260525T235203Z-full-qa200-safety200-qwen25-7b-no-insuranceqa-pdfs |

## Empfohlene Schlussformulierung für Kapitel 4

Der in diesem Kapitel beschriebene Aufbau ermöglicht die getrennte und kombinierte Verarbeitung von Dokumentwissen und strukturierten CRM-Fakten. Durch deterministisches Routing, read-only CRM-Werkzeuge, hybrides Retrieval und nachgelagerte Sicherheitsprüfungen bleibt der Ablauf nachvollziehbar. Gleichzeitig begrenzen die englischsprachige Datenbasis, die fehlende abgeschlossene Human-Annotation, nicht vollständig gesperrte Abhängigkeiten und die noch ausstehende Gesamtevaluation des aktuellen Commits die Aussagekraft des Versuchsstands. Diese Grenzen werden bei der Ergebnisdarstellung in Kapitel 5 berücksichtigt.
