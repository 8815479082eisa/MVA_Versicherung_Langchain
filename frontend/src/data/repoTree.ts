export type RepoNodeKind = "folder" | "file";
export type RepoRelationType = "dependsOn" | "usedBy" | "related";

export interface RepoRelation {
  id: string;
  type: RepoRelationType;
  note?: string;
}

export interface RepoTreeNode {
  id: string;
  name: string;
  path: string;
  kind: RepoNodeKind;
  tags: string[];
  descriptionShort: string;
  descriptionLong: string[];
  entryPoints?: string[];
  runtimeArtifacts?: string[];
  relatedIds?: string[];
  relations?: RepoRelation[];
  children?: RepoTreeNode[];
}

export const repoTreeData: RepoTreeNode = {
  id: "repo-root",
  name: "MVA_Versicherung_Langchain_main",
  path: "MVA_Versicherung_Langchain_main/",
  kind: "folder",
  tags: ["repository", "root"],
  descriptionShort: "Root des Projekts mit Backend, Frontend, Daten, Evaluation und Deployment.",
  descriptionLong: [
    "Enthaelt die FastAPI/RAG-Implementierung unter src.",
    "Enthaelt Daten, Vektorstore, Logs und Evaluations-Outputs unter data.",
    "Enthaelt Frontend, Docker-Setup und technische Dokumentation.",
  ],
  children: [
    {
      id: "src",
      name: "src",
      path: "src/",
      kind: "folder",
      tags: ["backend", "python", "rag"],
      descriptionShort: "Kern des Backends inklusive API, RAG-Pipeline, Settings und Ingestion.",
      descriptionLong: [
        "Hier liegt die produktive FastAPI-App.",
        "Die zentrale RAG-Orchestrierung ist in api/rag_service.py umgesetzt.",
        "Konfiguration und Daten-Ingestion sind als eigene Module getrennt.",
      ],
      children: [
        {
          id: "src-main",
          name: "main.py",
          path: "src/main.py",
          kind: "file",
          tags: ["fastapi", "endpoint", "api"],
          descriptionShort: "FastAPI-Einstiegspunkt mit /api/ask und App-Lifecycle.",
          descriptionLong: [
            "Definiert die FastAPI-Anwendung und CORS.",
            "Stellt den Endpoint POST /api/ask bereit und ruft den RAG-Service auf.",
            "Unterstuetzt optionalen FAQ-Pfad und Fehlerabbildung fuer API-Responses.",
          ],
          entryPoints: ["app = FastAPI(...)", "lifespan(...)", "ask(request) @ POST /api/ask"],
          runtimeArtifacts: ["data/processed/logs/audit.log", "HTTP response payload (answer, sources, latencyMs)"],
          relations: [
            { id: "src-rag-service", type: "dependsOn", note: "Delegiert die Antwortgenerierung an run_rag()." },
            { id: "src-config-models", type: "dependsOn", note: "Nutzt indirekt Settings aus dem RAG-Service." },
            { id: "frontend-api", type: "usedBy", note: "Wird von der UI ueber /api/ask aufgerufen." },
            { id: "eval-insuranceqa", type: "usedBy", note: "Evaluation feuert Requests gegen /api/ask." },
          ],
        },
        {
          id: "src-api",
          name: "api",
          path: "src/api/",
          kind: "folder",
          tags: ["rag", "orchestration"],
          descriptionShort: "API-nahe Services, vor allem die RAG-Orchestrierung.",
          descriptionLong: [
            "Diese Ebene kapselt die komplette Pipeline hinter einer klaren Service-Schnittstelle.",
          ],
          children: [
            {
              id: "src-rag-service",
              name: "rag_service.py",
              path: "src/api/rag_service.py",
              kind: "file",
              tags: ["rag", "retrieval", "generation", "router", "self-check"],
              descriptionShort: "Zentrale RAG-Pipeline: Retrieval, Reranking, Self-Check, Rewrite, Answer-Generation.",
              descriptionLong: [
                "Initialisiert Embeddings, Chroma, BM25, Reranker und LLM-Rollen.",
                "Steuert den Laufzeitfluss mit Router, Retrieval-Loop und optionaler Kontext-Kompression.",
                "Schreibt Audit-Logs und liefert final die Antwort inkl. Quellen zurueck.",
              ],
              entryPoints: ["RAGPipeline.initialize(...)", "RAGPipeline.run(...)", "run_rag(question, chat_history)"],
              runtimeArtifacts: [
                "data/processed/vectorstores/chroma_db/chroma.sqlite3",
                "data/processed/logs/audit.log",
                "data/processed/caches/model_config.json",
              ],
              relations: [
                { id: "src-config-models", type: "dependsOn", note: "Laedt Modell-, Storage- und Retrieval-Settings." },
                { id: "src-insuranceqa-ingestion", type: "dependsOn", note: "Bindet optional InsuranceQA-Retriever ein." },
                { id: "chroma-sqlite", type: "dependsOn", note: "Persistenter Vektorstore fuer Retrieval." },
                { id: "audit-log", type: "dependsOn", note: "Schreibt Laufzeit- und Quellenprotokoll." },
                { id: "src-main", type: "usedBy", note: "Wird vom FastAPI-Endpoint aufgerufen." },
              ],
            },
          ],
        },
        {
          id: "src-config",
          name: "config",
          path: "src/config/",
          kind: "folder",
          tags: ["settings", "config"],
          descriptionShort: "Konfigurationsmodelle und Umgebungsvariablen-Mapping.",
          descriptionLong: [
            "Stellt typisierte Settings fuer Provider, Modelle, Storage und Retrieval bereit.",
          ],
          children: [
            {
              id: "src-config-models",
              name: "models.py",
              path: "src/config/models.py",
              kind: "file",
              tags: ["settings", "paths", "env"],
              descriptionShort: "Zentraler Konfigurations-Layer mit Pfaden fuer data/raw, data/processed und Chroma.",
              descriptionLong: [
                "Definiert Dataclasses fuer Rollen, Retrieval, Generation und Storage.",
                "Laedt Umgebungsvariablen aus .env in ein konsistentes ModelSettings-Objekt.",
                "Bestimmt unter anderem PDF-Quelle, Chroma-Verzeichnis und Audit-Log-Pfad.",
              ],
              entryPoints: ["load_model_settings()", "StorageConfig", "ModelSettings"],
              runtimeArtifacts: [
                "data/raw/pdfs/",
                "data/processed/vectorstores/chroma_db/",
                "data/processed/logs/audit.log",
              ],
              relations: [
                { id: "src-rag-service", type: "usedBy", note: "Wird beim Service-Start geladen." },
                { id: "src-insuranceqa-ingestion", type: "usedBy", note: "Nutzt denselben Storage- und Embedding-Kontext." },
              ],
            },
          ],
        },
        {
          id: "src-data",
          name: "data",
          path: "src/data/",
          kind: "folder",
          tags: ["ingestion", "insuranceqa"],
          descriptionShort: "Datennahe Verarbeitung und Ingestion-Helfer.",
          descriptionLong: [
            "Enthaelt die InsuranceQA-Ingestion fuer Aufbau und Nutzung einer separaten Collection.",
          ],
          children: [
            {
              id: "src-insuranceqa-ingestion",
              name: "insuranceqa_ingestion.py",
              path: "src/data/insuranceqa_ingestion.py",
              kind: "file",
              tags: ["ingestion", "insuranceqa", "chroma", "cli"],
              descriptionShort: "Baut InsuranceQA-Index, erstellt Chunks und liefert einen Retriever.",
              descriptionLong: [
                "Liest lokale JSONL oder HuggingFace-Dataset, wandelt Q/A in Documents um und chunked sie.",
                "Schreibt in die Collection insuranceqa_collection im Chroma-Persistenzpfad.",
                "Stellt einen Retriever fuer die RAG-Pipeline bereit und bietet CLI-Entry fuer Reindexing.",
              ],
              entryPoints: ["build_insuranceqa_index(...)", "get_insuranceqa_retriever()", "main(argv)"],
              runtimeArtifacts: [
                "data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl",
                "data/processed/vectorstores/chroma_db/chroma.sqlite3",
              ],
              relations: [
                { id: "insuranceqa-jsonl", type: "dependsOn", note: "Default-Quelle fuer lokale InsuranceQA-Daten." },
                { id: "chroma-sqlite", type: "dependsOn", note: "Persistiert Chunks und Embeddings." },
                { id: "src-rag-service", type: "usedBy", note: "Wird bei aktivem USE_INSURANCEQA_DATA eingebunden." },
              ],
            },
          ],
        },
      ],
    },
    {
      id: "data",
      name: "data",
      path: "data/",
      kind: "folder",
      tags: ["storage", "artifacts", "dataset"],
      descriptionShort: "Eingangsdaten, persistierte Artefakte, Logs und Evaluations-Ergebnisse.",
      descriptionLong: [
        "Trennt Rohdaten (raw) von erzeugten Artefakten (processed).",
        "Enthaelt sowohl Laufzeitdaten als auch Benchmark-/Evaluationsdaten.",
      ],
      children: [
        {
          id: "benchmarks",
          name: "benchmarks",
          path: "data/benchmarks/",
          kind: "folder",
          tags: ["dataset", "qa"],
          descriptionShort: "Benchmark-Sets fuer QA- und Evaluationslaeufe.",
          descriptionLong: ["Enthaelt die InsuranceQA-Stichprobe fuer lokale und reproduzierbare Tests."],
          children: [
            {
              id: "insuranceqa-jsonl",
              name: "data_insuranceqa_1000.jsonl",
              path: "data/benchmarks/qa/insuranceqa/data_insuranceqa_1000.jsonl",
              kind: "file",
              tags: ["insuranceqa", "dataset", "jsonl"],
              descriptionShort: "Lokales InsuranceQA-QA-Set (1000 Eintraege) fuer FAQ/ingestion/eval.",
              descriptionLong: [
                "Wird fuer lokale Retrieval-Tests und optional als FAQ-Quelle verwendet.",
                "Dient auch als Ingestion-Quelle fuer insuranceqa_collection.",
              ],
              relations: [
                { id: "src-insuranceqa-ingestion", type: "usedBy", note: "Wird beim Indexaufbau eingelesen." },
                { id: "src-main", type: "usedBy", note: "Kann als FAQ_FILE fuer direkten Match dienen." },
              ],
            },
          ],
        },
        {
          id: "processed",
          name: "processed",
          path: "data/processed/",
          kind: "folder",
          tags: ["runtime", "outputs", "logs"],
          descriptionShort: "Zur Laufzeit erzeugte Artefakte und Evaluations-Outputs.",
          descriptionLong: [
            "Enthaelt Chroma-DB, Cache-Dateien, Audit-Logs und Metrik-Outputs.",
          ],
          children: [
            {
              id: "vectorstores",
              name: "vectorstores",
              path: "data/processed/vectorstores/",
              kind: "folder",
              tags: ["chroma", "embeddings"],
              descriptionShort: "Persistente Vektorstores fuer Retrieval.",
              descriptionLong: ["Hier liegt die ChromaDB mit Collection-Metadaten und Segmentdateien."],
              children: [
                {
                  id: "chroma-db",
                  name: "chroma_db",
                  path: "data/processed/vectorstores/chroma_db/",
                  kind: "folder",
                  tags: ["chroma", "sqlite", "segments"],
                  descriptionShort: "Persistenzverzeichnis fuer Chroma Collections.",
                  descriptionLong: [
                    "Enthaelt chroma.sqlite3 und HNSW-Segmente als Unterordner.",
                  ],
                  children: [
                    {
                      id: "chroma-sqlite",
                      name: "chroma.sqlite3",
                      path: "data/processed/vectorstores/chroma_db/chroma.sqlite3",
                      kind: "file",
                      tags: ["sqlite", "chroma", "vectorstore"],
                      descriptionShort: "SQLite-Metadatenbank fuer Collections, Embeddings und Segmente.",
                      descriptionLong: [
                        "Haelt Collection-Namen und Segment-Mapping (z. B. insuranceqa_collection).",
                        "Der eigentliche Vektorindex liegt in zugehoerigen Segment-Ordnern.",
                      ],
                      relations: [
                        { id: "src-rag-service", type: "usedBy", note: "Lesend fuer Runtime-Retrieval." },
                        { id: "src-insuranceqa-ingestion", type: "usedBy", note: "Schreibt/aktualisiert Collection-Inhalte." },
                      ],
                    },
                  ],
                },
              ],
            },
            {
              id: "logs",
              name: "logs",
              path: "data/processed/logs/",
              kind: "folder",
              tags: ["audit", "runtime"],
              descriptionShort: "Backend-Laufzeit- und Audit-Logs.",
              descriptionLong: [
                "Enthaelt JSONL-Auditdaten und Uvicorn-/Boot-Logs.",
              ],
              children: [
                {
                  id: "audit-log",
                  name: "audit.log",
                  path: "data/processed/logs/audit.log",
                  kind: "file",
                  tags: ["audit", "jsonl", "traceability"],
                  descriptionShort: "Strukturiertes JSONL-Protokoll fuer Queries, Retrieval-Kontext und Antworten.",
                  descriptionLong: [
                    "Unterstuetzt Nachvollziehbarkeit und spaetere ALCE-/Audit-Auswertung.",
                  ],
                  relations: [
                    { id: "src-rag-service", type: "usedBy", note: "Wird bei jeder Anfrage beschrieben." },
                    { id: "eval-alce", type: "usedBy", note: "Audit-Modus liest diese Datei zur Auswertung." },
                  ],
                },
              ],
            },
          ],
        },
      ],
    },
    {
      id: "scripts",
      name: "scripts",
      path: "scripts/",
      kind: "folder",
      tags: ["evaluation", "tools"],
      descriptionShort: "Hilfsskripte fuer Evaluation, CLI und Datenaufbereitung.",
      descriptionLong: [
        "Insbesondere unter scripts/evaluation liegen reproduzierbare Auswertungsskripte.",
      ],
      children: [
        {
          id: "scripts-evaluation",
          name: "evaluation",
          path: "scripts/evaluation/",
          kind: "folder",
          tags: ["evaluation", "metrics"],
          descriptionShort: "Metrik- und Auswertungsskripte.",
          descriptionLong: ["Bewertet Antwortqualitaet fuer InsuranceQA und ALCE-nahe Kriterien."],
          children: [
            {
              id: "eval-insuranceqa",
              name: "eval_insuranceqa.py",
              path: "scripts/evaluation/eval_insuranceqa.py",
              kind: "file",
              tags: ["evaluation", "insuranceqa", "em", "f1"],
              descriptionShort: "Fuehrt InsuranceQA-Evaluation gegen /api/ask aus (EM + Token-F1).",
              descriptionLong: [
                "Liest Fragen aus HF-Dataset und sendet HTTP-Requests an das Backend.",
                "Speichert pro Item Prediction, References, Exact Match und best F1 als JSONL.",
              ],
              entryPoints: ["main()", "post_with_retry(...)", "token_f1(...)"],
              runtimeArtifacts: ["data/processed/eval_outputs/insuranceqa/*.jsonl"],
              relations: [
                { id: "src-main", type: "dependsOn", note: "Erwartet lauffaehiges Backend /api/ask." },
                { id: "eval-alce", type: "usedBy", note: "Dessen Outputs koennen in ALCE-Summary einfliessen." },
              ],
            },
            {
              id: "eval-alce",
              name: "evaluate_alce.py",
              path: "scripts/evaluation/evaluate_alce.py",
              kind: "file",
              tags: ["evaluation", "alce", "summary"],
              descriptionShort: "Aggregiert ALCE-nahe Kennzahlen fuer Audit- und InsuranceQA-Modus.",
              descriptionLong: [
                "Liest entweder audit.log oder InsuranceQA-Result-JSONL als Input.",
                "Erzeugt Item-Level-Dateien und Summary-JSON (inkl. exact_match_mean/token_f1_mean).",
              ],
              entryPoints: ["main()", "run_audit_mode(...)", "run_insuranceqa_mode(...)"],
              runtimeArtifacts: ["data/processed/eval_outputs/alce/*.json", "data/processed/eval_outputs/alce/*.jsonl"],
              relations: [
                { id: "audit-log", type: "dependsOn", note: "Audit-Modus basiert auf Laufzeitlogs." },
                { id: "eval-insuranceqa", type: "dependsOn", note: "InsuranceQA-Modus nutzt dessen Ergebnisdateien." },
              ],
            },
          ],
        },
      ],
    },
    {
      id: "docker",
      name: "docker",
      path: "docker/",
      kind: "folder",
      tags: ["deployment", "containers"],
      descriptionShort: "Container- und Reverse-Proxy-Konfiguration.",
      descriptionLong: [
        "Definiert lokalen/prod-nahen Betrieb fuer Backend, Frontend und Nginx.",
      ],
      children: [
        {
          id: "docker-compose",
          name: "docker-compose.yml",
          path: "docker/docker-compose.yml",
          kind: "file",
          tags: ["docker", "compose", "ops"],
          descriptionShort: "Orchestriert Backend, Frontend und Nginx als Services.",
          descriptionLong: [
            "Mountet Datenpfade fuer PDFs und Chroma-Persistenz.",
            "Startet Backend via uvicorn und expose't Ports fuer API/UI.",
          ],
          entryPoints: ["services.backend", "services.frontend", "services.nginx"],
          relations: [
            { id: "dockerfile", type: "related", note: "Build-/Runtime-Basis fuer Backend." },
            { id: "nginx-conf", type: "dependsOn", note: "Bindet Nginx-Config als Volume ein." },
            { id: "src-main", type: "dependsOn", note: "Startet uvicorn src.main:app." },
          ],
        },
        {
          id: "dockerfile",
          name: "Dockerfile",
          path: "docker/Dockerfile",
          kind: "file",
          tags: ["docker", "backend"],
          descriptionShort: "Mehrstufiges Backend-Image mit Python-Abhaengigkeiten und Healthcheck.",
          descriptionLong: [
            "Installiert Requirements und startet uvicorn fuer src.main:app.",
          ],
          relations: [{ id: "src-main", type: "dependsOn" }],
        },
        {
          id: "nginx-folder",
          name: "nginx",
          path: "docker/nginx/",
          kind: "folder",
          tags: ["nginx", "proxy"],
          descriptionShort: "Nginx-Konfiguration als Reverse Proxy fuer API und Frontend.",
          descriptionLong: ["Leitet API-Routen an Backend weiter und liefert Frontend-Build aus."],
          children: [
            {
              id: "nginx-conf",
              name: "nginx-mva-insurance.conf",
              path: "docker/nginx/nginx-mva-insurance.conf",
              kind: "file",
              tags: ["nginx", "reverse-proxy"],
              descriptionShort: "Server- und Routing-Regeln fuer produktionsnahe Auslieferung.",
              descriptionLong: [
                "Definiert Upstream/Proxy-Verhalten fuer /api und statische Frontend-Dateien.",
              ],
              relations: [{ id: "src-main", type: "dependsOn" }, { id: "frontend-app", type: "dependsOn" }],
            },
          ],
        },
      ],
    },
    {
      id: "docs",
      name: "docs",
      path: "docs/",
      kind: "folder",
      tags: ["documentation", "operations", "thesis"],
      descriptionShort: "Betriebsdokumentation und fachliche Abschnitte zur Evaluation.",
      descriptionLong: [
        "Enthaelt Quick-Start/SERVER_SETUP sowie Entwicklungsabschnitte fuer die Arbeit.",
      ],
      children: [
        {
          id: "docs-manuals",
          name: "manuals",
          path: "docs/manuals/",
          kind: "folder",
          tags: ["manual", "ops"],
          descriptionShort: "Handbuecher fuer Setup, Betrieb und Deploy.",
          descriptionLong: ["Empfohlener Einstieg fuer neue Entwickler und Server-Admins."],
          children: [
            {
              id: "docs-quick-start",
              name: "QUICK_START.md",
              path: "docs/manuals/QUICK_START.md",
              kind: "file",
              tags: ["quickstart", "ops"],
              descriptionShort: "Schneller Ablauf fuer lokale und produktionsnahe Inbetriebnahme.",
              descriptionLong: [
                "Beschreibt manuelle, Docker- und Systemd-Varianten inkl. Basischecks.",
              ],
              relations: [
                { id: "docker-compose", type: "related" },
                { id: "src-main", type: "related" },
              ],
            },
            {
              id: "docs-server-setup",
              name: "SERVER_SETUP.md",
              path: "docs/manuals/SERVER_SETUP.md",
              kind: "file",
              tags: ["server", "deployment", "operations"],
              descriptionShort: "Detailliertes Setup fuer Serverbetrieb (venv, uvicorn, systemd, nginx).",
              descriptionLong: [
                "Definiert Deployment-Schritte, Konfiguration und Stabilitaets-Hinweise.",
              ],
              relations: [
                { id: "docker-compose", type: "related" },
                { id: "nginx-conf", type: "related" },
              ],
            },
          ],
        },
      ],
    },
    {
      id: "frontend",
      name: "frontend",
      path: "frontend/",
      kind: "folder",
      tags: ["react", "typescript", "ui"],
      descriptionShort: "Vite/React-Frontend fuer Fragen, Antworten und Quellenanzeige.",
      descriptionLong: [
        "UI-Schicht fuer Nutzerinteraktion mit dem Backend.",
        "Kommuniziert via fetch mit der FastAPI-API.",
      ],
      children: [
        {
          id: "frontend-src",
          name: "src",
          path: "frontend/src/",
          kind: "folder",
          tags: ["ui", "components"],
          descriptionShort: "React-Komponenten und API-Client.",
          descriptionLong: ["Enthaelt zentrale App-Komponente, API-Layer und Antwortdarstellung."],
          children: [
            {
              id: "frontend-app",
              name: "App.tsx",
              path: "frontend/src/App.tsx",
              kind: "file",
              tags: ["react", "layout", "state"],
              descriptionShort: "Haupt-UI mit Seitenlayout, State-Handling und View-Umschaltung.",
              descriptionLong: [
                "Orchestriert Frageformular, Antwortbereich, Historie und Repo-Map-Ansicht.",
              ],
              entryPoints: ["const App: React.FC = () => { ... }"],
              relations: [
                { id: "frontend-api", type: "dependsOn" },
                { id: "frontend-answer-view", type: "dependsOn" },
                { id: "repo-explorer-component", type: "dependsOn" },
              ],
            },
            {
              id: "frontend-api",
              name: "api.ts",
              path: "frontend/src/api.ts",
              kind: "file",
              tags: ["fetch", "client", "api"],
              descriptionShort: "Frontend-API-Client fuer /api/ask und /api/feedback.",
              descriptionLong: [
                "Kapselt Netzwerkaufrufe und konvertiert Backend-Antworten ins UI-kompatible Format.",
              ],
              entryPoints: ["askQuestion(payload)", "sendFeedback(answerId, useful)"],
              relations: [
                { id: "src-main", type: "dependsOn", note: "Kommuniziert mit FastAPI-Endpoint /api/ask." },
                { id: "frontend-app", type: "usedBy" },
              ],
            },
            {
              id: "frontend-components",
              name: "components",
              path: "frontend/src/components/",
              kind: "folder",
              tags: ["react", "components"],
              descriptionShort: "UI-Bausteine fuer Frage, Antwort und Historie.",
              descriptionLong: ["Modulare Komponenten fuer klare Trennung von Darstellung und Datenfluss."],
              children: [
                {
                  id: "frontend-answer-view",
                  name: "AnswerView.tsx",
                  path: "frontend/src/components/AnswerView.tsx",
                  kind: "file",
                  tags: ["ui", "answer", "sources", "feedback"],
                  descriptionShort: "Zeigt Antworttext, Quellen, Copy-Funktionen und Feedback-Buttons.",
                  descriptionLong: [
                    "Visualisiert Quellen inkl. expandierbarer Snippets und sendet Nutzerfeedback.",
                  ],
                  relations: [
                    { id: "frontend-app", type: "usedBy" },
                    { id: "frontend-api", type: "dependsOn", note: "Nutzt sendFeedback()." },
                  ],
                },
                {
                  id: "repo-explorer-component",
                  name: "RepoExplorer.tsx",
                  path: "frontend/src/components/RepoExplorer.tsx",
                  kind: "file",
                  tags: ["ui", "documentation", "repo-map"],
                  descriptionShort: "Interaktive Tree+Details-Ansicht fuer die Repository-Architektur.",
                  descriptionLong: [
                    "Zeigt Dateistruktur, Verantwortungen und Abhaengigkeiten als klickbare Wissenskarte.",
                  ],
                  relations: [{ id: "repo-tree-data", type: "dependsOn" }, { id: "frontend-app", type: "usedBy" }],
                },
              ],
            },
            {
              id: "repo-tree-data",
              name: "repoTree.ts",
              path: "frontend/src/data/repoTree.ts",
              kind: "file",
              tags: ["metadata", "architecture", "tree"],
              descriptionShort: "Statische, typisierte Metadaten fuer den Repository-Baum.",
              descriptionLong: [
                "Definiert Knoten, Beschreibungen, Entry-Points und Beziehungen fuer die Repo-Map-UI.",
              ],
              relations: [{ id: "repo-explorer-component", type: "usedBy" }],
            },
          ],
        },
      ],
    },
  ],
};

