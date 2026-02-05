# Agentic RAG System for Insurance Documents

Ein intelligentes Retrieval-Augmented Generation (RAG) System für die Verarbeitung und Beantwortung von Fragen zu Versicherungsdokumenten.

## 🎯 Übersicht

Dieses System implementiert ein agentisches RAG-System, das Versicherungsdokumente (PDFs) indiziert, durchsucht und präzise Antworten mit Quellenangaben generiert. Es kombiniert hybride Retrieval-Verfahren, LLM-basiertes Re-Ranking, Self-Check-Mechanismen und strukturierte Audit-Logs.

## ✨ Hauptfunktionen

- **Hybrid Retrieval**: Kombination aus BM25 (lexikalisch) und semantischer Vektorsuche (ChromaDB)
- **LLM-basiertes Re-Ranking**: GPT-4o-mini zur Relevanzbewertung
- **Self-Check & Query-Rewriting**: Automatische Relevanzprüfung und Query-Optimierung
- **Agentisches Routing**: Intelligente Entscheidung zwischen Retrieval und direkter Antwort
- **Strukturierte Audit-Logs**: Vollständige Nachvollziehbarkeit aller Verarbeitungsschritte
- **Automatische Index-Verwaltung**: Erkennt Änderungen an Quelldokumenten und re-indiziert automatisch

## 🏗️ Architektur

Das System folgt einer dreischichtigen Architektur:

1. **Document Handling & Indexing Layer**: Dokumentenverarbeitung, Chunking, Embedding-Generierung, persistente Speicherung
2. **Retrieval & Reasoning Layer**: Hybrid-Retrieval, Re-Ranking, Self-Check, Query-Rewriting, Antwortgenerierung
3. **Safety & Audit Layer**: Audit-Logging, Safety-Checks, Session-Management

## 🛠️ Technologie-Stack

### Backend
- **Python 3.11**
- **FastAPI**: REST-API-Framework
- **LangChain**: Framework für RAG-Pipeline
- **ChromaDB**: Persistenter Vektorspeicher
- **OpenAI API**:
  - `text-embedding-3-large` für Embeddings
  - `gpt-4o` für Antwortgenerierung
  - `gpt-4o-mini` für Re-Ranking und Context-Compression
  - `gpt-3.5-turbo` für Router, Self-Check und Query-Rewrite

### Frontend
- **React 18** mit TypeScript
- **Vite**: Build-Tool
- **Tailwind CSS**: Styling

## 📋 Voraussetzungen

### Backend
- Python 3.11 oder höher
- OpenAI API Key
- Installierte Abhängigkeiten (siehe `requirements.txt`)

### Frontend
- Node.js 18 oder höher
- npm oder yarn

## 🚀 Installation

1. Repository klonen:
```bash
git clone <repository-url>
cd LangChain
```

2. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

3. Umgebungsvariablen konfigurieren:
Erstellen Sie eine `.env` Datei im Hauptverzeichnis:
```
OPENAI_API_KEY=your_api_key_here
```

4. Frontend-Abhängigkeiten installieren:
```bash
cd frontend
npm install
```

## 💻 Verwendung

### Option 1: Web-UI (Empfohlen)

1. **Backend starten:**
   ```bash
   # Im Hauptverzeichnis
   python backend_api.py
   ```
   Das Backend läuft dann auf `http://localhost:8000`

2. **Frontend starten:**
   ```bash
   # In einem neuen Terminal, im frontend-Verzeichnis
   cd frontend
   npm run dev
   ```
   Das Frontend läuft dann auf `http://localhost:3000` und öffnet sich automatisch im Browser.

3. **Verwendung:**
   - Versicherungsdokumente in den `./docs` Ordner legen (PDF-Format)
   - Im Browser Fragen stellen - die Antworten werden vom echten RAG-System generiert

### Option 2: CLI (Kommandozeile)

1. Versicherungsdokumente in den `./docs` Ordner legen (PDF-Format)

2. System starten:
```bash
python main.py
```

3. Fragen stellen:
Das System lädt automatisch alle PDFs aus `./docs` (außer `example.pdf`), indiziert sie und startet eine interaktive CLI-Session.

4. Beispiel-Fragen:
- "Wie hoch ist die Deckungssumme im Tarif Baloise All-in Gold?"
- "Was sind die Bedingungen für Kaskoversicherung?"
- "Welche Selbstbeteiligung gilt bei Diebstahl?"

5. Beenden:
Geben Sie `exit` ein, um das System zu beenden.

### API-Endpoints

Das FastAPI-Backend stellt folgende Endpoints zur Verfügung:

- `GET /` - Health-Check
- `GET /health` - Health-Check
- `POST /api/ask` - Frage stellen und Antwort erhalten
  - Request Body: `{ "question": "Ihre Frage hier" }`
  - Response: `{ "answer": "...", "sources": [...], "latencyMs": 1250 }`
- `POST /api/feedback` - Feedback zu einer Antwort senden
  - Request Body: `{ "answer_id": "...", "useful": true/false }`

## 📁 Projektstruktur

```
LangChain/
├── main.py                 # Hauptanwendung (CLI)
├── backend_api.py          # FastAPI-Backend für Web-UI
├── requirements.txt        # Python-Abhängigkeiten
├── README.md              # Diese Datei
├── .env                   # Umgebungsvariablen (nicht versioniert)
├── docs/                  # Versicherungsdokumente (PDFs)
├── chroma_db/             # ChromaDB Vektorspeicher
├── audit.log              # Audit-Logs (JSONL-Format)
├── .pdf_hashes.json      # PDF Hash-Tracking für Index-Updates
├── src/                   # Quellcode-Module
│   ├── api/               # API-Layer
│   │   └── rag_service.py # Zentrale RAG-Service-Funktion
│   ├── components/        # RAG-Komponenten
│   ├── core/              # Safety & Audit, Session Memory
│   ├── document_handling/ # Dokumentenverarbeitung
│   └── retrieval_pipeline/ # Retrieval-Logik
└── frontend/              # React-Frontend
    ├── src/
    │   ├── api.ts         # Frontend-API (Backend-Verbindung)
    │   ├── App.tsx        # Hauptkomponente
    │   └── components/    # UI-Komponenten
    ├── package.json
    └── vite.config.ts
```

## ⚙️ Konfiguration

Die wichtigsten Konfigurationsparameter in `main.py`:

- `CHUNK_SIZE = 1000`: Größe der Dokumenten-Chunks (Zeichen)
- `CHUNK_OVERLAP = 200`: Overlap zwischen Chunks (Zeichen)
- `COLLECTION_NAME = "insurance_rag_collection"`: ChromaDB Collection-Name
- `CHROMA_PERSIST_DIRECTORY = "./chroma_db"`: Speicherort des Vektorspeichers
- `AUDIT_LOG_FILE = "./audit.log"`: Pfad zu den Audit-Logs

## 🔍 Funktionsweise

1. **Dokumentenverarbeitung**: PDFs werden geladen, in Chunks segmentiert und mit Metadaten (Quelle, Seite) versehen
2. **Embedding-Generierung**: Jeder Chunk wird in einen hochdimensionalen Vektor eingebettet
3. **Indexierung**: Embeddings werden in ChromaDB gespeichert, BM25-Index wird aufgebaut
4. **Anfrageverarbeitung**:
   - Router entscheidet, ob Retrieval benötigt wird
   - Hybrid-Retriever kombiniert BM25 und Vektorsuche
   - Re-Ranker bewertet Relevanz der Dokumente
   - Self-Check validiert Relevanz
   - Bei Bedarf: Query-Rewriting und erneuter Retrieval-Versuch
   - Antwortgenerierung mit Quellenangaben
5. **Audit-Logging**: Alle Schritte werden protokolliert

## 📊 Audit-Logs

Jeder Verarbeitungsschritt wird in strukturierten JSONL-Logs gespeichert:
- Timestamp (ISO 8601)
- Query
- Retrieved Documents (mit Metadaten)
- Compressed Context
- Generated Answer
- Chat History

## 🔧 Fehlerbehandlung

- **Keine relevanten Dokumente**: Generische Fehlermeldung mit Hinweis auf alternative Formulierung
- **Self-Check fehlgeschlagen**: Automatisches Query-Rewriting und erneuter Retrieval-Versuch
- **Max Retries erreicht**: System gibt Fehlermeldung zurück (verhindert Endlosschleifen)

## 🧪 Testing

Das System wurde mit Versicherungsdokumenten der Baloise getestet. Beispiel-Fragen zu Deckungssummen, Tarifen und Bedingungen werden korrekt beantwortet.

## 📝 Lizenz

Dieses Projekt ist Teil einer Masterarbeit. Bitte beachten Sie die entsprechenden Lizenzbestimmungen.

## 🤝 Beitragen

Dieses Projekt ist Teil einer wissenschaftlichen Arbeit. Für Fragen oder Anregungen öffnen Sie bitte ein Issue.

## 📚 Referenzen

- LangChain Documentation: https://python.langchain.com/
- ChromaDB Documentation: https://www.trychroma.com/
- OpenAI API Documentation: https://platform.openai.com/docs

## 🔒 Datenschutz

- Audit-Logs enthalten möglicherweise sensible Informationen
- Session-Daten werden im Arbeitsspeicher gehalten
- Für Produktions-Deployment sind zusätzliche Sicherheitsmaßnahmen erforderlich (PII-Erkennung, Verschlüsselung, etc.)

## 🧪 Testing

### Backend testen

1. Backend starten:
```bash
python backend_api.py
```

2. API testen (in einem neuen Terminal):
```bash
# Health-Check
curl http://localhost:8000/health

# Frage stellen
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Wie hoch ist die Deckungssumme für Personenschäden?"}'
```

### Frontend testen

1. Backend und Frontend starten (siehe Verwendung)
2. Browser öffnen: `http://localhost:3000`
3. Testfragen stellen

## 🚧 Geplante Erweiterungen

- Docker/Kubernetes Containerisierung
- PostgreSQL für Metadaten und Session-Management
- Open Policy Agent (OPA) für RBAC/ABAC
- NeMo Guardrails für PII-Erkennung und Prompt-Injection-Detection
- Erweiterte Monitoring- und Metriken-Sammlung

## 📧 Kontakt

Für Fragen oder Anregungen öffnen Sie bitte ein Issue im Repository.

