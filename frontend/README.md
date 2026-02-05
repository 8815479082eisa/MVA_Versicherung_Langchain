# Baloise Dokumenten-Assistent - Web UI

Eine einfache, moderne Web-UI für das agentische RAG-System zur Beantwortung von Fragen zu Versicherungsdokumenten.

## 🎯 Übersicht

Diese React + TypeScript Anwendung bietet eine schlichte, benutzerfreundliche Oberfläche für Mitarbeitende einer Versicherung, um Fragen zu Produkten, Bedingungen und Tarifen zu stellen und strukturierte Antworten mit Quellenangaben zu erhalten.

## ✨ Features

- **Frageeingabe**: Einfaches Formular zum Stellen von Fragen mit optionalen Formatierungsoptionen
- **Antwortanzeige**: Strukturierte Darstellung der Antworten mit Quellenangaben
- **Quellenblock**: Detailierte Anzeige der verwendeten Dokumente mit Snippet-Ansicht
- **Fragehistorie**: Übersicht der letzten gestellten Fragen mit Zeitstempel
- **Feedback**: Bewertung der Antworten als hilfreich oder nicht hilfreich
- **Responsives Design**: Funktioniert auf Desktop und Tablet-Geräten

## 🛠️ Technologie-Stack

- **React 18** mit TypeScript
- **Vite** als Build-Tool
- **Tailwind CSS** für Styling
- **Functional Components** mit React Hooks

## 📋 Voraussetzungen

- Node.js 18 oder höher
- npm oder yarn

## 🚀 Installation

1. In das Frontend-Verzeichnis wechseln:
```bash
cd frontend
```

2. Abhängigkeiten installieren:
```bash
npm install
```

3. Entwicklungsserver starten:
```bash
npm run dev
```

Die Anwendung läuft nun auf `http://localhost:3000` und öffnet sich automatisch im Browser.

## 📦 Build für Produktion

```bash
npm run build
```

Die optimierten Dateien werden im `dist` Verzeichnis erstellt.

## 🔧 Backend-Integration

Aktuell verwendet die Anwendung Mock-APIs in `src/api.ts`. Für die Integration mit dem echten Python-Backend:

1. Erstellen Sie einen REST-API-Endpoint im Python-Backend (z.B. mit FastAPI):
   - `POST /api/ask` - Frage stellen und Antwort erhalten
   - `POST /api/feedback` - Feedback zu einer Antwort senden

2. Aktualisieren Sie die Funktionen in `src/api.ts`:
   - Ersetzen Sie die Mock-Implementierungen durch echte `fetch`-Aufrufe
   - Passen Sie die API-URLs an Ihre Backend-Konfiguration an

3. Konfigurieren Sie CORS im Backend, falls Frontend und Backend auf verschiedenen Ports laufen.

## 📁 Projektstruktur

```
frontend/
├── src/
│   ├── components/
│   │   ├── QuestionForm.tsx      # Frageeingabe-Formular
│   │   ├── AnswerView.tsx        # Antwort- und Quellenanzeige
│   │   └── HistoryList.tsx       # Fragehistorie
│   ├── App.tsx                   # Hauptkomponente
│   ├── api.ts                    # Mock-API (später Backend-Integration)
│   ├── main.tsx                  # Einstiegspunkt
│   └── index.css                 # Globale Styles (Tailwind)
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── postcss.config.js
```

## 🎨 Design

Das Design orientiert sich an einer modernen, seriösen Unternehmens-Website mit:
- Klarem, minimalistischem Layout
- Professioneller Farbpalette (Blau-Töne)
- Gut lesbarer Typografie
- Responsivem Grid-Layout

## 📝 Lizenz

Dieses Projekt ist Teil einer Masterarbeit. Bitte beachten Sie die entsprechenden Lizenzbestimmungen.

