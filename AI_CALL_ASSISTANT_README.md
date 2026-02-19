# 🤖 AI Call Assistant - Vollständige Integration mit Vapi.ai

Herzlich Willkommen! Dieses Dokument führt Sie durch die vollständige Nutzung des AI Call Assistant Features in Ihrem MVA Versicherungs-System.

## 📋 Inhaltsverzeichnis

- [Was wurde implementiert?](#was-wurde-implementiert)
- [Quick Start](#quick-start)
- [Backend-Architektur](#backend-architektur)
- [Frontend-Features](#frontend-features)
- [Vapi.ai Setup](#vapiai-setup)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Production Deployment](#production-deployment)

---

## 🎉 Was wurde implementiert?

### ✅ Backend (Python/FastAPI)

1. **Vapi Service Klasse** (`src/api/vapi_service.py`)
   - Vollständige Integration mit Vapi.ai REST API
   - Telefonnummer-Validierung (E.164 Format)
   - Anruf-Initiierung
   - Status-Abfrage
   - Anruf-Beendigung
   - Webhook-Handler für RAG-Integration

2. **Neue API Endpoints** (`src/main.py`)
   - `POST /api/call/start` - Anruf starten
   - `GET /api/call/status/{call_id}` - Status abrufen
   - `POST /api/call/end` - Anruf beenden
   - `POST /api/call/vapi-webhook` - Webhook für Vapi Events

3. **Pydantic Models**
   - `InitiateCallRequest` - Anfrage-Validierung
   - `CallStatusResponse` - Status-Response
   - `EndCallRequest` - Beenden-Anfrage

### ✅ Frontend (React/TypeScript)

1. **CallAssistant Komponente** (`frontend/src/components/CallAssistant.tsx`)
   - Telefonnummer-Eingabe mit Validierung
   - Optionaler Kundenname
   - Anruf-Status in Echtzeit
   - Status-Badge mit Farbcodierung
   - Polling für automatische Updates
   - Anruf beenden Funktion
   - Responsive Design mit Tailwind CSS

2. **API-Funktionen** (`frontend/src/api.ts`)
   - `initiateCall()` - Anruf starten
   - `getCallStatus()` - Status abrufen
   - `endCall()` - Anruf beenden

3. **UI/UX Verbesserungen** (`frontend/src/App.tsx`)
   - Tab-Navigation (Dokumenten-Assistent / AI Call Assistant)
   - Einheitliches Design
   - Fehlerbehandlung
   - Ladeanimationen

### ✅ Dokumentation

- `VAPI_SETUP.md` - Ausführliche Setup-Anleitung für Vapi.ai
- `AI_CALL_ASSISTANT_README.md` - Dieses Dokument
- Code-Kommentare und Dokumentation

---

## 🚀 Quick Start

### Schritt 1: Dependencies installieren

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

### Schritt 2: Vapi.ai konfigurieren

Folgen Sie der detaillierten Anleitung in **[VAPI_SETUP.md](./VAPI_SETUP.md)** um:
- Vapi.ai Account zu erstellen
- Telefonnummer zu kaufen
- Assistant zu konfigurieren
- API Keys zu erhalten

### Schritt 3: Umgebungsvariablen setzen

Erstellen Sie eine `.env` Datei im Projektroot:

```bash
# Bestehende OpenAI Config
OPENAI_API_KEY=sk-your-openai-key

# Vapi.ai Configuration
VAPI_API_KEY=sk_live_your_vapi_api_key
VAPI_ASSISTANT_ID=asst_your_assistant_id
VAPI_PHONE_NUMBER_ID=ph_your_phone_number_id

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

### Schritt 4: Backend starten

```bash
# Option 1: Direkt mit Python
python src/main.py

# Option 2: Mit uvicorn
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Sie sollten diese Meldung sehen:
```
INFO:     Started server process
INFO:     Waiting for application startup.
Backend started (lazy pipeline init).
Backend running at http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Schritt 5: Frontend starten

```bash
cd frontend
npm run dev
```

Das Frontend läuft auf: **http://localhost:5173**

### Schritt 6: Testen!

1. Öffnen Sie **http://localhost:5173**
2. Klicken Sie auf den Tab **"AI Call Assistant"**
3. Geben Sie eine Telefonnummer ein (z.B. `+491234567890`)
4. Optional: Geben Sie einen Kundennamen ein
5. Klicken Sie auf **"AI-Anruf starten"**
6. Beobachten Sie den Status in Echtzeit!

---

## 🏗️ Backend-Architektur

### Dateistruktur

```
src/
├── api/
│   ├── rag_service.py          # Bestehender RAG Service
│   └── vapi_service.py         # ✨ NEU: Vapi Integration
├── components/
│   └── ...                      # RAG Komponenten
├── core/
│   └── ...                      # Core Funktionalität
└── main.py                      # ✨ ERWEITERT: Neue Endpoints
```

### VapiService Klasse

```python
class VapiService:
    """Service-Klasse für Vapi.ai Integration"""
    
    def __init__(self):
        """Lädt Konfiguration aus Umgebungsvariablen"""
        self.api_key = os.getenv("VAPI_API_KEY")
        self.assistant_id = os.getenv("VAPI_ASSISTANT_ID")
        # ...
    
    async def initiate_call(self, phone_number: str, ...):
        """Startet einen ausgehenden Anruf"""
        # Validierung
        # API Call zu Vapi
        # Response verarbeiten
        
    async def get_call_status(self, call_id: str):
        """Ruft aktuellen Call-Status ab"""
        
    async def end_call(self, call_id: str):
        """Beendet einen laufenden Anruf"""
```

### API Endpoints

#### 1. Anruf starten

**Request:**
```bash
POST /api/call/start
Content-Type: application/json

{
  "phoneNumber": "+491234567890",
  "customerName": "Max Mustermann"  // Optional
}
```

**Response:**
```json
{
  "callId": "call_abc123xyz",
  "status": "initiated",
  "phoneNumber": "+491234567890",
  "createdAt": "2024-01-15T10:30:00Z"
}
```

**Mögliche Status:**
- `initiated` - Anruf wurde gestartet
- `queued` - In Warteschlange
- `ringing` - Telefon klingelt
- `in-progress` - Anruf läuft
- `ended` - Anruf beendet
- `failed` - Fehler aufgetreten

#### 2. Status abrufen

**Request:**
```bash
GET /api/call/status/{call_id}
```

**Response:**
```json
{
  "callId": "call_abc123xyz",
  "status": "in-progress",
  "phoneNumber": "+491234567890",
  "createdAt": "2024-01-15T10:30:00Z",
  "duration": 45.5,
  "cost": 0.0625
}
```

#### 3. Anruf beenden

**Request:**
```bash
POST /api/call/end
Content-Type: application/json

{
  "callId": "call_abc123xyz"
}
```

**Response:**
```json
{
  "success": true,
  "callId": "call_abc123xyz",
  "status": "ended",
  "duration": 120.3,
  "cost": 0.1502
}
```

#### 4. Webhook (für Vapi)

**Request von Vapi:**
```bash
POST /api/call/vapi-webhook
Content-Type: application/json

{
  "type": "function-call",
  "functionCall": {
    "name": "search_insurance_info",
    "parameters": {
      "question": "Was deckt die Hausratversicherung ab?"
    }
  }
}
```

**Response an Vapi:**
```json
{
  "result": "Die Hausratversicherung deckt Schäden durch..."
}
```

---

## 🎨 Frontend-Features

### CallAssistant Komponente

Die Hauptkomponente bietet:

1. **Telefonnummer-Eingabe**
   - Validierung während der Eingabe
   - E.164 Format-Prüfung
   - Hilfetext

2. **Status-Anzeige**
   - Farbcodierte Badges
   - Animierte Pulse-Effekte
   - Call ID Anzeige
   - Dauer und Kosten

3. **Real-time Updates**
   - Automatisches Polling alle 3 Sekunden
   - Stoppt bei Anruf-Ende
   - Fehlerbehandlung

4. **Actions**
   - "AI-Anruf starten" Button
   - "Anruf beenden" Button (während aktiv)
   - "Neuer Anruf" Button (nach Ende)

### Verwendete Technologien

- **React** 18.2 - UI Framework
- **TypeScript** - Type Safety
- **Tailwind CSS** - Styling
- **Vite** - Build Tool

### Status-Farben

| Status | Farbe | Bedeutung |
|--------|-------|-----------|
| `initiated` | 🔵 Blau | Anruf wird initiiert |
| `queued` | 🔵 Blau | In Warteschlange |
| `ringing` | 🟡 Gelb | Telefon klingelt |
| `in-progress` | 🟢 Grün | Gespräch läuft |
| `ended` | ⚪ Grau | Beendet |
| `failed` | 🔴 Rot | Fehler |

---

## 🔧 Vapi.ai Setup

Bitte folgen Sie der **ausführlichen Anleitung** in:

### 📄 [VAPI_SETUP.md](./VAPI_SETUP.md)

Diese Anleitung enthält:
- ✅ Account-Erstellung
- ✅ Telefonnummer kaufen
- ✅ Assistant konfigurieren (mit deutschen Prompts)
- ✅ Voice Settings (deutsche Stimmen)
- ✅ Function Configuration (RAG-Integration)
- ✅ Webhook Setup
- ✅ Testing-Strategien

---

## 🧪 Testing

### 1. Backend API Testing mit cURL

**Anruf starten:**
```bash
curl -X POST http://localhost:8000/api/call/start \
  -H "Content-Type: application/json" \
  -d '{"phoneNumber": "+491234567890", "customerName": "Test User"}'
```

**Status abrufen:**
```bash
curl http://localhost:8000/api/call/status/call_abc123xyz
```

**Anruf beenden:**
```bash
curl -X POST http://localhost:8000/api/call/end \
  -H "Content-Type: application/json" \
  -d '{"callId": "call_abc123xyz"}'
```

### 2. Backend API Testing mit Postman

1. Importieren Sie diese Collection:

```json
{
  "info": {
    "name": "MVA Call Assistant API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Start Call",
      "request": {
        "method": "POST",
        "header": [{"key": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"phoneNumber\": \"+491234567890\",\n  \"customerName\": \"Test User\"\n}"
        },
        "url": {
          "raw": "http://localhost:8000/api/call/start",
          "protocol": "http",
          "host": ["localhost"],
          "port": "8000",
          "path": ["api", "call", "start"]
        }
      }
    }
  ]
}
```

### 3. Frontend Testing

1. Starten Sie das Frontend: `npm run dev`
2. Öffnen Sie: http://localhost:5173
3. Wechseln Sie zum Tab "AI Call Assistant"
4. Testen Sie verschiedene Szenarien:

**Test-Szenarien:**

| Szenario | Eingabe | Erwartetes Ergebnis |
|----------|---------|---------------------|
| Gültige Nummer | `+491234567890` | Anruf wird initiiert |
| Ungültige Nummer | `0123456789` | Fehlermeldung: "E.164 Format" |
| Leere Nummer | ` ` | Fehlermeldung: "Bitte eingeben" |
| Ohne + | `491234567890` | Fehlermeldung: "E.164 Format" |
| Mit Leerzeichen | `+49 123 456` | Wird automatisch bereinigt |

### 4. End-to-End Testing

**Vollständiger Test-Flow:**

1. ✅ Backend startet erfolgreich
2. ✅ Frontend lädt ohne Fehler
3. ✅ Tab-Wechsel funktioniert
4. ✅ Telefonnummer-Validierung funktioniert
5. ✅ Anruf kann gestartet werden
6. ✅ Status wird angezeigt und aktualisiert
7. ✅ Anruf kann beendet werden
8. ✅ "Neuer Anruf" setzt Form zurück

---

## 🐛 Troubleshooting

### Problem: "VAPI_API_KEY ist nicht konfiguriert"

**Lösung:**
```bash
# Überprüfen Sie .env Datei
cat .env | grep VAPI

# Stellen Sie sicher, dass .env im Projektroot liegt
ls -la .env

# Backend neu starten
python src/main.py
```

### Problem: "Ungültige Telefonnummer"

**Ursache:** Telefonnummer ist nicht im E.164 Format

**Lösung:**
```bash
# ✅ Richtig:
+491234567890

# ❌ Falsch:
0123 456 7890
(0123) 456-7890
123456789
```

### Problem: "Backend nicht erreichbar"

**Diagnose:**
```bash
# 1. Backend läuft?
curl http://localhost:8000/health

# 2. Port richtig?
netstat -an | grep 8000

# 3. CORS erlaubt Frontend?
# Überprüfen Sie CORS_ALLOW_ORIGINS in .env
```

**Lösung:**
```bash
# Backend neu starten
python src/main.py

# Oder mit uvicorn
uvicorn src.main:app --reload
```

### Problem: "Function Call funktioniert nicht"

**Ursache:** Webhook ist nicht korrekt konfiguriert

**Lösung:**

1. **Für lokales Testing: Ngrok verwenden**
```bash
# Ngrok installieren und starten
ngrok http 8000

# URL kopieren (z.B. https://abc123.ngrok.io)
# In Vapi Assistant Einstellungen eintragen:
# Webhook URL: https://abc123.ngrok.io/api/call/vapi-webhook
```

2. **Webhook Events aktivieren:**
   - `call-started` ✅
   - `call-ended` ✅
   - `function-call` ✅

3. **Function korrekt konfiguriert:**
   - Name: `search_insurance_info`
   - Parameters: `{"question": "string"}`
   - Server URL gesetzt

### Problem: "Status wird nicht aktualisiert"

**Ursache:** Polling funktioniert nicht

**Diagnose:**
```javascript
// Browser Console öffnen (F12)
// Schauen Sie nach Fehler-Meldungen
```

**Lösung:**
- Überprüfen Sie Browser Console auf Fehler
- Stellen Sie sicher, dass Backend erreichbar ist
- Call ID ist korrekt

### Problem: "Anruf kommt nicht an"

**Mögliche Ursachen:**

1. **Keine Credits auf Vapi Account**
   - Lösung: Credits aufladen in Vapi Dashboard

2. **Telefonnummer ist falsch**
   - Lösung: E.164 Format verwenden

3. **Phone Number ID fehlt**
   - Lösung: In .env setzen: `VAPI_PHONE_NUMBER_ID=...`

4. **Nummer ist blockiert**
   - Lösung: Test mit eigener Nummer

---

## 🚀 Production Deployment

### 1. Backend Deployment

#### Option A: Docker

```dockerfile
# Dockerfile (bereits vorhanden, anpassen)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build und Run
docker build -t mva-backend .
docker run -p 8000:8000 --env-file .env mva-backend
```

#### Option B: Systemd Service

```bash
# /etc/systemd/system/mva-backend.service
[Unit]
Description=MVA Insurance Backend
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/mva
Environment="PATH=/var/www/mva/venv/bin"
EnvironmentFile=/var/www/mva/.env
ExecStart=/var/www/mva/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000

[Install]
WantedBy=multi-user.target
```

```bash
# Aktivieren
sudo systemctl enable mva-backend
sudo systemctl start mva-backend
sudo systemctl status mva-backend
```

### 2. Frontend Deployment

```bash
# Build
cd frontend
npm run build

# Output in dist/ Ordner
# Deployen Sie dist/ zu:
# - Vercel
# - Netlify
# - AWS S3 + CloudFront
# - Nginx (statische Dateien)
```

**Nginx Konfiguration:**

```nginx
server {
    listen 80;
    server_name ihre-domain.com;

    # Frontend
    location / {
        root /var/www/mva/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. Umgebungsvariablen (Production)

```bash
# .env.production
OPENAI_API_KEY=sk-prod-...
VAPI_API_KEY=sk_live_...
VAPI_ASSISTANT_ID=asst_prod_...
VAPI_PHONE_NUMBER_ID=ph_prod_...

BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
CORS_ALLOW_ORIGINS=https://ihre-domain.com

# Sicherheit
INIT_PIPELINE_ON_STARTUP=true
```

### 4. Vapi Webhook (Production)

**In Vapi Assistant Einstellungen:**

```
Webhook URL: https://ihre-domain.com/api/call/vapi-webhook
```

**Wichtig:** HTTPS ist erforderlich!

### 5. SSL/TLS Zertifikat

```bash
# Let's Encrypt mit Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d ihre-domain.com
```

### 6. Monitoring & Logging

**Logs ansehen:**
```bash
# Systemd Service
sudo journalctl -u mva-backend -f

# Docker
docker logs -f mva-backend

# Nginx
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

**Monitoring Tools:**
- **Uptime:** UptimeRobot, Pingdom
- **Errors:** Sentry
- **Performance:** New Relic, DataDog
- **Logs:** CloudWatch, Papertrail

---

## 📊 Kosten-Übersicht

### Vapi.ai Kosten (Stand 2024)

| Service | Kosten |
|---------|--------|
| Deutsche Telefonnummer | ~$2-5/Monat |
| Ausgehende Anrufe | ~$0.05-0.15/Minute |
| ElevenLabs Stimmen | In Minute-Rate enthalten |
| Deepgram Transcription | In Minute-Rate enthalten |

### OpenAI Kosten (für RAG + Assistant)

| Modell | Input | Output |
|--------|-------|--------|
| GPT-4-turbo | $0.01/1K tokens | $0.03/1K tokens |
| GPT-3.5-turbo | $0.0005/1K tokens | $0.0015/1K tokens |
| text-embedding-ada-002 | $0.0001/1K tokens | - |

**Beispiel-Rechnung (10 Anrufe/Tag):**

```
10 Anrufe × 2 Minuten × $0.10/Min = $2.00/Tag
30 Tage × $2.00 = $60/Monat für Anrufe

+ Telefonnummer: $5/Monat
+ OpenAI GPT-4: ~$20/Monat (geschätzt)
= ~$85/Monat Gesamt
```

---

## 🎯 Best Practices

### Sicherheit

1. ✅ Niemals API Keys im Frontend
2. ✅ Verwendung von Umgebungsvariablen
3. ✅ HTTPS für Production
4. ✅ Rate Limiting implementieren
5. ✅ Input Validierung (Backend + Frontend)

### Performance

1. ✅ Status-Polling nur bei aktiven Anrufen
2. ✅ Caching für häufige RAG-Anfragen
3. ✅ CDN für Frontend Assets
4. ✅ Database für Call History (optional)

### User Experience

1. ✅ Klare Fehlermeldungen
2. ✅ Ladeanimationen
3. ✅ Status-Feedback in Echtzeit
4. ✅ Mobile-responsive Design
5. ✅ Accessibility (ARIA Labels)

---

## 🔮 Zukünftige Erweiterungen

Mögliche Features für die Zukunft:

### Call Recording
```python
# In VapiService
payload["recordingEnabled"] = True
```

### Call Analytics
```python
# Call History in Datenbank speichern
# Dashboard für Call-Statistiken
```

### Eingehende Anrufe
```python
# Webhook für eingehende Anrufe
@app.post("/api/call/incoming")
async def handle_incoming_call():
    # TwiML Response
```

### Multi-Language Support
```python
# Assistant mit mehreren Sprachen
assistantOverrides = {
    "language": "de" | "en" | "fr"
}
```

### CRM Integration
```python
# Salesforce, HubSpot Integration
# Automatisches Logging von Anrufen
```

---

## 📞 Support & Ressourcen

### Dokumentation

- **Vapi.ai Docs:** https://docs.vapi.ai
- **FastAPI Docs:** https://fastapi.tiangolo.com
- **React Docs:** https://react.dev

### Community

- **Vapi Discord:** https://discord.gg/vapi
- **GitHub Issues:** [Ihr Repository]

### Hilfe bekommen

Bei Fragen oder Problemen:

1. 📚 Lesen Sie VAPI_SETUP.md
2. 🐛 Schauen Sie im Troubleshooting Abschnitt
3. 📧 Kontaktieren Sie Vapi Support: support@vapi.ai
4. 💬 Fragen Sie in der Vapi Community

---

## ✅ Checkliste: Deployment

Vor dem Go-Live:

- [ ] Vapi.ai Account konfiguriert
- [ ] Telefonnummer gekauft und getestet
- [ ] Assistant vollständig konfiguriert (deutsche Prompts)
- [ ] Alle Umgebungsvariablen gesetzt
- [ ] Backend startet ohne Fehler
- [ ] Frontend baut ohne Fehler
- [ ] API Endpoints getestet (Postman/cURL)
- [ ] End-to-End Test erfolgreich
- [ ] Webhook funktioniert (RAG Integration)
- [ ] SSL Zertifikat installiert
- [ ] Domain konfiguriert
- [ ] Monitoring aufgesetzt
- [ ] Backup-Strategie definiert
- [ ] Credits auf Vapi Account aufgeladen
- [ ] Team geschult

---

## 🎉 Gratulation!

Sie haben erfolgreich einen vollständigen AI Call Assistant in Ihr MVA Versicherungs-System integriert!

### Was Sie jetzt können:

✅ AI-Anrufe an beliebige Telefonnummern starten  
✅ Status in Echtzeit verfolgen  
✅ Anrufe programmatisch beenden  
✅ RAG-System während Anrufen nutzen  
✅ Deutsche Sprachunterstützung  
✅ Production-ready Deployment  

**Viel Erfolg mit Ihrem AI Call Assistant! 🚀**

---

## 📝 Changelog

### Version 1.0.0 (2024-02-19)

**Neue Features:**
- ✨ Vollständige Vapi.ai Integration
- ✨ VapiService Backend Klasse
- ✨ 4 neue API Endpoints
- ✨ CallAssistant React Komponente
- ✨ Tab-Navigation im Frontend
- ✨ Real-time Status Updates
- ✨ Telefonnummer-Validierung
- ✨ Webhook für RAG-Integration
- ✨ Vollständige Dokumentation

**Dependencies:**
- Added: `httpx` (für async HTTP requests)

**Konfiguration:**
- Added: `VAPI_API_KEY` environment variable
- Added: `VAPI_ASSISTANT_ID` environment variable
- Added: `VAPI_PHONE_NUMBER_ID` environment variable

---

**Erstellt:** 2024-02-19  
**Autor:** AI Assistant  
**Lizenz:** Entsprechend Ihrem Projekt  
**Status:** ✅ Production Ready
