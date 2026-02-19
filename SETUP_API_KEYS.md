# 🔑 API Keys Setup - Schnellanleitung

## Ihre Vapi.ai API Keys:

- **Private Key (Backend):** `0b92e45d-69af-45ca-887a-37015fe31e97`
- **Public Key (Frontend):** `ff84605d-ec83-4b30-8a97-e4dbfd264e88`

---

## ⚡ Schnelle Einrichtung

### Schritt 1: Backend `.env` Datei erstellen

Führen Sie diesen Befehl im Projektroot aus:

**PowerShell:**
```powershell
@"
# OpenAI API Key (TODO: Fügen Sie Ihren Key ein)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
UVICORN_RELOAD=false
INIT_PIPELINE_ON_STARTUP=false
CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173

# Vapi.ai Configuration
VAPI_API_KEY=0b92e45d-69af-45ca-887a-37015fe31e97
VAPI_ASSISTANT_ID=your-assistant-id-here
VAPI_PHONE_NUMBER_ID=your-phone-number-id-here
"@ | Out-File -FilePath .env -Encoding UTF8
```

**Linux/Mac:**
```bash
cat > .env << 'EOF'
# OpenAI API Key (TODO: Fügen Sie Ihren Key ein)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
UVICORN_RELOAD=false
INIT_PIPELINE_ON_STARTUP=false
CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173

# Vapi.ai Configuration
VAPI_API_KEY=0b92e45d-69af-45ca-887a-37015fe31e97
VAPI_ASSISTANT_ID=your-assistant-id-here
VAPI_PHONE_NUMBER_ID=your-phone-number-id-here
EOF
```

### Schritt 2: Frontend `.env.local` Datei erstellen

Im `frontend/` Ordner:

**PowerShell:**
```powershell
cd frontend
@"
# Backend API URL
VITE_API_BASE_URL=http://localhost:8000

# Vapi.ai Public Key
VITE_VAPI_PUBLIC_KEY=ff84605d-ec83-4b30-8a97-e4dbfd264e88
"@ | Out-File -FilePath .env.local -Encoding UTF8
cd ..
```

**Linux/Mac:**
```bash
cd frontend
cat > .env.local << 'EOF'
# Backend API URL
VITE_API_BASE_URL=http://localhost:8000

# Vapi.ai Public Key
VITE_VAPI_PUBLIC_KEY=ff84605d-ec83-4b30-8a97-e4dbfd264e88
EOF
cd ..
```

---

## 📝 Manuelle Erstellung

Wenn die Befehle nicht funktionieren, erstellen Sie die Dateien manuell:

### Backend: `.env` (im Projektroot)

```env
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
UVICORN_RELOAD=false
INIT_PIPELINE_ON_STARTUP=false
CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173

# Vapi.ai
VAPI_API_KEY=0b92e45d-69af-45ca-887a-37015fe31e97
VAPI_ASSISTANT_ID=your-assistant-id-here
VAPI_PHONE_NUMBER_ID=your-phone-number-id-here
```

### Frontend: `frontend/.env.local`

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_VAPI_PUBLIC_KEY=ff84605d-ec83-4b30-8a97-e4dbfd264e88
```

---

## ✅ Nächste Schritte

### 1. OpenAI API Key hinzufügen

Ersetzen Sie `sk-your-openai-api-key-here` mit Ihrem echten OpenAI Key:
- Holen Sie sich von: https://platform.openai.com/api-keys

### 2. Vapi Assistant ID hinzufügen

1. Gehen Sie zu: https://dashboard.vapi.ai/assistants
2. Erstellen Sie einen neuen Assistant (siehe VAPI_SETUP.md)
3. Kopieren Sie die Assistant ID (z.B. `asst_abc123xyz`)
4. Ersetzen Sie `your-assistant-id-here` in `.env`

### 3. Vapi Phone Number ID hinzufügen

1. Gehen Sie zu: https://dashboard.vapi.ai/phone-numbers
2. Kaufen Sie eine deutsche Nummer
3. Kopieren Sie die Phone Number ID (z.B. `ph_xyz789abc`)
4. Ersetzen Sie `your-phone-number-id-here` in `.env`

### 4. Backend starten

```bash
# Virtual environment aktivieren
.venv\Scripts\Activate.ps1  # Windows
# oder
source .venv/bin/activate    # Linux/Mac

# Backend starten
python src/main.py
```

### 5. Frontend starten

```bash
cd frontend
npm install
npm run dev
```

---

## 🔒 Sicherheit

- ✅ `.env` ist bereits in `.gitignore` 
- ✅ Private Key wird **nur** im Backend verwendet
- ✅ Public Key ist **sicher** für Frontend
- ⚠️ **NIEMALS** `.env` committen oder teilen!

---

## 🐛 Troubleshooting

### "VAPI_API_KEY ist nicht konfiguriert"

```bash
# Überprüfen Sie, dass .env existiert
ls -la .env          # Linux/Mac
dir .env             # Windows

# Überprüfen Sie den Inhalt
cat .env             # Linux/Mac
type .env            # Windows

# Backend neu starten
python src/main.py
```

### Backend erkennt .env nicht

```bash
# Stellen Sie sicher, dass Sie im richtigen Ordner sind
pwd                  # Linux/Mac
cd                   # Windows

# .env muss im Projektroot sein (gleiche Ebene wie src/)
```

---

## 📚 Weitere Dokumentation

- **Vollständige Vapi Setup:** Siehe `VAPI_SETUP.md`
- **Vollständige Dokumentation:** Siehe `AI_CALL_ASSISTANT_README.md`

---

**Viel Erfolg! 🚀**
