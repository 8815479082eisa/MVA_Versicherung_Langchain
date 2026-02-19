# Vapi.ai Setup-Anleitung

Diese Anleitung erklärt, wie Sie die AI Call Assistant Funktion mit Vapi.ai in Ihrem MVA Versicherung System einrichten.

## Übersicht

Mit der Vapi.ai Integration können Sie:
- ✅ AI-gestützte Telefonanrufe initiieren
- ✅ Kunden automatisch mit einem intelligenten Voice Assistant verbinden
- ✅ Das RAG-System während Anrufen nutzen (für Dokumentenabfragen)
- ✅ Anrufstatus in Echtzeit verfolgen

## Schritt 1: Vapi.ai Account erstellen

1. Gehen Sie zu [https://vapi.ai](https://vapi.ai)
2. Klicken Sie auf "Sign Up" und erstellen Sie einen Account
3. Verifizieren Sie Ihre E-Mail-Adresse
4. Loggen Sie sich ins Dashboard ein: [https://dashboard.vapi.ai](https://dashboard.vapi.ai)

## Schritt 2: API Key erhalten

1. Gehen Sie zu **Account Settings**: [https://dashboard.vapi.ai/account](https://dashboard.vapi.ai/account)
2. Unter "API Keys" finden Sie Ihren API Key
3. Kopieren Sie den Key (Sie benötigen ihn später)

## Schritt 3: Telefonnummer kaufen

1. Navigieren Sie zu **Phone Numbers**: [https://dashboard.vapi.ai/phone-numbers](https://dashboard.vapi.ai/phone-numbers)
2. Klicken Sie auf **"Buy Number"**
3. Wählen Sie:
   - Land: **Germany (+49)**
   - Nummer auswählen (z.B. +49 30 1234567)
4. Bestätigen Sie den Kauf
5. **Wichtig:** Kopieren Sie die **Phone Number ID** (nicht die Nummer selbst!)
   - Sieht aus wie: `ph_1234567890abcdef`

## Schritt 4: AI Assistant erstellen

1. Gehen Sie zu **Assistants**: [https://dashboard.vapi.ai/assistants](https://dashboard.vapi.ai/assistants)
2. Klicken Sie auf **"Create Assistant"**

### Basis-Konfiguration

```
Name: MVA Versicherungs Assistent
Description: AI Assistant für MVA Versicherungsanfragen
```

### Voice Settings

```
Provider: ElevenLabs (oder Azure/Google)
Voice: 
  - ElevenLabs: "Daniel" oder "Serena" (deutsche Stimmen)
  - Azure: "de-DE-KatjaNeural" (weiblich) oder "de-DE-ConradNeural" (männlich)
  - Google: "de-DE-Neural2-F" (weiblich) oder "de-DE-Neural2-D" (männlich)

Language: German (de-DE)
```

### Model Settings

```
Provider: OpenAI
Model: gpt-4-turbo (empfohlen) oder gpt-4
Temperature: 0.7
```

### System Prompt (Deutsch)

```
Du bist ein professioneller Versicherungsassistent für MVA (Versicherungsunternehmen).

Deine Aufgaben:
- Beantworte Kundenfragen zu Versicherungsprodukten höflich und professionell
- Sprich immer auf Deutsch
- Sei hilfsbereit und geduldig
- Wenn du Informationen aus Dokumenten benötigst, nutze die Function "search_insurance_info"
- Fasse komplexe Informationen verständlich zusammen
- Bei rechtlichen Fragen weise darauf hin, dass dies keine rechtliche Beratung ersetzt

Verhalten:
- Begrüße den Anrufer freundlich: "Guten Tag! Ich bin Ihr MVA Versicherungs-Assistent. Wie kann ich Ihnen heute helfen?"
- Stelle Rückfragen, wenn etwas unklar ist
- Sei empathisch und verstehe die Bedürfnisse des Kunden
- Am Ende des Gesprächs: "Gibt es noch etwas, womit ich Ihnen helfen kann?"
```

### Transcriber Settings (Speech-to-Text)

```
Provider: Deepgram
Model: nova-2
Language: German (de)
```

### Function Configuration (für RAG-Integration)

Fügen Sie diese Function hinzu, damit der Assistant auf Ihr RAG-System zugreifen kann:

**Function Name:**
```
search_insurance_info
```

**Description:**
```
Sucht nach Informationen in den MVA Versicherungsdokumenten. Nutze diese Function, wenn du spezifische Informationen über Versicherungsprodukte, Bedingungen, Tarife oder Leistungen benötigst.
```

**Parameters Schema:**
```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "Die Frage, die im Versicherungsdokumentensystem gesucht werden soll"
    }
  },
  "required": ["question"]
}
```

**Server URL:**
```
https://ihre-domain.com/api/call/vapi-webhook
```

(Ersetzen Sie `ihre-domain.com` mit Ihrer tatsächlichen Domain oder ngrok-URL für Tests)

### 5. Assistant speichern

- Klicken Sie auf **"Create"** oder **"Save"**
- **Wichtig:** Kopieren Sie die **Assistant ID**
  - Sieht aus wie: `asst_1234567890abcdef`

## Schritt 5: Umgebungsvariablen konfigurieren

Erstellen Sie eine `.env` Datei im Projektroot (falls noch nicht vorhanden):

```bash
# ============================================================================
# Vapi.ai Configuration
# ============================================================================

# Von Schritt 2: API Key
VAPI_API_KEY=sk_live_your_actual_vapi_api_key_here

# Von Schritt 5: Assistant ID
VAPI_ASSISTANT_ID=asst_1234567890abcdef

# Von Schritt 3: Phone Number ID
VAPI_PHONE_NUMBER_ID=ph_1234567890abcdef

# Andere bestehende Konfigurationen...
OPENAI_API_KEY=sk-your-openai-key
BACKEND_PORT=8000
```

**⚠️ Wichtig:** Fügen Sie `.env` zu Ihrer `.gitignore` hinzu!

## Schritt 6: Dependencies installieren

```bash
pip install -r requirements.txt
```

Die neue Abhängigkeit `httpx` wurde bereits zu `requirements.txt` hinzugefügt.

## Schritt 7: Backend starten

```bash
python src/main.py
```

Oder wenn Sie die alte Struktur nutzen:

```bash
python main.py
```

Das Backend sollte nun starten mit den neuen Vapi-Endpoints:
- `POST /api/call/start` - Anruf initiieren
- `GET /api/call/status/{call_id}` - Status abrufen
- `POST /api/call/end` - Anruf beenden
- `POST /api/call/vapi-webhook` - Webhook für Vapi

## Schritt 8: Testing mit Postman oder cURL

### Anruf starten

```bash
curl -X POST http://localhost:8000/api/call/start \
  -H "Content-Type: application/json" \
  -d '{
    "phoneNumber": "+491234567890",
    "customerName": "Max Mustermann"
  }'
```

**Response:**
```json
{
  "callId": "call_xyz123",
  "status": "initiated",
  "phoneNumber": "+491234567890",
  "createdAt": "2024-01-15T10:30:00Z"
}
```

### Status abrufen

```bash
curl http://localhost:8000/api/call/status/call_xyz123
```

### Anruf beenden

```bash
curl -X POST http://localhost:8000/api/call/end \
  -H "Content-Type: application/json" \
  -d '{"callId": "call_xyz123"}'
```

## Schritt 9: Webhook konfigurieren (für Production)

Für die volle Integration (RAG während Anrufen):

### Option A: Ngrok für lokales Testing

```bash
# Ngrok installieren: https://ngrok.com
ngrok http 8000
```

Kopieren Sie die ngrok-URL (z.B. `https://abc123.ngrok.io`) und setzen Sie in Vapi:

**Webhook URL:**
```
https://abc123.ngrok.io/api/call/vapi-webhook
```

### Option B: Production Server

Wenn Ihr Server öffentlich erreichbar ist:

**Webhook URL:**
```
https://ihre-domain.com/api/call/vapi-webhook
```

**Events aktivieren:**
- ✅ `call-started`
- ✅ `call-ended`
- ✅ `function-call`

## Schritt 10: Frontend nutzen (nächster Schritt)

Sobald das Backend läuft, können Sie die neue **Call Assistant UI** im Frontend verwenden:

1. Frontend starten:
   ```bash
   cd frontend
   npm install  # falls noch nicht gemacht
   npm run dev
   ```

2. Öffnen Sie [http://localhost:5173](http://localhost:5173)

3. Sie sehen jetzt einen neuen "AI-Anruf starten" Button

4. Telefonnummer eingeben (z.B. `+491234567890`) und testen!

## Troubleshooting

### "VAPI_API_KEY ist nicht konfiguriert"

- Überprüfen Sie, dass `.env` im Projektroot liegt
- Stellen Sie sicher, dass `python-dotenv` installiert ist
- Starten Sie das Backend neu

### "Ungültige Telefonnummer"

- Verwenden Sie das E.164 Format: `+49` gefolgt von der Nummer
- Keine Leerzeichen, Bindestriche oder Klammern
- Beispiel: `+491234567890` ✅
- Falsch: `0123 456 7890` ❌

### "Anruf wird nicht initiiert"

1. Überprüfen Sie die API Keys in `.env`
2. Stellen Sie sicher, dass Sie Credits auf Ihrem Vapi-Account haben
3. Prüfen Sie die Logs: `tail -f backend.log`
4. Testen Sie die Vapi API direkt mit Postman

### "Function Call funktioniert nicht"

1. Webhook URL in Vapi korrekt konfiguriert?
2. Server öffentlich erreichbar? (Nutzen Sie ngrok für lokales Testing)
3. Überprüfen Sie die Logs auf eingehende Webhook-Requests

## Kosten

**Vapi.ai Pricing (ca.):**
- 🇩🇪 Deutsche Nummer: ~$2-5/Monat
- 📞 Anrufe: ~$0.05-0.15 pro Minute
- 🎤 ElevenLabs Stimmen: Premium, in Minute-Rate enthalten
- 🤖 OpenAI GPT-4: Separat abgerechnet über Ihr OpenAI-Konto

**Tipp:** Starten Sie mit einem kleinen Test-Budget und skalieren Sie nach Bedarf.

## Support

- 📧 Vapi Support: [support@vapi.ai](mailto:support@vapi.ai)
- 📚 Vapi Dokumentation: [https://docs.vapi.ai](https://docs.vapi.ai)
- 💬 Vapi Discord: [https://discord.gg/vapi](https://discord.gg/vapi)

## Nächste Schritte

✅ Backend ist konfiguriert  
⏭️ Jetzt: Frontend-Integration mit der CallAssistant Komponente  
🚀 Production: Server deployen und DNS konfigurieren

---

**Viel Erfolg mit Ihrem AI Call Assistant! 🎉**
