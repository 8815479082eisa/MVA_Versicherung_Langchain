## Quick Start Guide for Production Server

### **Option 1: Manual setup (recommended for debugging or development)**
```bash
## 1. Connect to the server and go to the project directory
ssh user@your_server_ip
cd /home/user/projects

## 2. Clone the repository and switch to the working branch
git clone https://github.com/8815479082eisa/MVA_Versicherung_Langchain.git
cd MVA_Versicherung_Langchain
git checkout main  # or your feature branch

## 3. Run the setup script to install dependencies and prepare the workspace
bash setup.sh

## 4. Fill in settings
nano .env   # set OLLAMA_BASE_URL, model roles, PDF_DIRECTORY, CHROMA_PERSIST_DIRECTORY, etc.

## 5. Start the backend (UVicorn detects PDF changes and rebuilds the index automatically)
source .venv/bin/activate
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

## 6. Build or preview the frontend in another shell
cd frontend
npm install
npm run build  # or npm run dev for development
```

---

### **Option 2: Docker compose (preferred for production packing)**
```bash
## 1. Install Docker / Docker Compose
sudo apt update
sudo apt install docker.io docker-compose

## 2. Clone repository and configure .env (copy example if needed)
git clone https://github.com/8815479082eisa/MVA_Versicherung_Langchain.git
cd MVA_Versicherung_Langchain
cp .env.example .env
nano .env  # update OLLAMA_BASE_URL, model roles, PDF_DIRECTORY, CHROMA_PERSIST_DIRECTORY

## 3. Launch services
docker compose -f docker/docker-compose.yml up -d

## 4. Follow backend logs
docker compose -f docker/docker-compose.yml logs -f backend
```

**Access**:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:80`
- API docs: `http://localhost:8000/docs`

---

### **Option 3: Systemd service (recommended for long-running deployments)**
```bash
## 1. Copy the service file into place
sudo cp docs/development/mva-backend.service /etc/systemd/system/mva-backend.service
sudo nano /etc/systemd/system/mva-backend.service  # adapt paths/user

## 2. Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable mva-backend.service
sudo systemctl start mva-backend.service

## 3. Check service status and follow logs
sudo systemctl status mva-backend.service
sudo journalctl -u mva-backend.service -f

## 4. Configure Nginx as reverse proxy
sudo cp docker/nginx/nginx-mva-insurance.conf /etc/nginx/sites-available/mva-insurance
sudo nano /etc/nginx/sites-available/mva-insurance  # set server_name and root
sudo ln -s /etc/nginx/sites-available/mva-insurance /etc/nginx/sites-enabled/mva-insurance
sudo nginx -t
sudo systemctl restart nginx
```

---

## Configuration references
| File | Purpose |
|------|---------|
| `docs/manuals/SERVER_SETUP.md` | Full deployment checklist for admins |
| `setup.sh` | Automated setup helper that checks Python, dependencies, and build steps |
| `docs/development/mva-backend.service` | Systemd unit file for the backend |
| `docker/nginx/nginx-mva-insurance.conf` | Nginx reverse proxy configuration |
| `docker/docker-compose.yml` | Docker compose definition for backend + frontend |
| `docker/Dockerfile` | Backend container definition |

---

## Sanity checks (run after deploy)
```bash
## 1. Backend health
curl http://localhost:8000/health

## 2. Frontend asset server
curl http://localhost:80/

## 3. Query the API
echo '{"question":"What is the coverage for tariff X?"}' | curl -s http://localhost:8000/api/ask -X POST -H "Content-Type: application/json" -d @-

## 4. Follow logs
sudo journalctl -u mva-backend.service -f
# or docker compose -f docker/docker-compose.yml logs -f
```

---

## Key reminders
- Always set `OLLAMA_BASE_URL` and role models inside `.env` before starting the backend.
- Add or refresh PDFs inside `data/raw/pdfs/` whenever the knowledge base changes.
- The backend automatically rebuilds the Chroma index when it detects new/changed PDFs, but deleting `data/processed/vectorstores/chroma_db` and `data/processed/caches/pdf_hashes.json` enforces a full rebuild before restart.
- Serve the frontend over HTTPS behind Nginx in production; adjust firewall rules for ports 80/443/8000.
- Back up `data/processed/vectorstores/chroma_db` and `data/raw/pdfs/` regularly.
