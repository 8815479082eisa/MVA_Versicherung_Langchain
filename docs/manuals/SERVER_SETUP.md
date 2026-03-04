# Server Setup Guide

## 1. Connect to the server
```bash
ssh user@your_server_ip
cd /home/user/projects
```
Adjust the path above if you keep repositories elsewhere.

## 2. Clone the repository
```bash
git clone https://github.com/8815479082eisa/MVA_Versicherung_Langchain.git
cd MVA_Versicherung_Langchain
git checkout main  # or your deployment branch
```

## 3. Install Python and create the virtual environment
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## 4. Install project dependencies
```bash
pip install -r requirements.txt
```

## 5. Prepare the environment file
```bash
cp .env.example .env  # if the example exists
nano .env
```
Populate the key values (especially `OPENAI_API_KEY`, `PDF_DIRECTORY`, `CHROMA_PERSIST_DIRECTORY`, and `AUDIT_LOG_FILE`). The script also defaults `DATA_DIR` to `./data`.

## 6. Place the PDF documents
```bash
mkdir -p data/raw/pdfs
```
Drop the latest insurance PDFs into `data/raw/pdfs`. The backend compares hash sums; adding or replacing files triggers an index rebuild automatically. For a clean rebuild, remove `data/processed/vectorstores/chroma_db` and `data/processed/caches/pdf_hashes.json` before restarting the app.

## 7. Run the setup script (optional but helpful)
```bash
bash setup.sh
```
It verifies the Python toolchain, dependencies, directories, backend health, and builds the frontend output.

## 8. Start the backend for the first time
```bash
source .venv/bin/activate
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```
UVicorn will detect new documents and rebuild the Chroma index. Leave this running while you validate the stack.

## 9. Build the frontend
```bash
cd frontend
npm install
npm run build
cd ..
```
Use `npm run dev` for local development instead of a production build.

## 10. Configure the systemd service
Copy a service file into `/etc/systemd/system`:
```bash
sudo cp docs/development/mva-backend.service /etc/systemd/system/mva-backend.service
sudo nano /etc/systemd/system/mva-backend.service
```
Adjust `User`, `WorkingDirectory`, and `ExecStart` so they point to your deployment paths and virtual environment. Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mva-backend.service
sudo systemctl start mva-backend.service
sudo systemctl status mva-backend.service
```

## 11. Configure Nginx as a reverse proxy
```bash
sudo cp docker/nginx/nginx-mva-insurance.conf /etc/nginx/sites-available/mva-insurance
sudo nano /etc/nginx/sites-available/mva-insurance  # set server_name and root
sudo ln -s /etc/nginx/sites-available/mva-insurance /etc/nginx/sites-enabled/mva-insurance
sudo nginx -t
sudo systemctl restart nginx
```
The configuration proxies `/api/` and `/health` to the backend and serves the `frontend/dist` build for all other routes.

## 12. Optional: obtain SSL certificates
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain.com -d www.your_domain.com
```

## 13. Verification checklist
- Backend: `curl http://localhost:8000/health`
- Frontend: `curl http://localhost/` or `http://localhost:80/`
- API: `curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" -d '{"question":"What is the coverage for tariff X?"}'`
- Logs: `sudo journalctl -u mva-backend.service -f` or `docker compose -f docker/docker-compose.yml logs -f`

## 14. Tips for stability
- Always keep `OPENAI_API_KEY` up to date inside `.env` before starting the backend.
- Store PDFs inside `data/raw/pdfs/` and trigger the backend to rebuild the index whenever they change.
- Protect the frontend with HTTPS in production and open firewall ports 80/443/8000 only as needed.
- Back up `data/processed/vectorstores/chroma_db` and `data/raw/pdfs/` regularly, especially before replacing documents.
