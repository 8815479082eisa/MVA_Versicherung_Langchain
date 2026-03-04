#!/bin/bash
# Quick start script for MVA Insurance RAG server setup

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "MVA Insurance RAG System - Server Setup"
echo "========================================"

### 1. Python
echo -e "\n${YELLOW}1. Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}Python ${PYTHON_VERSION} is available${NC}"
else
    echo -e "${RED}Python3 is not installed. Install: sudo apt install python3${NC}"
    exit 1
fi

### 2. Virtual environment
echo -e "\n${YELLOW}2. Creating virtual environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}.venv created${NC}"
else
    echo -e "${GREEN}.venv already exists${NC}"
fi

### 3. Activate venv
source .venv/bin/activate
echo -e "${GREEN}Virtual environment activated${NC}"

### 4. Install dependencies
echo -e "\n${YELLOW}3. Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed${NC}"

### 5. Prepare .env
echo -e "\n${YELLOW}4. Ensuring .env file exists${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        cat <<'EOF' > .env
# Backend configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here

# Data paths
DATA_DIR=./data
PDF_DIRECTORY=./data/raw/pdfs
CHROMA_PERSIST_DIRECTORY=./data/processed/vectorstores/chroma_db
AUDIT_LOG_FILE=./data/processed/logs/audit.log
PDF_HASH_FILE=./data/processed/caches/pdf_hashes.json
MODEL_CONFIG_FILE=./data/processed/caches/model_config.json

# API URL
API_BASE_URL=http://localhost:8000
EOF
    fi
    echo -e "${YELLOW}.env created; please edit OPENAI_API_KEY and paths${NC}"
else
    echo -e "${GREEN}.env already exists${NC}"
fi

### 6. Ensure document directories
echo -e "\n${YELLOW}5. Preparing data folders...${NC}"
mkdir -p data/raw/pdfs data/processed/vectorstores/chroma_db data/processed/logs data/processed/caches
echo -e "${GREEN}Required directories are in place${NC}"

### 7. Health check after starting backend briefly
echo -e "\n${YELLOW}6. Verifying backend startup...${NC}"
timeout 10 python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 5
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}Backend responded to health check${NC}"
    kill $BACKEND_PID 2>/dev/null || true
else
    echo -e "${RED}Backend did not start correctly${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

### 8. Build frontend
echo -e "\n${YELLOW}7. Building frontend...${NC}"
pushd frontend > /dev/null
npm install
npm run build
popd > /dev/null
echo -e "${GREEN}Frontend build complete${NC}"

### Final reminder
echo -e "\n========================================"
echo "Setup complete!"
echo "========================================"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit .env to set OPENAI_API_KEY and data paths."
echo "2. Start the backend with: python -m uvicorn src.main:app --host 0.0.0.0 --port 8000."
echo "3. Optionally run the systemd service (see docs/development/mva-backend.service)."
echo "4. Serve the frontend through Nginx or open port 3000 for dev tooling."
echo ""
echo -e "${YELLOW}Additional references:${NC}"
echo "docs/manuals/SERVER_SETUP.md for the full checklist."
