#!/bin/bash
# Quick Start Script برای راه‌اندازی سریع روی سرور

set -e

echo "🚀 MVA Insurance RAG System - Server Setup"
echo "=========================================="

# رنگ‌ها
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. بررسی Python
echo -e "\n${YELLOW}1️⃣ بررسی Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✅ Python $PYTHON_VERSION موجود است${NC}"
else
    echo -e "${RED}❌ Python3 نصب نشده است. دستور نصب: sudo apt install python3${NC}"
    exit 1
fi

# 2. ایجاد Virtual Environment
echo -e "\n${YELLOW}2️⃣ ایجاد Virtual Environment...${NC}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual Environment ایجاد شد${NC}"
else
    echo -e "${GREEN}✅ Virtual Environment قبلاً موجود است${NC}"
fi

# 3. فعال‌سازی Virtual Environment
source .venv/bin/activate
echo -e "${GREEN}✅ Virtual Environment فعال شد${NC}"

# 4. نصب Dependencies
echo -e "\n${YELLOW}3️⃣ نصب Dependencies...${NC}"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies نصب شدند${NC}"

# 5. بررسی .env File
echo -e "\n${YELLOW}4️⃣ بررسی Environment Variables...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  فایل .env موجود نیست${NC}"
    cp .env.example .env 2>/dev/null || {
        cat > .env << 'EOF'
# Backend Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# OpenAI API Key
OPENAI_API_KEY=sk-your-api-key-here

# Database
PDF_DIRECTORY=./docs
CHROMA_PERSIST_DIRECTORY=./chroma_db

# API URL
API_BASE_URL=http://localhost:8000
EOF
        echo -e "${YELLOW}⚠️  فایل .env ایجاد شد - لطفاً OPENAI_API_KEY را تنظیم کنید${NC}"
    }
else
    echo -e "${GREEN}✅ فایل .env موجود است${NC}"
fi

# 6. بررسی پوشه Docs
echo -e "\n${YELLOW}5️⃣ بررسی پوشه‌های لازم...${NC}"
mkdir -p docs chroma_db
echo -e "${GREEN}✅ پوشه‌ها ایجاد شدند${NC}"

# 7. آزمایش Backend
echo -e "\n${YELLOW}6️⃣ آزمایش Backend...${NC}"
timeout 10 python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 5

if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ Backend درست کار می‌کند${NC}"
    kill $BACKEND_PID 2>/dev/null || true
else
    echo -e "${RED}❌ Backend به مشکل خورد${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# 8. بیلد Frontend
echo -e "\n${YELLOW}7️⃣ بیلد کردن Frontend...${NC}"
cd frontend
npm install
npm run build
cd ..
echo -e "${GREEN}✅ Frontend بیلد شد${NC}"

# 9. اطلاعات نهایی
echo -e "\n${GREEN}=========================================="
echo "✅ Setup تکمیل شد!"
echo "==========================================${NC}"
echo -e "\n${YELLOW}مراحل بعدی:${NC}"
echo "1. ویرایش فایل .env و تنظیم OPENAI_API_KEY"
echo "2. شروع Backend: python -m uvicorn src.main:app --host 0.0.0.0 --port 8000"
echo "3. یا استفاده از Systemd Service: sudo systemctl start mva-backend"
echo "4. Frontend بر روی http://localhost:3000 اجرا می‌شود"
echo ""
echo -e "${YELLOW}برای اطلاعات بیشتر:${NC}"
echo "📖 ببینید: SERVER_SETUP.md"
