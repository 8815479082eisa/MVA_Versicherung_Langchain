## 🚀 راه‌اندازی سریع روی سرور

### **گزینه 1: راه‌اندازی دستی (ساده‌تر برای شروع)**

```bash
# 1. متصل شو به سرور
ssh user@your_server_ip

# 2. Clone کردن پروژه
git clone https://github.com/8815479082eisa/MVA_Versicherung_Langchain.git
cd MVA_Versicherung_Langchain

# 3. اجرای Setup Script
bash setup.sh

# 4. تنظیم API Key
nano .env
# ویرایش OPENAI_API_KEY

# 5. شروع Backend
source .venv/bin/activate
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# 6. در Terminal دیگر شروع Frontend
cd frontend
npm run preview
```

---

### **گزینه 2: استفاده از Docker (برای Production)**

```bash
# 1. نصب Docker و Docker Compose
sudo apt update
sudo apt install docker.io docker-compose

# 2. Clone پروژه
git clone https://github.com/8815479082eisa/MVA_Versicherung_Langchain.git
cd MVA_Versicherung_Langchain

# 3. تنظیم .env
cp .env.example .env
nano .env  # تنظیم OPENAI_API_KEY

# 4. اجرای Docker Compose
docker-compose up -d

# 5. بررسی وضعیت
docker-compose logs -f backend
```

**آدرس‌های دسترسی:**
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:80`
- API Docs: `http://localhost:8000/docs`

---

### **گزینه 3: استفاده از Systemd Service (برای بهترین کنترل)**

```bash
# 1. تنظیم Backend Service
sudo cp mva-backend.service /etc/systemd/system/
sudo nano /etc/systemd/system/mva-backend.service
# ویرایش paths

# 2. فعال‌سازی
sudo systemctl daemon-reload
sudo systemctl enable mva-backend.service
sudo systemctl start mva-backend.service

# 3. بررسی وضعیت
sudo systemctl status mva-backend.service
sudo journalctl -u mva-backend.service -f

# 4. تنظیم Nginx
sudo cp nginx-mva-insurance.conf /etc/nginx/sites-available/mva-insurance
sudo nano /etc/nginx/sites-available/mva-insurance  # ویرایش domain
sudo ln -s /etc/nginx/sites-available/mva-insurance /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 📝 فایل‌های تنظیمات

| فایل | توضیح |
|------|-------|
| `SERVER_SETUP.md` | راهنمای کامل (به زبان فارسی) |
| `setup.sh` | اسکریپت خودکار برای تنظیم |
| `mva-backend.service` | Systemd Service برای Backend |
| `nginx-mva-insurance.conf` | تنظیمات Nginx |
| `docker-compose.yml` | Docker Compose Configuration |
| `Dockerfile` | Docker Image برای Backend |

---

## ✅ بررسی‌های اساسی

```bash
# 1. بررسی Backend
curl http://localhost:8000/health

# 2. بررسی Frontend
curl http://localhost:80/

# 3. بررسی API
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"سلام"}'

# 4. Log‌ها
sudo journalctl -u mva-backend.service -f  # Backend logs
docker-compose logs -f  # Docker logs
```

---

## ⚠️ نکات مهم

✅ تنظیم `OPENAI_API_KEY` در `.env` الزامی است  
✅ حتماً فایل‌های PDF را در پوشه `docs/` قرار دهید  
✅ از HTTPS استفاده کنید در Production  
✅ تنظیمات Firewall را بررسی کنید (Port 80, 443, 8000)  
✅ Regular backup از `chroma_db` و `docs` بگیرید  

---

سؤالی داری؟ 🤔
