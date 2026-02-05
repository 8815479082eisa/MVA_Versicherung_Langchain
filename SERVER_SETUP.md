# راهنمای راه‌اندازی پروژه روی سرور

## 1️⃣ SSH به سرور متصل شو
```bash
ssh user@your_server_ip
cd /home/user/projects  # یا مسیر دیگری
```

## 2️⃣ Clone کردن Repository
```bash
git clone https://github.com/8815479082eisa/MVA_Versicherung_Langchain.git
cd MVA_Versicherung_Langchain
git checkout new-feature-branch  # یا branch مورد نظر
```

## 3️⃣ نصب Python و Virtual Environment
```bash
# بررسی نسخه Python (باید 3.9 یا بالاتر)
python3 --version

# ایجاد Virtual Environment
python3 -m venv .venv

# فعال‌سازی Virtual Environment
source .venv/bin/activate  # روی Linux/Mac
# یا روی Windows:
.venv\Scripts\activate
```

## 4️⃣ نصب Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 5️⃣ تنظیم Environment Variables
```bash
# کپی کردن فایل نمونه
cp .env.example .env  # اگر وجود دارد
# یا ایجاد .env نو
nano .env  # یا vim .env
```

### محتوای .env:
```
# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
BACKEND_SECRET=your-secret-key

# OpenAI API
OPENAI_API_KEY=sk-xxxxxxxxxxxxx

# Database & Documents
PDF_DIRECTORY=/path/to/docs
CHROMA_PERSIST_DIRECTORY=/path/to/chroma_db

# API
API_BASE_URL=http://your_server_ip:8000
```

## 6️⃣ آزمایش Backend (روی سرور)
```bash
# بررسی اینکه Backend کار می‌کند
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# اگر کار کرد، Ctrl+C دکمه بزن
```

## 7️⃣ بیلد کردن Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

## 8️⃣ تنظیم Systemd Service برای Backend

### یک فایل سرویس بسازید:
```bash
sudo nano /etc/systemd/system/mva-backend.service
```

### محتوای فایل:
```ini
[Unit]
Description=MVA Insurance RAG Backend
After=network.target

[Service]
Type=notify
User=www-data          # یا username شما
WorkingDirectory=/home/user/projects/MVA_Versicherung_Langchain
Environment="PATH=/home/user/projects/MVA_Versicherung_Langchain/.venv/bin"
ExecStart=/home/user/projects/MVA_Versicherung_Langchain/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### فعال‌سازی سرویس:
```bash
sudo systemctl daemon-reload
sudo systemctl enable mva-backend.service
sudo systemctl start mva-backend.service
sudo systemctl status mva-backend.service
```

## 9️⃣ تنظیم Nginx برای Frontend و Reverse Proxy

### ایجاد Nginx Config:
```bash
sudo nano /etc/nginx/sites-available/mva-insurance
```

### محتوای فایل:
```nginx
upstream backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your_domain.com www.your_domain.com;

    # Frontend - Static Files
    location / {
        root /home/user/projects/MVA_Versicherung_Langchain/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API - Reverse Proxy
    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health Check
    location /health {
        proxy_pass http://backend;
    }
}
```

### فعال‌سازی Nginx Config:
```bash
sudo ln -s /etc/nginx/sites-available/mva-insurance /etc/nginx/sites-enabled/
sudo nginx -t  # بررسی درستی config
sudo systemctl restart nginx
```

## 🔟 SSL Certificate (اختیاری - اما توصیه می‌شود)
```bash
# نصب Certbot
sudo apt install certbot python3-certbot-nginx

# دریافت SSL Certificate
sudo certbot --nginx -d your_domain.com -d www.your_domain.com
```

## ✅ بررسی نهایی

```bash
# 1. بررسی Backend
curl http://localhost:8000/health

# 2. بررسی Frontend
curl http://localhost:80/

# 3. بررسی API
curl http://localhost:8000/api/ask -X POST -H "Content-Type: application/json" -d '{"question":"سلام"}'

# 4. بررسی log های سرویس
sudo journalctl -u mva-backend.service -f
```

## 🔧 نکات مهم

1. **OPENAI_API_KEY**: حتماً قبل از راه‌اندازی در .env تنظیم کنید
2. **پوشه docs**: حتماً فایل‌های PDF را در `./docs` قرار دهید
3. **Database**: اولین بار اجرا شدن، Chroma DB خودکار ایجاد می‌شود
4. **Permissions**: اگر از Nginx استفاده می‌کنید، اطمینان حاصل کنید که فایل‌ها قابل دسترسی هستند

## 📊 Monitoring

```bash
# بررسی استفاده CPU و Memory
top

# بررسی Disk Space
df -h

# بررسی Network Ports
sudo netstat -tulpn | grep LISTEN
```

---

سوالی داری؟ بپرس! 🚀
