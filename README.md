# Baby Monitor – Motion & Fall Detection

## Setup

Clone: git clone <repo>
cd baby_monitor
pip install -r requirements.txt
## Run locally

python app.py

Navigate to http://localhost:5000

---

## 🎯 Remote Deployment

### Using a Linux VPS (e.g., Ubuntu)

git clone project.
Install packages and system deps:
sudo apt update sudo apt install python3 python3-venv libglib2.0-0 python3 -m venv venv source venv/bin/activate pip install -r requirements.txt

Expose RTSP camera (if remote).
Run:
nohup gunicorn app:app -b 0.0.0.0:5000 --workers 2 &

Setup a reverse proxy (Nginx):
```nginx
server {
listen 80;
server_name your-domain.com;
location / {
 proxy_pass http://127.0.0.1:5000;
 proxy_http_version 1.1;
 proxy_set_header Upgrade $http_upgrade;
 proxy_set_header Connection keep-alive;
 proxy_set_header Host $host;
 proxy_cache_bypass $http_upgrade;
}
}

sudo systemctl restart nginx

Access via http://your-domain.com