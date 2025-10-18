# Deployment & Security Guide

## 🔒 Security Checklist

### 1. Environment Variables & Secrets
Never commit sensitive data. Use environment variables or secrets management.

```bash
# .streamlit/secrets.toml (NEVER commit this)
[database]
host = "your-db-host"
password = "your-password"

[api]
key = "your-api-key"
```

Add to `.gitignore`:
```
.streamlit/secrets.toml
*.env
.env.local
```

### 2. Rate Limiting
Prevent abuse by limiting requests per user.

```python
# Add to src/dashboard.py
import streamlit as st
from datetime import datetime, timedelta

def check_rate_limit():
    """Simple rate limiting - 10 predictions per hour per session"""
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
        st.session_state.last_reset = datetime.now()
    
    # Reset counter every hour
    if datetime.now() - st.session_state.last_reset > timedelta(hours=1):
        st.session_state.prediction_count = 0
        st.session_state.last_reset = datetime.now()
    
    if st.session_state.prediction_count >= 10:
        st.error("Rate limit exceeded. Please try again in an hour.")
        return False
    
    return True

# Use in predict button
if st.button("Generate Predictions"):
    if check_rate_limit():
        # ... existing prediction code
        st.session_state.prediction_count += 1
```

### 3. Input Validation
Always validate user inputs.

```python
def validate_inputs(season: int, week: int) -> bool:
    """Validate prediction inputs"""
    current_year = datetime.now().year
    
    if not (2010 <= season <= current_year + 1):
        st.error(f"Season must be between 2010 and {current_year + 1}")
        return False
    
    if not (1 <= week <= 18):
        st.error("Week must be between 1 and 18")
        return False
    
    return True
```

### 4. HTTPS/SSL
**Streamlit Cloud:** Automatic HTTPS ✅  
**AWS:** Use AWS Certificate Manager + ALB  
**Others:** Let's Encrypt (free SSL)

### 5. Authentication (Optional)
If you want to restrict access:

```python
# Add to src/dashboard.py
import streamlit as st
import hmac

def check_password():
    """Returns True if user entered correct password"""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    # Show input for password
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

# Add at start of main()
if not check_password():
    st.stop()
```

### 6. Model File Security
Your model files are large - don't commit them to GitHub.

**Current Status:** ✅ Already in `.gitignore`

**For Production:**
- Store models in S3/cloud storage
- Download on app startup
- Use versioning for model updates

```python
# Add to src/dashboard.py
import boto3
import os

def download_model_from_s3():
    """Download latest model from S3 if not present"""
    model_path = "models/random_forest_latest.pkl"
    
    if not os.path.exists(model_path):
        s3 = boto3.client('s3')
        s3.download_file(
            'your-bucket-name', 
            'models/random_forest_latest.pkl',
            model_path
        )
```

### 7. API Security (Flask API)
If deploying the Flask API (`src/api.py`):

```python
# Add CORS protection
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# Configure CORS - only allow your frontend domain
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdashboard.streamlit.app"]
    }
})

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    # ... existing code
```

### 8. Logging & Monitoring
Track usage and errors.

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log predictions
def log_prediction(season, week, user_ip):
    logger.info(f"Prediction: season={season}, week={week}, ip={user_ip}")

# Log errors
try:
    predictions = predictor.predict(season, week)
except Exception as e:
    logger.error(f"Prediction failed: {str(e)}", exc_info=True)
    st.error("An error occurred. Please try again.")
```

### 9. Content Security Policy (CSP)
Prevent XSS attacks.

```python
# Add to .streamlit/config.toml
[server]
enableXsrfProtection = true
enableCORS = false
```

### 10. Data Backup
Regularly backup your data and models.

```bash
# AWS S3 backup script
aws s3 sync data/raw/ s3://your-bucket/backups/data/raw/
aws s3 sync models/ s3://your-bucket/backups/models/
```

---

## 🚀 Deployment Instructions

### Option 1: Streamlit Community Cloud (RECOMMENDED)

1. **Push to GitHub** (Already done ✅)

2. **Go to:** https://share.streamlit.io

3. **Click:** "New app"

4. **Configure:**
   - Repository: `joemm24/nfl-prediction-model`
   - Branch: `main`
   - Main file path: `src/dashboard.py`

5. **Advanced settings:**
   - Python version: `3.11`

6. **Click:** "Deploy!"

**That's it!** Your app will be live at: `https://[your-app-name].streamlit.app`

---

### Option 2: AWS EC2 (Advanced)

**Prerequisites:**
- AWS Account
- Domain name (optional, for custom URL)

**Steps:**

1. **Launch EC2 Instance**
```bash
# Choose: Ubuntu Server 22.04 LTS
# Instance type: t3.small (2GB RAM)
# Security Group: Allow ports 22, 80, 443
```

2. **SSH into instance**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Setup environment**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install nginx (reverse proxy)
sudo apt install nginx

# Clone your repo
git clone https://github.com/joemm24/nfl-prediction-model.git
cd nfl-prediction-model

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

4. **Configure systemd service**
```bash
sudo nano /etc/systemd/system/nfl-dashboard.service
```

```ini
[Unit]
Description=NFL Prediction Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/nfl-prediction-model
Environment="PATH=/home/ubuntu/nfl-prediction-model/venv/bin"
ExecStart=/home/ubuntu/nfl-prediction-model/venv/bin/streamlit run src/dashboard.py --server.port=8501 --server.address=localhost

Restart=always

[Install]
WantedBy=multi-user.target
```

5. **Configure Nginx reverse proxy**
```bash
sudo nano /etc/nginx/sites-available/nfl-dashboard
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

6. **Enable and start services**
```bash
sudo ln -s /etc/nginx/sites-available/nfl-dashboard /etc/nginx/sites-enabled/
sudo systemctl restart nginx
sudo systemctl enable nfl-dashboard
sudo systemctl start nfl-dashboard
```

7. **Setup SSL with Let's Encrypt**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

**Your app is now live at:** `https://your-domain.com`

---

### Option 3: Docker + AWS ECS (Production-Grade)

1. **Create Dockerfile** (shown above)

2. **Build and push to ECR**
```bash
aws ecr create-repository --repository-name nfl-prediction-model

# Build image
docker build -t nfl-prediction-model .

# Tag and push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account-id.dkr.ecr.us-east-1.amazonaws.com
docker tag nfl-prediction-model:latest your-account-id.dkr.ecr.us-east-1.amazonaws.com/nfl-prediction-model:latest
docker push your-account-id.dkr.ecr.us-east-1.amazonaws.com/nfl-prediction-model:latest
```

3. **Create ECS Cluster and Task Definition** (via AWS Console or Terraform)

4. **Configure ALB with HTTPS**

---

## 💰 Cost Comparison

| Provider | Monthly Cost | Pros | Cons |
|----------|-------------|------|------|
| **Streamlit Cloud** | **FREE** | Zero config, auto-updates | Public only (or $200/mo for private) |
| **Railway** | $5-20 | Easy, auto-scaling | Limited free tier |
| **Render** | $7-25 | Simple, good support | Less control |
| **AWS EC2** | $10-50 | Full control, scalable | Requires DevOps knowledge |
| **AWS ECS/Fargate** | $20-100 | Auto-scaling, production-ready | More complex setup |
| **Heroku** | $7-25 | Simple deployment | Can be expensive at scale |

---

## 🎯 Recommendation for Your Project

**Start with Streamlit Community Cloud** because:
1. ✅ Free
2. ✅ Takes 5 minutes to deploy
3. ✅ Automatic HTTPS and security
4. ✅ Perfect for showcasing your model
5. ✅ Easy to share with others

**Upgrade to AWS if:**
- You need private/authenticated access
- Expect high traffic (>10,000 users/month)
- Want custom domain without Streamlit branding
- Need database integration
- Want advanced monitoring/analytics

---

## 📊 Current Resource Usage

Your app requires:
- **RAM:** ~500MB (model + data)
- **Storage:** ~2GB (data files)
- **CPU:** Minimal (predictions are fast)

**Streamlit Cloud FREE tier provides:**
- 1GB RAM ✅
- 1 CPU core ✅
- Enough for hundreds of concurrent users ✅

---

## 🔐 Security Best Practices Summary

1. ✅ Never commit secrets (.gitignore is set up)
2. ✅ Use HTTPS (automatic on Streamlit Cloud)
3. ⚠️ Add rate limiting (implement if deploying)
4. ⚠️ Add input validation (implement if deploying)
5. ⚠️ Add authentication (optional, for private access)
6. ✅ Log errors and usage
7. ✅ Regular backups (git already does this for code)
8. ⚠️ Monitor for abuse
9. ✅ Keep dependencies updated
10. ⚠️ Add CORS protection (for API)

---

## 📝 Next Steps

1. Deploy to Streamlit Cloud (5 minutes)
2. Test the live app
3. Share the URL
4. Monitor usage
5. Upgrade to AWS if needed

**Would you like me to:**
- Add rate limiting to the dashboard?
- Add authentication?
- Create a Dockerfile for AWS deployment?
- Set up monitoring/logging?

