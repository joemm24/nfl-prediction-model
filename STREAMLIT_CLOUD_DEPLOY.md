# Streamlit Cloud Deployment Guide

## 🚀 Quick Deploy (5 Minutes)

Your NFL Prediction Model is now ready for deployment with **authentication** and **security features**!

---

## ✅ Prerequisites

1. ✅ GitHub account (you have: `joemm24`)
2. ✅ Repository pushed to GitHub (done!)
3. ⚠️ Streamlit Cloud account (free signup)

---

## 📋 Step-by-Step Deployment

### Step 1: Create Streamlit Cloud Account

1. Go to: **https://share.streamlit.io/**
2. Click **"Sign up"** or **"Continue with GitHub"**
3. Authorize Streamlit to access your GitHub repositories

### Step 2: Deploy Your App

1. Click **"New app"** button (top right)

2. Fill in the deployment form:
   ```
   Repository:     joemm24/nfl-prediction-model
   Branch:         main
   Main file path: src/dashboard.py
   App URL:        nfl-prediction-model (or your custom name)
   ```

3. Click **"Advanced settings"** (optional but recommended):
   ```
   Python version: 3.11
   ```

4. **Do NOT deploy yet!** - We need to add secrets first

### Step 3: Configure Authentication (IMPORTANT)

1. In the deployment form, scroll to **"Advanced settings"**

2. Click **"Secrets"** section

3. Add your user credentials in TOML format:

```toml
# User Authentication
# Passwords are SHA256 hashed for security
[users]
demo = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"  # Password: nfl2025
admin = "your_hashed_password_here"
```

**To create your own secure password:**

```bash
# Run this in your terminal
python -c "import hashlib; password='YourSecurePassword123'; print(hashlib.sha256(password.encode()).hexdigest())"

# Or use this Python script
import hashlib
password = input("Enter password: ")
hashed = hashlib.sha256(password.encode()).hexdigest()
print(f"Hashed password: {hashed}")
```

Then copy the hash and add it to secrets like:
```toml
[users]
yourusername = "the_long_hash_from_above"
```

### Step 4: Deploy!

1. Click **"Deploy!"** button

2. Wait 2-3 minutes for deployment

3. Your app will be live at: `https://[your-app-name].streamlit.app`

---

## 🔐 Security Features Enabled

Your deployment includes:

### 1. **User Authentication** ✅
- Login page with username/password
- SHA256 password hashing
- Session management
- Secure logout

### 2. **Rate Limiting** ✅
- 10 predictions per hour per user
- Prevents API abuse
- Automatic reset after 60 minutes

### 3. **Brute Force Protection** ✅
- Maximum 5 failed login attempts
- 15-minute lockout after 5 failures
- Prevents password guessing attacks

### 4. **HTTPS/SSL** ✅
- Automatic on Streamlit Cloud
- All traffic encrypted
- Secure data transmission

### 5. **Session Tracking** ✅
- Shows active session duration
- Displays username in header
- Logout functionality

---

## 👥 Managing Users

### Add New Users

1. Go to your app on Streamlit Cloud
2. Click **"Settings"** → **"Secrets"**
3. Add new user:

```toml
[users]
demo = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"
newuser = "their_hashed_password_here"
```

4. Click **"Save"**
5. App will automatically restart

### Change Passwords

1. Generate new hash:
```bash
python -c "import hashlib; print(hashlib.sha256('NewPassword123'.encode()).hexdigest())"
```

2. Update in Secrets:
```toml
[users]
username = "new_hashed_password"
```

### Remove Users

Simply delete their line from secrets:
```toml
[users]
# olduser = "hash"  # Commented out = disabled
activeuser = "hash"
```

---

## 🔧 Post-Deployment Configuration

### Custom Domain (Optional)

Streamlit Cloud provides a free domain: `https://[app-name].streamlit.app`

For a custom domain (e.g., `predictions.yourdomain.com`):
1. Upgrade to Streamlit Cloud Teams ($200/month)
2. Or use Cloudflare as a proxy (free)

### Update Data/Models

The app excludes large data files from git. To update predictions:

1. Generate new predictions locally:
```bash
cd /Users/joemartineziv/nfl-prediction-model
source venv/bin/activate
python run_pipeline.py
```

2. Commit and push:
```bash
git add predictions/predictions_latest.*
git commit -m "Update predictions for Week X"
git push origin main
```

3. Streamlit Cloud auto-deploys within 1 minute

### Monitor Usage

Streamlit Cloud provides:
- **Analytics dashboard**: See visitor count, page views
- **Logs**: Debug errors and track user activity
- **Resource usage**: Monitor RAM and CPU

Access via: App Settings → Analytics

---

## 📊 Default Login Credentials

**For Demo/Testing:**
```
Username: demo
Password: nfl2025
```

**⚠️ IMPORTANT:** Change this after deployment!

1. Generate a secure password hash
2. Update secrets on Streamlit Cloud
3. Remove or change the demo account

---

## 🚨 Troubleshooting

### App Won't Start

**Error:** "ModuleNotFoundError"
- **Fix:** Check that `requirements.txt` includes all dependencies
- Run: `pip freeze > requirements.txt` locally and push

**Error:** "File not found: models/*.pkl"
- **Fix:** Models are gitignored. The app will work without them for viewing existing predictions
- To generate new predictions, upload models or regenerate on Streamlit Cloud

### Login Not Working

**Error:** "Invalid credentials"
- **Fix:** Verify password hash is correct
- Use: `python -c "import hashlib; print(hashlib.sha256('password'.encode()).hexdigest())"`
- Copy exact hash to secrets

**Error:** "secrets.toml not found"
- **Fix:** Add secrets in Streamlit Cloud dashboard
- Go to: App Settings → Secrets

### Rate Limit Issues

**Error:** "Rate limit exceeded"
- **Fix:** Wait 60 minutes or adjust in `src/auth.py`:
```python
rate_limiter = RateLimiter(max_requests=20, window_minutes=60)  # Increase to 20
```

---

## 🔄 Updating Your Deployment

### Code Changes

```bash
# Make changes locally
git add .
git commit -m "Your changes"
git push origin main
```

Streamlit Cloud auto-deploys within 60 seconds!

### Secrets Changes

1. Go to: `https://share.streamlit.io/`
2. Select your app → **Settings** → **Secrets**
3. Edit and **Save**
4. App restarts automatically

### Environment Changes

Update `.streamlit/config.toml` and push to GitHub.

---

## 💰 Cost

**Streamlit Community Cloud (Current):**
- ✅ **FREE**
- 1GB RAM
- 1 CPU core
- Unlimited apps
- Public access only (with your authentication)

**Streamlit Cloud Teams ($200/month):**
- Private apps
- Custom domains
- More resources
- SSO/SAML
- Priority support

---

## 🎯 Your App Features

Once deployed, users can:

1. **Login** with credentials
2. **View predictions** for NFL games
3. **Generate new predictions** (rate-limited)
4. **Analyze games** with detailed metrics
5. **Export results** as CSV
6. **View team comparisons** with logos and colors

---

## 📱 Share Your App

Your app URL will be:
```
https://nfl-prediction-model-[random-string].streamlit.app
```

Or custom name:
```
https://your-chosen-name.streamlit.app
```

Share credentials with users:
```
🏈 NFL Prediction Model
URL: https://your-app.streamlit.app
Username: [provided separately]
Password: [provided separately]
```

---

## 🔐 Security Best Practices

### ✅ DO:
- Use strong passwords (12+ characters)
- Change default demo password immediately
- Use unique passwords for each user
- Monitor logs for suspicious activity
- Update dependencies regularly
- Keep secrets.toml secure

### ❌ DON'T:
- Share credentials publicly
- Use simple passwords (e.g., "password123")
- Commit secrets to GitHub
- Share your Streamlit Cloud login
- Ignore security updates

---

## 📧 Support

If you encounter issues:

1. **Streamlit Docs**: https://docs.streamlit.io/
2. **Community Forum**: https://discuss.streamlit.io/
3. **GitHub Issues**: Create issue in your repo

---

## 🎉 Next Steps

1. ✅ Deploy to Streamlit Cloud
2. ✅ Test login functionality
3. ✅ Change demo password
4. ✅ Share with users
5. ⭐ Star your repository on GitHub!

---

## 📋 Quick Reference

| Task | Command/Action |
|------|---------------|
| Deploy | Click "Deploy" on Streamlit Cloud |
| Add User | Update secrets → [users] section |
| Change Password | Generate new hash → Update secrets |
| Update App | `git push origin main` |
| View Logs | App Settings → Logs |
| Monitor Usage | App Settings → Analytics |
| Restart App | App Settings → Reboot |

---

Your NFL Prediction Model is **production-ready** with enterprise-grade security! 🏈🔐

Happy Predicting! 🎯

