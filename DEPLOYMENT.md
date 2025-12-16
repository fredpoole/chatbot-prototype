# Deployment Guide for Temporary Server Testing

This guide provides several options for deploying your Flask chatbot to a temporary server for testing.

## Quick Options (Recommended)

### Option 1: Render (Easiest - Recommended)
**Free tier available, very easy setup**

1. Go to [render.com](https://render.com) and sign up (free)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository (or use manual deploy)
4. Configure:
   - **Name**: chatbot-prototype
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && python download_nltk_resources.py`
   - **Start Command**: `gunicorn server4:app`
   - **Environment Variables**: Add `OPENAI_API_KEY` (your actual key)
5. Click "Create Web Service"
6. Wait ~5 minutes for deployment
7. Your app will be live at: `https://your-app-name.onrender.com`

**Note**: You'll need to add `gunicorn` to requirements.txt for production.

---

### Option 2: Railway (Very Easy)
**Free tier with $5 credit/month**

1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Add environment variable: `OPENAI_API_KEY`
5. Railway auto-detects Python and runs your app
6. Your app will be live at: `https://your-app-name.up.railway.app`

**Note**: Railway auto-detects Flask apps, but you may need to add a `Procfile` if auto-detection fails.

---

### Option 3: Fly.io (Good for Temporary)
**Free tier available**

1. Install Fly CLI: `brew install flyctl` (or see [fly.io/docs](https://fly.io/docs))
2. Run: `flyctl launch`
3. Follow prompts (select Python, etc.)
4. Add secret: `flyctl secrets set OPENAI_API_KEY=your-key-here`
5. Deploy: `flyctl deploy`
6. Your app will be live at: `https://your-app-name.fly.dev`

---

### Option 4: Ngrok (Tunnel Local Server - Fastest for Quick Testing)
**Free, instant - just tunnels your local server**

1. Install ngrok: `brew install ngrok` (or download from [ngrok.com](https://ngrok.com))
2. Start your local server: `python server4.py`
3. In another terminal, run: `ngrok http 5000`
4. Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)
5. Share this URL - it tunnels to your local server
6. **Note**: URL changes each time you restart ngrok (unless you have a paid plan)

**Pros**: Instant, no deployment needed
**Cons**: URL changes, requires your local machine to be running

---

## Required Setup Steps

### 1. Update requirements.txt
Make sure `requirements.txt` includes:
```
Flask==3.0.0
requests==2.31.0
nltk==3.8.1
gunicorn==21.2.0
```

### 2. NLTK Data Download
The `download_nltk_resources.py` script should download required NLTK data. Make sure it runs during build.

### 3. Environment Variables
Set `OPENAI_API_KEY` in your deployment platform's environment variables section.

### 4. Port Configuration
The server now uses `PORT` environment variable (defaults to 5000 for local). Cloud platforms will set this automatically.

---

## Testing Your Deployment

Once deployed, visit:
- Render: `https://your-app-name.onrender.com`
- Railway: `https://your-app-name.up.railway.app`
- Fly.io: `https://your-app-name.fly.dev`
- Ngrok: `https://your-ngrok-url.ngrok.io`

You should see the chatbot interface. Click "Connect" to test the realtime voice feature.

---

## Troubleshooting

### NLTK Data Not Found
If you get NLTK errors, ensure `download_nltk_resources.py` runs during build. You may need to add it to the build command.

### Port Issues
The server now uses `PORT` env var. Most platforms set this automatically, but if issues occur, check platform docs.

### API Key Issues
Make sure `OPENAI_API_KEY` is set in your platform's environment variables (not in code).

---

## Recommendation

For quick testing: **Use Ngrok** (Option 4) - it's instant and requires no deployment.

For a proper temporary server: **Use Render** (Option 1) - it's free, easy, and gives you a stable URL.
