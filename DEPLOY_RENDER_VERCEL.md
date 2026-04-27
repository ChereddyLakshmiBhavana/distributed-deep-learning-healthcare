# Render + Vercel deployment

This repo is split into:
- **Render** for the Flask API in `backend/`
- **Vercel** for the static frontend in `frontend/`

## 1. Deploy the backend on Render

1. Create a new **Web Service** on Render from this GitHub repo.
2. Use the repo root file [render.yaml](render.yaml), or set these values manually:

```yaml
rootDir: backend
buildCommand: pip install -r requirements.txt
startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
healthCheckPath: /health
```

3. Add any environment variables you need. The backend works with the included model files under `models/`.
4. After deploy, note the Render URL, for example:

```text
https://pneumonia-detection-api.onrender.com
```

## 2. Deploy the frontend on Vercel

1. Create a new Vercel project from the same GitHub repo.
2. Set the **Root Directory** to `frontend`.
3. Set the environment variable:

```text
API_BASE_URL=https://your-render-service.onrender.com
```

4. Vercel will run:

```text
npm run build
```

5. The build script writes `dist/config.js` with the backend URL and Vercel serves the files from `dist/`.

## 3. Verify the deployment

Open the Vercel frontend, upload a chest X-ray, and confirm the API calls reach the Render backend.

If you need a local test after changing `API_BASE_URL`, run:

```powershell
cd frontend
npm run build
```
