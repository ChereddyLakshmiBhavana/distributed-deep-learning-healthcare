# Deployment instructions

1. Copy `.env.template` to `.env` and fill secrets.

```powershell
copy .env.template .env
# edit .env
```

2. Build and start containers:

```powershell
docker compose up --build -d
```

3. Run smoke tests (local / CI):

```powershell
python scripts/ci_smoke_test.py
```

4. For production, ensure:
- Persistent volumes for `prediction_logs`, `reports`, `explanations`, `distributed_state` are backed by host volumes or managed storage.
- `FLASK_DEBUG=0` in `.env` and a strong `SECRET_KEY`.
- `git lfs install` and move large artifacts to LFS or external artifact storage.
