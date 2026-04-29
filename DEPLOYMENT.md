# Deployment Guide

## Render backend

1. Push changes to `main`.
2. In Render, confirm the service uses the repository root `requirements.txt` and the start command `gunicorn app:app --bind 0.0.0.0:$PORT`.
3. Wait for the deploy to finish, then verify:
   - `GET /health`
   - a test prediction using `random_forest_model`

Backend URL:
- `https://distributed-deep-learning-healthcare.onrender.com`

## Netlify frontend

1. Set the production environment variable `API_BASE_URL` to the backend URL above.
2. Build the frontend from `frontend/`.
3. Deploy `frontend/dist` to Netlify.
4. Verify `config.js` on the live site contains the Render backend URL.

Production URL:
- `https://distributed-deep-learning-healthcare.netlify.app`

## Model files

The deployed backend loads models from the repository `models/` directory.

Important files:
- `models/random_forest_model.pkl`
- `models/logistic_regression_model.pkl`
- `models/decision_tree_model.pkl`
- `models/k-nearest_neighbors_model.pkl`
- `models/naive_bayes_model.pkl`
- `models/feature_scaler.pkl`
- `models/label_encoder.pkl`
- `models/fast_resnet18_xray.pth`
- `models/cnn_model.pth`

## Quick verification

Use:

```bash
python scripts/test_predict.py
```

Expected result:
- `/health` returns `healthy`
- `/predict` returns a successful prediction
