# Distributed Deep Learning for Smart Healthcare Diagnosis

## Project Overview
An intelligent healthcare diagnostic system that detects pneumonia from chest X-ray images using both classical machine learning and deep learning approaches, with distributed learning capabilities.

## Project Objective
Develop an automated diagnostic system to assist medical professionals by providing quick and reliable pneumonia predictions from chest X-ray images, while exploring distributed deep learning for scalability across healthcare institutions.

## Problem Statement
Manual analysis of chest X-ray images is time-consuming and requires expert radiologists. In remote healthcare settings, such expertise may not be available. This system provides automated analysis that is:

## Technologies Used

## Dataset
  - NORMAL: 1,341 images
  - PNEUMONIA: 3,875 images

## Models Implemented

### Classical Machine Learning Models (Baseline)
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Naive Bayes

### Deep Learning Model (Advanced)
  - Custom architecture with Conv2D, MaxPooling, Dropout
  - Adam optimizer
  - Cross-Entropy Loss

## Project Structure
```
Distributed-deep learning-smart-health care/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_classical_ml_models.ipynb
│   └── 03_deep_learning_cnn.ipynb
├── backend/
│   ├── app.py
│   ├── model_loader.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── cnn_model.pth
├── data/
│   └── (dataset files)
├── artifacts/
│   ├── architecture_diagram.png
│   └── evaluation_results.png
└── README.md
```

## Evaluation Metrics

## Installation & Setup

### Prerequisites
```bash
pip install -r backend/requirements.txt
```

### Run Backend API
```bash
cd backend
python app.py
```

### Access Frontend
Open `frontend/index.html` in a web browser or serve via:
```bash
python -m http.server 8000
```

## API Endpoints

### POST /predict
Predicts pneumonia from uploaded X-ray image.

**Request:**
```json
{
  "image": "base64_encoded_image",
  "model": "random_forest" 
}
```

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.87,
  "model_used": "random_forest"
}
```

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 77.40% | 78.62% | 87.69% | 82.91% |
| Decision Tree | 73.56% | 72.91% | 91.79% | 81.27% |
| Random Forest | 72.28% | 70.83% | 94.62% | 81.01% |
| KNN | 76.60% | 75.42% | 92.82% | 83.22% |
| Naive Bayes | 71.79% | 79.72% | 73.59% | 76.53% |
| CNN (Deep Learning) | See `artifacts/` for the latest run | See `artifacts/` for the latest run | See `artifacts/` for the latest run | See `artifacts/` for the latest run |

## Distributed Learning Approach

## Distributed Engineering Stack

The project now includes a practical distributed-engineering layer in the backend:


Container support is also included with `docker-compose.yml` so backend and frontend can run as services.
Run `worker` service as well to process queued distributed jobs.

See `DISTRIBUTED_ENGINEERING.md` for end-to-end commands.

## Deployment Options

If you do not want to use Docker locally, you can deploy:
- the Flask backend to Render using [render.yaml](render.yaml)
- the static frontend to Vercel using the files under `frontend/`

See [DEPLOY_RENDER_VERCEL.md](DEPLOY_RENDER_VERCEL.md) for the exact Render/Vercel setup.

## Team
[Your Name/Team Names]

## License
Educational Project

## References

[![Smoke Test](https://github.com/ChereddyLakshmiBhavana/distributed-deep-learning-healthcare/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/ChereddyLakshmiBhavana/distributed-deep-learning-healthcare/actions/workflows/smoke-test.yml)
