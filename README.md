# Distributed Deep Learning for Smart Healthcare Diagnosis

## Project Overview
An intelligent healthcare diagnostic system that detects pneumonia from chest X-ray images using both classical machine learning and deep learning approaches, with distributed learning capabilities.

## Project Objective
Develop an automated diagnostic system to assist medical professionals by providing quick and reliable pneumonia predictions from chest X-ray images, while exploring distributed deep learning for scalability across healthcare institutions.

## Problem Statement
Manual analysis of chest X-ray images is time-consuming and requires expert radiologists. In remote healthcare settings, such expertise may not be available. This system provides automated analysis that is:
- Fast and accurate
- Scalable for large datasets
- Distributable across multiple healthcare institutions

## Technologies Used
- **Python 3.x**
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Deep Learning**: PyTorch, Torchvision
- **Visualization**: Matplotlib, Seaborn
- **Backend API**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Model Persistence**: Pickle, Joblib

## Dataset
- **Source**: Chest X-Ray Pneumonia Dataset (Kaggle)
- **Total Images**: 5,216
  - NORMAL: 1,341 images
  - PNEUMONIA: 3,875 images
- **Split**: Training, Validation, Testing sets

## Models Implemented

### Classical Machine Learning Models (Baseline)
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. K-Nearest Neighbors (KNN)
5. Naive Bayes

### Deep Learning Model (Advanced)
- Convolutional Neural Network (CNN)
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
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

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
| Logistic Regression | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD |
| Naive Bayes | TBD | TBD | TBD | TBD |
| CNN (Deep Learning) | TBD | TBD | TBD | TBD |

## Distributed Learning Approach
- **Current**: Data parallelism with PyTorch
- **Future**: Federated learning across multiple hospitals
- **Benefits**: Privacy-preserving, scalable, collaborative learning

## Team
[Your Name/Team Names]

## License
Educational Project

## References
- Chest X-Ray Images (Pneumonia) Dataset - Kaggle
- PyTorch Documentation
- Scikit-learn Documentation
