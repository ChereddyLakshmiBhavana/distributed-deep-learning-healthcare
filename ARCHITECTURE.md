# System Architecture

## Overview
The Pneumonia Detection System follows a **three-tier architecture**:
1. **Frontend (Presentation Layer)**: HTML/CSS/JavaScript user interface
2. **Backend (Application Layer)**: Flask REST API for model serving
3. **Model Layer**: ML/DL models for pneumonia diagnosis

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         USER                                 │
│                  (Medical Professional)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ uploads X-ray image
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND LAYER                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  HTML Form + CSS Styling + JavaScript Fetch API       │  │
│  │  - Image upload and preview                           │  │
│  │  - Model selection dropdown                           │  │
│  │  - Result display with confidence                     │  │
│  │  - Error handling                                     │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP POST JSON Request
                       │ {image: base64, model: "random_forest"}
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND API LAYER                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Flask REST API                           │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Endpoints:                                      │  │  │
│  │  │  - GET  /         (API info)                     │  │  │
│  │  │  - GET  /health   (health check)                 │  │  │
│  │  │  - GET  /models   (list models)                  │  │  │
│  │  │  - POST /predict  (main prediction)              │  │  │
│  │  │  - POST /predict/batch (batch prediction)        │  │  │
│  │  │  - GET  /test     (Postman testing)              │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  │                                                         │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  Request Processing:                             │  │  │
│  │  │  1. Receive base64 image                         │  │  │
│  │  │  2. Decode and preprocess                        │  │  │
│  │  │  3. Extract features OR prepare tensor           │  │  │
│  │  │  4. Load selected model                          │  │  │
│  │  │  5. Make prediction                              │  │  │
│  │  │  6. Return JSON response                         │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ loads models via Pickle/Joblib
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Classical ML Models (.pkl)                   │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │ - Logistic Regression                            │  │  │
│  │  │ - Decision Tree                                  │  │  │
│  │  │ - Random Forest ⭐ (Best performer)             │  │  │
│  │  │ - K-Nearest Neighbors                            │  │  │
│  │  │ - Naive Bayes                                    │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │        Deep Learning Model (.pth)                      │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │ Convolutional Neural Network (PyTorch)           │  │  │
│  │  │ - 3 Conv blocks with MaxPooling                  │  │  │
│  │  │ - 2 Fully connected layers                       │  │  │
│  │  │ - Dropout for regularization                     │  │  │
│  │  │ - Data Parallelism support                       │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         Preprocessing Components                       │  │
│  │  - Feature Scaler (StandardScaler)                    │  │
│  │  - Label Encoder (NORMAL/PNEUMONIA)                   │  │
│  │  - Image transformations (Resize, Normalize)          │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ returns prediction + confidence
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     RESPONSE                                 │
│  {                                                           │
│    "success": true,                                          │
│    "prediction": "PNEUMONIA" / "NORMAL",                     │
│    "confidence": 0.87,                                       │
│    "model_used": "random_forest_model"                       │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **User Action**: Medical professional uploads chest X-ray image via web interface
2. **Frontend Processing**: 
   - JavaScript converts image to base64
   - Sends POST request to `/predict` endpoint with JSON payload
3. **Backend Processing**:
   - Flask receives request
   - Decodes base64 image
   - Extracts features (for classical ML) or preprocesses tensor (for CNN)
   - Loads selected model from disk (Pickle/PyTorch)
   - Makes prediction
   - Returns JSON response
4. **Result Display**: Frontend displays prediction with confidence level

## Technology Stack

### Frontend
- **HTML5**: Structure and forms
- **CSS3**: Styling and responsive design
- **JavaScript (Vanilla)**: DOM manipulation, Fetch API for AJAX
- **Features**: Drag-and-drop, image preview, error handling

### Backend
- **Flask**: Python web framework for REST API
- **Flask-CORS**: Cross-origin resource sharing
- **Gunicorn**: Production WSGI server (optional)

### Machine Learning
- **Scikit-learn**: Classical ML algorithms and preprocessing
- **PyTorch**: Deep learning framework for CNN
- **NumPy/Pandas**: Data manipulation
- **Pillow**: Image processing
- **Joblib**: Model serialization

### Model Persistence
- **Pickle/Joblib**: For classical ML models (.pkl)
- **PyTorch state_dict**: For deep learning models (.pth)

## Distributed Learning Preparation

Current implementation includes:
- **Data Parallelism**: `nn.DataParallel` for multi-GPU training
- **Model Modularization**: Separate training and inference code
- **Scalable Architecture**: API-based serving for horizontal scaling

Future enhancements:
- **Federated Learning**: Train across multiple hospitals without sharing raw data
- **Model Aggregation**: FedAvg algorithm for weight combination
- **Privacy Preservation**: Differential privacy techniques

## Deployment Architecture

### Development
```
Frontend: File system (open index.html)
Backend: Flask dev server (localhost:5000)
Models: Local disk storage
```

### Production (Recommended)
```
Frontend: Static hosting (Netlify, Vercel, GitHub Pages)
Backend: Cloud deployment (AWS, Heroku, Azure)
Models: Cloud storage (S3, Azure Blob)
Database: Model metadata and prediction logs (PostgreSQL/MongoDB)
```

## Security Considerations

- Input validation for image uploads (size, type)
- Error handling to prevent information leakage
- CORS configuration for trusted domains only
- Model file integrity checks
- Rate limiting for API endpoints (recommended for production)

## Performance Metrics

| Model | Inference Time | Memory Usage |
|-------|---------------|--------------|
| Random Forest | ~10ms | Low |
| Logistic Regression | ~5ms | Very Low |
| CNN | ~50-100ms | Medium-High |

## Scalability

- **Horizontal Scaling**: Deploy multiple API instances with load balancer
- **Caching**: Cache frequently used models in memory
- **Async Processing**: Use Celery for batch predictions
- **Distributed Training**: Scale training across multiple nodes
