# Setup and Installation Guide

## Prerequisites

- **Python**: Version 3.8 or higher
- **pip**: Python package installer
- **Git**: For version control (optional)
- **Web Browser**: Chrome, Firefox, or Edge (latest version)

## Step 1: Clone or Download the Project

```bash
# If using Git
git clone <repository-url>
cd "Distributed-deep learning-smart-health care"

# Or download and extract the ZIP file
```

## Step 2: Download the Dataset (Optional for Training)

For actual model training, download the Chest X-Ray Pneumonia Dataset:

1. Visit Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Download the dataset
3. Extract to `data/chest_xray/` folder

**Note**: The notebooks include synthetic data generation for demonstration if the dataset is not available.

## Step 3: Install Python Dependencies

### Option A: Install for Backend Only

```bash
cd backend
pip install -r requirements.txt
```

### Option B: Install Everything (Recommended)

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install additional packages for notebooks
pip install jupyter notebook matplotlib seaborn
```

### Dependencies Installed:
- Flask (Web framework)
- NumPy (Numerical computing)
- Pandas (Data manipulation)
- Scikit-learn (ML algorithms)
- PyTorch (Deep learning)
- Pillow (Image processing)
- Joblib (Model persistence)
- Flask-CORS (API security)

## Step 4: Run the Jupyter Notebooks (Training Phase)

```bash
# Navigate to project root
cd ..

# Launch Jupyter Notebook
jupyter notebook
```

### Execute notebooks in order:

1. **01_data_preprocessing.ipynb**
   - Loads and preprocesses dataset
   - Extracts features for classical ML
   - Saves processed data

2. **02_classical_ml_models.ipynb**
   - Trains all required classical ML models
   - Evaluates performance metrics
   - Saves models as .pkl files

3. **03_deep_learning_cnn.ipynb**
   - Builds and trains CNN model
   - Compares with classical models
   - Saves PyTorch model

**Important**: Execute all cells in each notebook sequentially. This will create trained models in the `models/` folder.

## Step 5: Start the Backend API Server

```bash
# Navigate to backend folder
cd backend

# Start Flask server
python app.py
```

You should see:
```
Starting Pneumonia Detection API Server
...
Server will start on http://localhost:5000
```

**Keep this terminal window open** - the server must run continuously.

### Verify API is Running:

Open a new terminal:
```bash
curl http://localhost:5000/health
```

Or visit in browser: http://localhost:5000

## Step 6: Launch the Frontend

### Option A: Direct File Opening
1. Navigate to `frontend/` folder
2. Double-click `index.html`
3. It will open in your default browser

### Option B: Local Web Server (Recommended)

```bash
# In a new terminal, navigate to frontend folder
cd frontend

# Start a simple HTTP server
python -m http.server 8000
```

Then open browser: http://localhost:8000

### Option C: VS Code Live Server

1. Install "Live Server" extension in VS Code
2. Right-click `index.html`
3. Select "Open with Live Server"

## Step 7: Test the System

1. **Upload a chest X-ray image** (or use a sample image)
2. **Select a model** from the dropdown
3. **Click "Analyze X-Ray"**
4. **View the prediction results**

## Troubleshooting

### Backend Server Issues

**Error: "Address already in use"**
```bash
# Kill process on port 5000
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:5000 | xargs kill -9
```

**Error: "Module not found"**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Error: "Models not found"**
- Make sure you've run all Jupyter notebooks to train and save models
- Check that `models/` folder contains .pkl and .pth files

### Frontend Issues

**Error: "Failed to fetch" or "CORS error"**
- Ensure backend server is running on port 5000
- Check that Flask-CORS is installed
- Verify the API_BASE_URL in `script.js` matches your backend URL

**Error: "Image not loading"**
- Check file size (must be < 5MB)
- Use JPG, JPEG, or PNG format
- Try a different image

### Model Training Issues

**Error: "Out of memory"**
- Reduce batch size in CNN notebook
- Use fewer training samples
- Close other applications

**Error: "CUDA not available"**
- This is normal if you don't have a GPU
- Models will train on CPU (slower but works)

## Testing with Postman

### 1. Install Postman
Download from: https://www.postman.com/downloads/

### 2. Test Health Endpoint
- **Method**: GET
- **URL**: http://localhost:5000/health
- **Expected Response**: 
```json
{
  "status": "healthy",
  "models_loaded": [...]
}
```

### 3. Test Prediction Endpoint
- **Method**: POST
- **URL**: http://localhost:5000/predict
- **Headers**: 
  - Content-Type: application/json
- **Body** (raw JSON):
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "model": "random_forest_model"
}
```

### 4. Get Postman Collection
A ready-to-use Postman collection is available in `backend/postman_collection.json`

## System Requirements

### Minimum:
- **CPU**: Dual-core processor
- **RAM**: 4 GB
- **Disk**: 2 GB free space
- **OS**: Windows 10, macOS 10.14, or Linux

### Recommended:
- **CPU**: Quad-core processor or better
- **RAM**: 8 GB or more
- **GPU**: NVIDIA GPU with CUDA support (for faster CNN training)
- **Disk**: 10 GB free space (for dataset)

## Next Steps

After successful setup:
1. ✅ Test with sample X-ray images
2. ✅ Compare different model predictions
3. ✅ Document your results
4. ✅ Prepare demo for presentation
5. ✅ Create architecture diagrams
6. ✅ Write project report

## Support

For issues:
1. Check error messages in terminal
2. Review this setup guide
3. Verify all dependencies are installed
4. Ensure models are trained and saved
5. Check that ports 5000 and 8000 are available

## Quick Start Commands (Summary)

```bash
# Terminal 1: Start Backend
cd backend
pip install -r requirements.txt
python app.py

# Terminal 2: Start Frontend  
cd frontend
python -m http.server 8000

# Browser: Open http://localhost:8000
```

That's it! Your pneumonia detection system should now be fully operational. 🎉
