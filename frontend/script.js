// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeImageBtn = document.getElementById('removeImage');
const predictBtn = document.getElementById('predictBtn');
const modelSelect = document.getElementById('modelSelect');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsCard = document.getElementById('resultsCard');
const errorCard = document.getElementById('errorCard');
const errorMessage = document.getElementById('errorMessage');
const closeErrorBtn = document.getElementById('closeError');
const analyzeAnotherBtn = document.getElementById('analyzeAnother');

// State
let currentImageBase64 = null;

// Event Listeners
uploadArea.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', handleImageSelect);
removeImageBtn.addEventListener('click', resetUpload);
predictBtn.addEventListener('click', makePrediction);
closeErrorBtn.addEventListener('click', () => errorCard.classList.add('hidden'));
analyzeAnotherBtn.addEventListener('click', resetForNewAnalysis);

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    } else {
        showError('Please drop a valid image file');
    }
});

// Handle image selection
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Process selected file
function handleFile(file) {
    // Validate file size (5MB max)
    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size exceeds 5MB limit');
        return;
    }

    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    const reader = new FileReader();
    
    reader.onload = (e) => {
        currentImageBase64 = e.target.result;
        previewImg.src = currentImageBase64;
        
        // Show preview, hide upload area
        uploadArea.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        predictBtn.disabled = false;
        
        // Hide results and errors
        resultsCard.classList.add('hidden');
        errorCard.classList.add('hidden');
    };
    
    reader.onerror = () => {
        showError('Error reading file. Please try again.');
    };
    
    reader.readAsDataURL(file);
}

// Reset upload
function resetUpload() {
    currentImageBase64 = null;
    imageInput.value = '';
    uploadArea.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    predictBtn.disabled = true;
    resultsCard.classList.add('hidden');
}

// Reset for new analysis
function resetForNewAnalysis() {
    resetUpload();
    errorCard.classList.add('hidden');
}

// Make prediction
async function makePrediction() {
    if (!currentImageBase64) {
        showError('No image selected');
        return;
    }

    const selectedModel = modelSelect.value;
    
    // Show loading
    loadingIndicator.classList.remove('hidden');
    predictBtn.disabled = true;
    resultsCard.classList.add('hidden');
    errorCard.classList.add('hidden');

    try {
        // Send POST request to API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: currentImageBase64,
                model: selectedModel
            })
        });

        const data = await response.json();

        loadingIndicator.classList.add('hidden');

        if (response.ok && data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Prediction failed. Please try again.');
            predictBtn.disabled = false;
        }
    } catch (error) {
        loadingIndicator.classList.add('hidden');
        console.error('Error:', error);
        showError(`Connection error: ${error.message}. Make sure the backend server is running at ${API_BASE_URL}`);
        predictBtn.disabled = false;
    }
}

// Display results
function displayResults(data) {
    const predictionValue = document.getElementById('predictionValue');
    const confidenceValue = document.getElementById('confidenceValue');
    const confidenceProgress = document.getElementById('confidenceProgress');
    const modelUsed = document.getElementById('modelUsed');

    // Set prediction
    predictionValue.textContent = data.prediction;
    predictionValue.className = 'prediction-value ' + 
        (data.prediction === 'NORMAL' ? 'normal' : 'pneumonia');

    // Set confidence
    const confidencePercent = (data.confidence * 100).toFixed(1);
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceProgress.style.width = `${confidencePercent}%`;

    // Set model name
    const modelNames = {
        'random_forest_model': 'Random Forest',
        'logistic_regression_model': 'Logistic Regression',
        'decision_tree_model': 'Decision Tree',
        'k-nearest_neighbors_model': 'K-Nearest Neighbors',
        'naive_bayes_model': 'Naive Bayes',
        'cnn_model': 'CNN (Deep Learning)'
    };
    modelUsed.textContent = modelNames[data.model_used] || data.model_used;

    // Show results
    resultsCard.classList.remove('hidden');
    
    // Smooth scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show error
function showError(message) {
    errorMessage.textContent = message;
    errorCard.classList.remove('hidden');
    errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Check API health on page load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✓ API connection successful');
        } else {
            console.warn('⚠ API connection issue');
        }
    } catch (error) {
        console.error('❌ Cannot connect to API:', error);
        console.log('Make sure the backend server is running:');
        console.log('  1. cd backend');
        console.log('  2. python app.py');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAPIHealth();
    console.log('Pneumonia Detection System loaded');
});
