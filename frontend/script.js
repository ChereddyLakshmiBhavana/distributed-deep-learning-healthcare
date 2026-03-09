// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const removeImageBtn = document.getElementById('removeImage');
const predictBtn = document.getElementById('predictBtn');
const reportBtn = document.getElementById('reportBtn');
const explainBtn = document.getElementById('explainBtn');
const modelSelect = document.getElementById('modelSelect');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsCard = document.getElementById('resultsCard');
const reportCard = document.getElementById('reportCard');
const explainCard = document.getElementById('explainCard');
const errorCard = document.getElementById('errorCard');
const errorMessage = document.getElementById('errorMessage');
const closeErrorBtn = document.getElementById('closeError');
const analyzeAnotherBtn = document.getElementById('analyzeAnother');

// State
let currentImageBase64 = null;
let currentImageFile = null;

// Event Listeners
uploadArea.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', handleImageSelect);
removeImageBtn.addEventListener('click', resetUpload);
predictBtn.addEventListener('click', makePrediction);
reportBtn.addEventListener('click', generateReport);
explainBtn.addEventListener('click', generateExplanation);
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

    // Store the file object for later use
    currentImageFile = file;

    const reader = new FileReader();
    
    reader.onload = (e) => {
        currentImageBase64 = e.target.result;
        previewImg.src = currentImageBase64;
        
        // Show preview, hide upload area
        uploadArea.classList.add('hidden');
        imagePreview.classList.remove('hidden');
        predictBtn.disabled = false;
        reportBtn.disabled = false;
        explainBtn.disabled = false;
        
        // Hide results and errors
        resultsCard.classList.add('hidden');
        reportCard.classList.add('hidden');
        explainCard.classList.add('hidden');
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
    currentImageFile = null;
    imageInput.value = '';
    uploadArea.classList.remove('hidden');
    imagePreview.classList.add('hidden');
    predictBtn.disabled = true;
    reportBtn.disabled = true;
    explainBtn.disabled = true;
    resultsCard.classList.add('hidden');
    reportCard.classList.add('hidden');
    explainCard.classList.add('hidden');
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

// Generate PDF Report
async function generateReport() {
    if (!currentImageFile) {
        showError('No image selected');
        return;
    }

    const selectedModel = modelSelect.value;
    
    // Show loading
    loadingIndicator.classList.remove('hidden');
    reportBtn.disabled = true;
    reportCard.classList.add('hidden');
    errorCard.classList.add('hidden');

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', currentImageFile);
        formData.append('model', selectedModel);
        formData.append('include_all_models', 'true'); // Get comparison from all models

        // Send POST request to API
        const response = await fetch(`${API_BASE_URL}/predict/report`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        loadingIndicator.classList.add('hidden');

        if (response.ok && data.success) {
            displayReportResults(data);
        } else {
            showError(data.error || 'Report generation failed. Please try again.');
            reportBtn.disabled = false;
        }
    } catch (error) {
        loadingIndicator.classList.add('hidden');
        console.error('Error:', error);
        showError(`Connection error: ${error.message}. Make sure the backend server is running.`);
        reportBtn.disabled = false;
    }
}

// Display report results
function displayReportResults(data) {
    document.getElementById('reportPrediction').textContent = data.prediction;
    document.getElementById('reportConfidence').textContent = (data.confidence * 100).toFixed(1) + '%';
    
    // Set up download button
    const downloadBtn = document.getElementById('downloadReportBtn');
    downloadBtn.href = `${API_BASE_URL}${data.download_url}`;
    downloadBtn.download = data.report_pdf;
    
    // Display model comparison if available
    if (data.all_models && data.all_models.length > 0) {
        const comparisonDiv = document.getElementById('modelComparison');
        comparisonDiv.innerHTML = '<h3>All Models Comparison</h3>';
        
        const table = document.createElement('table');
        table.style.cssText = 'width: 100%; border-collapse: collapse; margin-top: 10px;';
        table.innerHTML = `
            <thead>
                <tr style="background: #2c5aa0; color: white;">
                    <th style="padding: 10px; text-align: left;">Model</th>
                    <th style="padding: 10px; text-align: center;">Prediction</th>
                    <th style="padding: 10px; text-align: center;">Confidence</th>
                </tr>
            </thead>
            <tbody>
                ${data.all_models.map(model => `
                    <tr style="border-bottom: 1px solid #ddd;">
                        <td style="padding: 10px;">${model.model}</td>
                        <td style="padding: 10px; text-align: center; font-weight: bold; color: ${model.prediction === 'NORMAL' ? '#28a745' : '#dc3545'};">
                            ${model.prediction}
                        </td>
                        <td style="padding: 10px; text-align: center;">
                            ${(model.confidence * 100).toFixed(1)}%
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        `;
        comparisonDiv.appendChild(table);
    }
    
    // Show report card
    reportCard.classList.remove('hidden');
    reportCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Generate SHAP Explanation
async function generateExplanation() {
    if (!currentImageFile) {
        showError('No image selected');
        return;
    }

    const selectedModel = modelSelect.value;
    
    // Show loading
    loadingIndicator.classList.remove('hidden');
    explainBtn.disabled = true;
    explainCard.classList.add('hidden');
    errorCard.classList.add('hidden');

    try {
        // Create FormData
        const formData = new FormData();
        formData.append('file', currentImageFile);
        formData.append('model', selectedModel);

        // Send POST request to API
        const response = await fetch(`${API_BASE_URL}/predict/explain`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        loadingIndicator.classList.add('hidden');

        if (response.ok && data.success) {
            displayExplanationResults(data);
        } else {
            showError(data.error || 'Explanation generation failed. Please try again.');
            explainBtn.disabled = false;
        }
    } catch (error) {
        loadingIndicator.classList.add('hidden');
        console.error('Error:', error);
        showError(`Connection error: ${error.message}. Make sure the backend server is running.`);
        explainBtn.disabled = false;
    }
}

// Display explanation results
function displayExplanationResults(data) {
    // Display top features
    const topFeaturesDiv = document.getElementById('topFeatures');
    topFeaturesDiv.innerHTML = '<h3>Top Contributing Features</h3>';
    
    const featuresList = document.createElement('ol');
    featuresList.style.cssText = 'line-height: 1.8;';
    
    data.top_features.forEach(feature => {
        const li = document.createElement('li');
        const directionColor = feature.direction === 'PNEUMONIA' ? '#dc3545' : '#28a745';
        li.innerHTML = `
            <strong>${feature.feature}</strong>: ${feature.description}<br>
            <span style="color: ${directionColor}; font-size: 0.9em;">
                Impact: ${(feature.importance * 100).toFixed(2)}% → Points toward ${feature.direction}
            </span>
        `;
        li.style.marginBottom = '10px';
        featuresList.appendChild(li);
    });
    
    topFeaturesDiv.appendChild(featuresList);
    
    // Display visualizations
    const vizDiv = document.getElementById('shapVisualizations');
    vizDiv.innerHTML = '';
    
    if (data.visualizations) {
        Object.entries(data.visualizations).forEach(([vizType, vizData]) => {
            if (vizData && vizData.download_url) {
                const vizContainer = document.createElement('div');
                vizContainer.style.cssText = 'border: 1px solid #ddd; padding: 10px; border-radius: 5px;';
                
                const title = document.createElement('h4');
                title.textContent = vizType.charAt(0).toUpperCase() + vizType.slice(1) + ' Plot';
                title.style.marginTop = '0';
                
                const img = document.createElement('img');
                img.src = `${API_BASE_URL}${vizData.download_url}`;
                img.alt = `${vizType} visualization`;
                img.style.cssText = 'width: 100%; max-width: 100%; height: auto; border-radius: 5px;';
                
                const downloadLink = document.createElement('a');
                downloadLink.href = `${API_BASE_URL}${vizData.download_url}`;
                downloadLink.textContent = 'Download Image';
                downloadLink.download = vizData.filename;
                downloadLink.style.cssText = 'display: inline-block; margin-top: 10px; color: #2c5aa0; text-decoration: none;';
                
                vizContainer.appendChild(title);
                vizContainer.appendChild(img);
                vizContainer.appendChild(downloadLink);
                vizDiv.appendChild(vizContainer);
            }
        });
    }
    
    // Display explanation text
    const explanationTextDiv = document.getElementById('explanationText');
    explanationTextDiv.textContent = data.explanation_text;
    
    // Show explanation card
    explainCard.classList.remove('hidden');
    explainCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
