"""
Quick System Verification Script
Tests all components before submission
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = Path(filepath).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_contents(dirpath, description, min_files=1):
    """Check if directory has minimum number of files"""
    path = Path(dirpath)
    if not path.exists():
        print(f"❌ {description}: {dirpath} (directory not found)")
        return False
    
    files = list(path.glob('*'))
    count = len(files)
    status = "✅" if count >= min_files else "❌"
    print(f"{status} {description}: {dirpath} ({count} files)")
    return count >= min_files

def main():
    print("\n" + "="*70)
    print("HACKATHON PROJECT VERIFICATION")
    print("="*70 + "\n")
    
    all_passed = True
    
    # 1. Check Notebooks
    print("📓 NOTEBOOKS (Training)")
    print("-" * 70)
    all_passed &= check_file_exists(
        'notebooks/01_data_preprocessing.ipynb',
        'Data preprocessing notebook'
    )
    all_passed &= check_file_exists(
        'notebooks/02_classical_ml_models.ipynb',
        'Classical ML models notebook (REQUIRED)'
    )
    all_passed &= check_file_exists(
        'notebooks/03_deep_learning_cnn.ipynb',
        'Deep learning CNN notebook'
    )
    print()
    
    # 2. Check Backend
    print("🖥 BACKEND (Flask API)")
    print("-" * 70)
    all_passed &= check_file_exists('backend/app.py', 'Flask API server')
    all_passed &= check_file_exists('backend/model_loader.py', 'Model loader')
    all_passed &= check_file_exists('backend/requirements.txt', 'Dependencies')
    check_file_exists('backend/postman_collection.json', 'Postman collection')
    print()
    
    # 3. Check Frontend
    print("🌐 FRONTEND (Demo UI)")
    print("-" * 70)
    all_passed &= check_file_exists('frontend/index.html', 'HTML interface')
    all_passed &= check_file_exists('frontend/style.css', 'CSS styling')
    all_passed &= check_file_exists('frontend/script.js', 'JavaScript logic')
    print()
    
    # 4. Check Models (CRITICAL)
    print("🤖 TRAINED MODELS (CRITICAL - Must exist!)")
    print("-" * 70)
    models_exist = check_directory_contents('models', 'Models directory', min_files=5)
    
    if models_exist:
        required_models = [
            'logistic_regression_model.pkl',
            'decision_tree_model.pkl',
            'random_forest_model.pkl',
            'k-nearest_neighbors_model.pkl',
            'naive_bayes_model.pkl',
            'feature_scaler.pkl',
            'label_encoder.pkl'
        ]
        
        for model_file in required_models:
            check_file_exists(f'models/{model_file}', f'  {model_file}')
    else:
        print("⚠️  CRITICAL: Models not found!")
        print("   ACTION: Run notebooks to train models:")
        print("   1. jupyter notebook")
        print("   2. Execute: 02_classical_ml_models.ipynb")
    print()
    
    # 5. Check Documentation
    print("📚 DOCUMENTATION")
    print("-" * 70)
    check_file_exists('README.md', 'Project README')
    check_file_exists('ARCHITECTURE.md', 'System architecture')
    check_file_exists('COMPLIANCE_CHECKLIST.md', 'Syllabus compliance')
    check_file_exists('TESTING_GUIDE.md', 'Postman testing guide')
    check_file_exists('SETUP_GUIDE.md', 'Setup instructions')
    check_file_exists('QUICKSTART_12HR.md', '12-hour quickstart guide')
    print()
    
    # 6. Check Artifacts
    print("📊 ARTIFACTS (Generated after training)")
    print("-" * 70)
    artifacts_exist = Path('artifacts').exists()
    if artifacts_exist:
        check_directory_contents('artifacts', 'Visualizations', min_files=1)
    else:
        print("⚠️  Artifacts folder empty (will be created after training)")
    print()
    
    # 7. Test Python imports
    print("🐍 PYTHON DEPENDENCIES")
    print("-" * 70)
    
    try:
        import flask
        print(f"✅ Flask: {flask.__version__}")
    except ImportError:
        print("❌ Flask not installed (required)")
        all_passed = False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not installed (required)")
        all_passed = False
    
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError:
        print("❌ Pandas not installed (required)")
        all_passed = False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError:
        print("❌ NumPy not installed (required)")
        all_passed = False
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("⚠️  PyTorch not installed (needed for CNN)")
    
    print()
    
    # Final Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    if models_exist and all_passed:
        print("✅ ALL SYSTEMS READY!")
        print("✅ Your project is ready for submission!")
        print("\nNext steps:")
        print("  1. Start backend: cd backend && python app.py")
        print("  2. Test with Postman")
        print("  3. Test frontend demo")
        print("  4. Take screenshots for report")
        print("  5. SUBMIT!")
    elif not models_exist:
        print("❌ CRITICAL ISSUE: Models not trained yet!")
        print("\n🚨 ACTION REQUIRED:")
        print("  1. Install dependencies: pip install -r backend/requirements.txt")
        print("  2. Launch Jupyter: jupyter notebook")
        print("  3. Run: 01_data_preprocessing.ipynb")
        print("  4. Run: 02_classical_ml_models.ipynb (MUST DO)")
        print("  5. Run: 03_deep_learning_cnn.ipynb (bonus)")
        print("  6. Run this script again to verify")
        print("\n⏱️  Estimated time: 2-3 hours")
    else:
        print("⚠️  Some components missing or not installed")
        print("   Review the checklist above and fix issues")
    
    print("="*70 + "\n")
    
    return 0 if models_exist else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
