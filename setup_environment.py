#!/usr/bin/env python3
"""
GPU-Accelerated Churn Prediction Environment Setup
================================================
This script automatically:
1. Checks virtual environment
2. Installs GPU packages
3. Creates project structure
4. Tests GPU status
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_virtual_environment():
    """Check virtual environment status"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        print("✅ Virtual environment active")
        print(f"   Path: {sys.prefix}")
        return True
    else:
        print("⚠️ Virtual environment not active!")
        print("\n🔧 To create virtual environment:")
        print("   python -m venv churn_env")
        if platform.system() == "Windows":
            print("   churn_env\\Scripts\\activate")
        else:
            print("   source churn_env/bin/activate")
        print("\n❓ Continue anyway? (y/N): ", end="")
        
        choice = input().lower()
        return choice in ['y', 'yes']

def install_base_packages():
    """Install base packages"""
    print("\n📦 Installing base packages...")
    
    base_packages = [
        "pip>=23.0.0",
        "wheel",
        "setuptools",
        "pandas>=2.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.3.0",
        "imbalanced-learn>=0.12.0",
        "joblib>=1.3.0",
        "mlflow>=2.7.0",
        "streamlit>=1.26.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "pyyaml>=6.0"
    ]
    
    try:
        # Pip upgrade
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Base packages
        for package in base_packages:
            print(f"   🔄 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
        
        print("✅ Base packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Base package installation error: {e}")
        return False

def install_gpu_packages():
    """Install GPU packages"""
    print("\n🎮 Installing GPU packages...")
    
    # 1. PyTorch GPU
    print("   🔥 Installing PyTorch GPU...")
    try:
        torch_cmd = [
            sys.executable, "-m", "pip", "install",
            "torch>=2.0.0", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        subprocess.check_call(torch_cmd)
        print("   ✅ PyTorch GPU installed")
    except subprocess.CalledProcessError:
        print("   ⚠️ PyTorch GPU installation failed, trying CPU version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    
    # 2. XGBoost GPU
    print("   🚀 Installing XGBoost GPU...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost[gpu]>=1.7.0"])
        print("   ✅ XGBoost GPU installed")
    except subprocess.CalledProcessError:
        print("   ⚠️ XGBoost GPU failed, installing CPU version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost>=1.7.0"])
    
    # 3. RAPIDS (optional)
    if platform.system() == "Linux":
        print("   🌊 Installing RAPIDS (optional)...")
        try:
            rapids_cmd = [
                sys.executable, "-m", "pip", "install",
                "cudf-cu12>=23.10.0", "cuml-cu12>=23.10.0",
                "--extra-index-url=https://pypi.nvidia.com"
            ]
            subprocess.check_call(rapids_cmd)
            print("   ✅ RAPIDS installed successfully")
        except subprocess.CalledProcessError:
            print("   ⚠️ RAPIDS installation failed (optional)")
    else:
        print("   ⚠️ RAPIDS not supported on Windows (use WSL2)")
    
    return True

def test_gpu_environment():
    """Test GPU environment"""
    print("\n🔍 Testing GPU environment...")
    
    # Test CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # Quick GPU test
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            y = torch.mm(x, x.t())
            print("✅ GPU computation test passed")
            del x, y
            torch.cuda.empty_cache()
        else:
            print("❌ CUDA not available")
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ GPU test error: {e}")
    
    # Test XGBoost GPU
    try:
        import xgboost as xgb
        print("✅ XGBoost installed")
        try:
            clf = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=10)
            print("✅ XGBoost GPU support available")
        except:
            print("⚠️ XGBoost GPU not available, CPU fallback")
    except ImportError:
        print("❌ XGBoost not installed")
    
    # Test RAPIDS
    try:
        import cuml
        print("✅ RAPIDS cuML available")
    except ImportError:
        print("⚠️ RAPIDS cuML not available (optional)")
    
    try:
        import cudf
        print("✅ RAPIDS cuDF available")
    except ImportError:
        print("⚠️ RAPIDS cuDF not available (optional)")

def create_project_structure():
    """Create project directory structure"""
    print("\n🏗️ Creating project structure...")
    
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "src",
        "mlruns"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}/")

def create_sample_data():
    """Create sample churn dataset if not exists"""
    data_path = Path("data/raw/churn.csv")
    
    if data_path.exists():
        print(f"✅ Data file exists: {data_path}")
        return
    
    print("📊 Creating sample churn dataset...")
    
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n_samples = 7043  # Typical churn dataset size
    
    # Generate realistic churn data
    data = {
        'customerID': [f'customer_{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples, p=[0.33, 0.15, 0.22, 0.30]),
        'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples),
        'TotalCharges': np.random.uniform(18.0, 8685.0, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])  # Realistic churn rate
    }
    
    df = pd.DataFrame(data)
    df['MonthlyCharges'] = df['MonthlyCharges'].round(2)
    df['TotalCharges'] = df['TotalCharges'].round(2)
    
    # Save dataset
    df.to_csv(data_path, index=False)
    print(f"✅ Sample dataset created: {data_path}")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📈 Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")

def main():
    """Main setup function"""
    print("🚀 GPU-Accelerated Churn Prediction Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check virtual environment
    if not check_virtual_environment():
        return
    
    # Install packages
    if not install_base_packages():
        return
    
    install_gpu_packages()
    
    # Create project structure
    create_project_structure()
    
    # Create sample data
    create_sample_data()
    
    # Test GPU environment
    test_gpu_environment()
    
    print("\n" + "=" * 60)
    print("✅ Environment setup completed successfully!")
    print("=" * 60)
    
    print("\n🎯 Next steps:")
    print("1. Run model comparison: python mlflow_model_comparison.py")
    print("2. View results: mlflow ui")
    print("3. Launch dashboard: streamlit run enhanced_streamlit_app.py")

if __name__ == "__main__":
    main()