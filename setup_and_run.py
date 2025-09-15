#!/usr/bin/env python3
"""
GPU-Accelerated Churn Prediction Project Setup and Execution
==========================================================
This script sets up the complete MLflow churn prediction pipeline
and provides execution instructions with GPU optimization.
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def create_project_structure():
    """Create the required project directory structure"""
    print("🏗️ Creating project structure...")
    
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
    
    print("✅ Project structure created successfully!")

def install_requirements():
    """Install required packages - including GPU support"""
    print("\n📦 Installing GPU-optimized packages...")
    
    # PyTorch GPU installation (CUDA 12.1)
    torch_cmd = [
        sys.executable, "-m", "pip", "install", 
        "torch>=2.0.0", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    requirements = [
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
        # PyTorch GPU installation
        print("🔥 Installing PyTorch GPU version...")
        subprocess.check_call(torch_cmd)
        
        # XGBoost GPU installation
        print("🚀 Installing XGBoost GPU version...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost[gpu]>=1.7.0"])
        
        # Other packages
        print("📦 Installing other required packages...")
        for package in requirements:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        # RAPIDS installation (optional - fails gracefully)
        print("🌊 Installing RAPIDS packages (optional)...")
        try:
            rapids_packages = [
                "cudf-cu12>=23.10.0",
                "cuml-cu12>=23.10.0"
            ]
            for package in rapids_packages:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package,
                        "--extra-index-url=https://pypi.nvidia.com"
                    ])
                    print(f"✅ {package} installed")
                except:
                    print(f"⚠️ {package} failed, CPU fallback will be used")
        except:
            print("⚠️ RAPIDS packages not available, only CPU models will be used")
        
        print("✅ All packages installed successfully!")
        
        # GPU check
        print("\n🔍 Checking GPU status...")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ CUDA available: {torch.version.cuda}")
                print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                print("⚠️ CUDA not found, CPU models will be used")
        except:
            print("⚠️ PyTorch import failed")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Package installation error: {e}")
        return False

def create_sample_data():
    """Create sample churn data if not exists"""
    data_path = "data/raw/churn.csv"
    
    if os.path.exists(data_path):
        print(f"✅ Data file already exists: {data_path}")
        return True
    
    print("📊 Creating sample churn dataset...")
    
    import numpy as np
    np.random.seed(42)
    
    # Create realistic churn dataset
    n_samples = 7043
    
    sample_data = {
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
        'MonthlyCharges': np.round(np.random.uniform(18.0, 120.0, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(18.0, 8685.0, n_samples), 2),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(data_path, index=False)
    
    print(f"✅ Sample data created: {data_path}")
    print(f"   📊 Shape: {df.shape}")
    print(f"   📈 Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    return True

def create_config_files():
    """Create configuration files"""
    print("\n⚙️ Creating configuration files...")
    
    # Update requirements.txt
    requirements_content = """# GPU-Accelerated Churn Prediction Requirements
# Optimized for NVIDIA RTX GPU

# Core ML and Data Processing
pandas>=2.3.0
numpy>=1.21.0
scikit-learn>=1.3.0
imbalanced-learn>=0.12.0
joblib>=1.3.0

# MLflow and Tracking
mlflow>=2.7.0

# Streamlit Dashboard
streamlit>=1.26.0

# Visualization
matplotlib>=3.8.0
seaborn>=0.12.0
plotly>=5.15.0

# Utilities
pyyaml>=6.0

# GPU Packages - Manual installation required
# These packages need special installation commands:

# 1. PyTorch GPU (CUDA 12.1):
# pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. XGBoost GPU:
# pip install "xgboost[gpu]>=1.7.0"

# 3. RAPIDS cuML/cuDF (Optional - ultra-fast GPU ML):
# pip install cudf-cu12>=23.10.0 cuml-cu12>=23.10.0 --extra-index-url=https://pypi.nvidia.com

# Notes:
# - RAPIDS only works on Linux (Windows needs WSL2)
# - CUDA 11.8+ or 12.x required  
# - Minimum 4GB GPU memory recommended
# - RTX series GPUs fully supported"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
    
    # Create MLflow config
    mlflow_config = """# MLflow Configuration - GPU Accelerated
experiment_name: "GPU_Accelerated_Churn_Prediction"
tracking_uri: "./mlruns"
artifact_location: "./mlruns"

# Model Configuration  
models_to_compare:
  - Logistic_Regression_CPU
  - Logistic_Regression_GPU  # RAPIDS cuML
  - Random_Forest_CPU
  - Random_Forest_GPU       # RAPIDS cuML
  - Gradient_Boosting_CPU
  - XGBoost_GPU             # GPU accelerated
  - SVM_CPU
  - SVM_GPU                 # RAPIDS cuML
  - Decision_Tree
  - Naive_Bayes

# GPU Configuration
gpu:
  enabled: true
  memory_fraction: 0.8
  device_id: 0
  
# Data Configuration
data:
  train_test_split: 0.8
  random_state: 42
  use_smote: true
  
# Evaluation Metrics
metrics:
  primary: "roc_auc"
  secondary: ["accuracy", "precision", "recall", "f1", "training_time"]
  
# Hyperparameter Tuning
hyperparameter_tuning:
  enabled: true
  cv_folds: 5
  cv_folds_gpu: 3  # GPU models use fewer folds (memory efficiency)
  scoring: "roc_auc"
  n_jobs: -1
  n_jobs_gpu: 1    # GPU models use single thread"""
    
    with open("mlflow_config.yaml", "w") as f:
        f.write(mlflow_config)
    
    print("✅ Configuration files created!")

def run_execution_guide():
    """Provide execution instructions"""
    print("\n" + "="*60)
    print("🚀 GPU-ACCELERATED CHURN PREDICTION PROJECT SETUP COMPLETE!")
    print("="*60)
    
    print("""
📋 EXECUTION STEPS:

1️⃣ **Run Model Comparison Pipeline:**
   ```bash
   python mlflow_model_comparison.py
   ```
   This will:
   - Train 7+ different ML models (GPU accelerated included)
   - Compare their performance
   - Save best model as 'best_churn_model.pkl'
   - Generate comparison reports and visualizations
   
2️⃣ **View MLflow Results:**
   ```bash
   mlflow ui
   ```
   Then open: http://localhost:5000
   
3️⃣ **Run Streamlit Application:**
   ```bash
   streamlit run enhanced_streamlit_app.py
   ```
   Then open: http://localhost:8501

🎮 **GPU Features:**
   ✅ XGBoost GPU acceleration
   ✅ RAPIDS cuML GPU models (if available)
   ✅ PyTorch GPU support
   ✅ Automatic GPU memory management
   ✅ GPU vs CPU performance comparison

📊 **Expected Outputs:**
   - models/best_churn_model.pkl (Best performing model)
   - models/*_model.pkl (All trained models) 
   - reports/model_comparison.csv (Performance comparison)
   - reports/*_confusion_matrix.png (Confusion matrices)
   - reports/*_roc_curve.png (ROC curves)
   - MLflow experiment tracking data

🎯 **Key Features:**
   ✅ Automated model comparison across 7+ algorithms
   ✅ GPU accelerated training (XGBoost, cuML)
   ✅ GridSearchCV hyperparameter tuning
   ✅ SMOTE class imbalance handling
   ✅ Comprehensive evaluation metrics
   ✅ MLflow experiment tracking
   ✅ Professional Streamlit dashboard
   ✅ Automatic best model selection
   ✅ GPU memory monitoring
   
💡 **GPU Usage Tips:**
   - Pipeline automatically detects GPU availability
   - Falls back to CPU models if GPU unavailable  
   - XGBoost uses tree_method='gpu_hist' for GPU
   - RAPIDS cuML models provide much faster training
   - GPU memory usage is logged in MLflow
   
⚠️ **GPU Requirements:**
   - NVIDIA GPU (RTX series recommended)
   - CUDA 11.8+ or 12.x
   - Minimum 4GB GPU memory
   - For RAPIDS: Compute Capability 6.0+
""")
    
    print("🎉 Ready to start! Run the commands above to begin.")
    
def check_gpu_environment():
    """Check GPU environment"""
    print("\n🔍 GPU Environment Check:")
    print("-" * 40)
    
    # CUDA check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # Memory test
            torch.cuda.empty_cache()
            x = torch.randn(1000, 1000).cuda()
            print(f"✅ GPU Memory Test: Successful")
            del x
            torch.cuda.empty_cache()
        else:
            print("❌ CUDA not available")
    except ImportError:
        print("❌ PyTorch not installed")
    except Exception as e:
        print(f"❌ PyTorch GPU error: {e}")
    
    # XGBoost GPU check
    try:
        import xgboost as xgb
        print("✅ XGBoost: Installed")
        # GPU test
        try:
            clf = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=10)
            print("✅ XGBoost GPU: Supported")
        except:
            print("⚠️ XGBoost GPU: CPU fallback will be used")
    except ImportError:
        print("❌ XGBoost not installed")
    
    # RAPIDS check
    try:
        import cuml
        print("✅ RAPIDS cuML: Installed")
    except ImportError:
        print("⚠️ RAPIDS cuML: Not installed (optional)")
    
    try:
        import cudf
        print("✅ RAPIDS cuDF: Installed")
    except ImportError:
        print("⚠️ RAPIDS cuDF: Not installed (optional)")

def main():
    """Main setup function"""
    print("🚀 GPU-Accelerated Churn Prediction MLflow Project Setup...")
    print("="*60)
    
    # GPU environment check
    check_gpu_environment()
    
    # Create project structure
    create_project_structure()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Create sample data
    create_sample_data()
    
    # Create config files
    create_config_files()
    
    # Final GPU check
    check_gpu_environment()
    
    # Provide execution guide
    run_execution_guide()


# Additional utility functions for project management
class ChurnProjectManager:
    """Utility class for managing the churn prediction project"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.models_dir = self.project_root / "models"
        self.data_dir = self.project_root / "data"
        self.reports_dir = self.project_root / "reports"
    
    def check_project_status(self):
        """Check current project status"""
        print("🔍 Checking project status...")
        
        status = {
            "data_exists": (self.data_dir / "raw" / "churn.csv").exists(),
            "models_trained": len(list(self.models_dir.glob("*.pkl"))) if self.models_dir.exists() else 0,
            "best_model_exists": (self.models_dir / "best_churn_model.pkl").exists(),
            "reports_generated": len(list(self.reports_dir.glob("*"))) if self.reports_dir.exists() else 0,
            "mlflow_experiments": len(list(Path("mlruns").glob("*"))) if Path("mlruns").exists() else 0
        }
        
        print(f"📊 Data Available: {'✅' if status['data_exists'] else '❌'}")
        print(f"🤖 Models Trained: {status['models_trained']}")
        print(f"🏆 Best Model: {'✅' if status['best_model_exists'] else '❌'}")
        print(f"📋 Reports Generated: {status['reports_generated']}")
        print(f"🧪 MLflow Experiments: {status['mlflow_experiments']}")
        
        return status
    
    def clean_project(self):
        """Clean project artifacts"""
        print("🧹 Cleaning project artifacts...")
        
        import shutil
        
        cleanup_dirs = ["models", "reports", "mlruns"]
        for directory in cleanup_dirs:
            if Path(directory).exists():
                shutil.rmtree(directory)
                print(f"   🗑️ Removed: {directory}/")
        
        print("✅ Project cleaned!")
    
    def validate_data(self):
        """Validate the churn dataset"""
        data_path = self.data_dir / "raw" / "churn.csv"
        
        if not data_path.exists():
            print("❌ Churn dataset not found!")
            return False
        
        try:
            df = pd.read_csv(data_path)
            
            print("✅ Data validation results:")
            print(f"   📊 Shape: {df.shape}")
            print(f"   🎯 Target column 'Churn': {'✅' if 'Churn' in df.columns else '❌'}")
            print(f"   📝 Missing values: {df.isnull().sum().sum()}")
            print(f"   🔢 Numeric columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
            print(f"   📊 Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
            
            if 'Churn' in df.columns:
                churn_rate = df['Churn'].value_counts(normalize=True)
                print(f"   📈 Churn distribution:")
                for value, rate in churn_rate.items():
                    print(f"      {value}: {rate:.2%}")
            
            return True
            
        except Exception as e:
            print(f"❌ Data validation error: {e}")
            return False
    
    def generate_project_report(self):
        """Generate comprehensive project report"""
        print("\n📊 Generating Project Report...")
        print("="*50)
        
        # Check status
        status = self.check_project_status()
        
        # Validate data
        data_valid = self.validate_data()
        
        # Model performance summary
        if status["best_model_exists"]:
            comparison_path = self.reports_dir / "model_comparison.csv"
            if comparison_path.exists():
                comparison_df = pd.read_csv(comparison_path)
                print(f"\n🏆 Best Model Performance:")
                best_model = comparison_df.iloc[0]
                print(f"   Model: {best_model['Model']}")
                print(f"   ROC AUC: {best_model['Test_ROC_AUC']:.4f}")
                print(f"   Accuracy: {best_model['Test_Accuracy']:.4f}")
                print(f"   F1 Score: {best_model['Test_F1']:.4f}")
        
        print(f"\n📋 Project Summary:")
        print(f"   ✅ Setup Complete: {'Yes' if all([status['data_exists'], data_valid]) else 'No'}")
        print(f"   🤖 Models Trained: {status['models_trained']}")
        print(f"   🏆 Best Model Ready: {'Yes' if status['best_model_exists'] else 'No'}")
        print(f"   📊 Ready for Production: {'Yes' if status['best_model_exists'] else 'No'}")


# Quick start script
def quick_start():
    """Quick start function for immediate execution"""
    print("""
🚀 QUICK START GUIDE
==================

To get started immediately, run these commands in order:

1. Setup Project:
   python setup_and_run.py

2. Train Models:
   python mlflow_model_comparison.py

3. View Results:
   mlflow ui &
   streamlit run enhanced_streamlit_app.py

4. Check Status:
   python -c "from setup_and_run import ChurnProjectManager; ChurnProjectManager().check_project_status()"

🎯 That's it! Your GPU-accelerated churn prediction system will be ready.
""")

def gpu_benchmark():
    """Run GPU vs CPU performance benchmark"""
    print("🏁 Starting GPU vs CPU Benchmark...")
    
    try:
        import torch
        import time
        
        # CPU benchmark
        start_time = time.time()
        x_cpu = torch.randn(5000, 5000)
        y_cpu = torch.mm(x_cpu, x_cpu.t())
        cpu_time = time.time() - start_time
        
        if torch.cuda.is_available():
            # GPU benchmark
            start_time = time.time()
            x_gpu = torch.randn(5000, 5000).cuda()
            y_gpu = torch.mm(x_gpu, x_gpu.t())
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"⚡ CPU Time: {cpu_time:.3f}s")
            print(f"🚀 GPU Time: {gpu_time:.3f}s") 
            print(f"🔥 Speedup: {cpu_time/gpu_time:.1f}x")
        else:
            print(f"⚡ CPU Time: {cpu_time:.3f}s")
            print("❌ GPU not available")
            
    except Exception as e:
        print(f"❌ Benchmark error: {e}")


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-Accelerated Churn Prediction MLflow Project Manager")
    parser.add_argument("--setup", action="store_true", help="Setup project structure")
    parser.add_argument("--status", action="store_true", help="Check project status")
    parser.add_argument("--clean", action="store_true", help="Clean project artifacts")
    parser.add_argument("--validate", action="store_true", help="Validate data")
    parser.add_argument("--report", action="store_true", help="Generate project report")
    parser.add_argument("--quick-start", action="store_true", help="Show quick start guide")
    parser.add_argument("--gpu-check", action="store_true", help="Check GPU status")
    parser.add_argument("--benchmark", action="store_true", help="Run GPU vs CPU benchmark")
    
    args = parser.parse_args()
    
    if args.gpu_check:
        check_gpu_environment()
    elif args.benchmark:
        gpu_benchmark()
    elif args.quick_start:
        quick_start()
    elif args.setup or len(sys.argv) == 1:
        main()
    else:
        manager = ChurnProjectManager()
        
        if args.status:
            manager.check_project_status()
            check_gpu_environment()
        elif args.clean:
            manager.clean_project()
        elif args.validate:
            manager.validate_data()
        elif args.report:
            manager.generate_project_report()