@echo off
REM GPU-Accelerated Churn Prediction Setup Script for Windows
REM ========================================================

echo 🚀 GPU-Accelerated Churn Prediction Setup Starting...
echo ========================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ first.
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

REM Create virtual environment
echo.
echo 🔧 Creating virtual environment...
if exist churn_env (
    echo ⚠️ Virtual environment already exists
) else (
    python -m venv churn_env
    echo ✅ Virtual environment created
)

REM Activate virtual environment
echo.
echo 🔌 Activating virtual environment...
call churn_env\Scripts\activate.bat

REM Upgrade pip
echo.
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Run setup script
echo.
echo 🛠️ Running environment setup...
python setup_environment.py

REM Check if setup was successful
if errorlevel 1 (
    echo ❌ Setup failed!
    pause
    exit /b 1
)

echo.
echo ========================================================
echo ✅ Setup completed successfully!
echo ========================================================
echo.
echo 🎯 Next steps:
echo 1. Run: python mlflow_model_comparison.py
echo 2. Run: mlflow ui
echo 3. Run: streamlit run enhanced_streamlit_app.py
echo.
echo 💡 To activate environment later:
echo    churn_env\Scripts\activate.bat
echo.

pause