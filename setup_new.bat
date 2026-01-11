@echo off
REM Setup script for Bone Fracture project

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install all required packages (no duplicates)
echo Installing required packages...
pip install tensorflow keras numpy matplotlib pillow customtkinter seaborn scikit-learn streamlit


echo Setup complete.
