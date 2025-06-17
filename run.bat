@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting FastAPI server...
uvicorn app.main:app --reload