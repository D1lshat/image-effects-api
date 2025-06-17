from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

@router.post("/generate_image_gemini/")
async def generate_image_gemini(prompt: str = Form(...)):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in .env")
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is empty")

    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        payload = {
            "contents": [{
                "parts": [{"text": f"Generate an image for: {prompt}"}]
            }]
        }

        response = requests.post(url, headers=headers, data=json.dumps(payload))
        data = response.json()

        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini image generation failed: {str(e)}")
