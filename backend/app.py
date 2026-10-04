"""
FastAPI Server for Voice Sentiment Analyzer
Provides audio upload analysis endpoints, sample file endpoints, and serves static frontend assets.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
import shutil
import tempfile

# Ensure backend directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import extract_features_from_file
from classifier import classifier
from sample_generator import ensure_sample_files_exist, SAMPLES_DIR

app = FastAPI(
    title="Voice Sentiment Analyzer API",
    description="Speech sentiment and emotion category prediction using Librosa and ML classifier",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Generate sample audio files on startup
ensure_sample_files_exist()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_SAMPLES_DIR = SAMPLES_DIR

# Mount static files for audio samples first
app.mount("/static/samples", StaticFiles(directory=STATIC_SAMPLES_DIR), name="static_samples")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Voice Sentiment Analyzer",
        "supported_emotions": classifier.emotions
    }


@app.get("/api/samples")
async def get_audio_samples():
    """
    Returns list of preset sample audio clips available for testing.
    """
    samples = [
        {"id": "happy", "label": "Joyful / Happy", "emotion": "Happy", "filename": "sample_happy.wav", "url": "/static/samples/sample_happy.wav"},
        {"id": "sad", "label": "Melancholic / Sad", "emotion": "Sad", "filename": "sample_sad.wav", "url": "/static/samples/sample_sad.wav"},
        {"id": "angry", "label": "Intense / Angry", "emotion": "Angry", "filename": "sample_angry.wav", "url": "/static/samples/sample_angry.wav"},
        {"id": "neutral", "label": "Calm / Neutral", "emotion": "Neutral", "filename": "sample_neutral.wav", "url": "/static/samples/sample_neutral.wav"},
        {"id": "excited", "label": "Energetic / Excited", "emotion": "Excited", "filename": "sample_excited.wav", "url": "/static/samples/sample_excited.wav"},
        {"id": "fearful", "label": "Anxious / Fearful", "emotion": "Fearful", "filename": "sample_fearful.wav", "url": "/static/samples/sample_fearful.wav"}
    ]
    return {"samples": samples}


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyzes uploaded audio file (WAV, MP3, WEBM, OGG, FLAC) using Librosa & ML classifier.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Read binary content
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

    # Save to temporary file for librosa processing
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Extract features
        extracted = extract_features_from_file(tmp_path)
        
        # Run ML sentiment prediction
        prediction = classifier.predict(extracted)

        response_payload = {
            "filename": file.filename,
            "prediction": prediction,
            "characteristics": extracted["characteristics"],
            "raw_features_summary": {
                "mfcc_1_mean": round(extracted["raw_features"]["mfccs_mean"][0], 2),
                "pitch_mean_hz": round(extracted["raw_features"]["pitch_mean"], 1),
                "loudness_rms": round(extracted["raw_features"]["rms_mean"], 4),
                "spectral_centroid_hz": round(extracted["raw_features"]["centroid_mean"], 1),
                "zero_crossing_rate": round(extracted["raw_features"]["zcr_mean"], 4),
                "tempo_bpm": round(extracted["raw_features"]["tempo_bpm"], 1)
            }
        }
        return JSONResponse(content=response_payload)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")
    
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Mount frontend static files at root '/'
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
