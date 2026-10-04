# Voice Sentiment Analyzer 🎙️🧠

An end-to-end Machine Learning web application that records a person's voice, extracts speech and acoustic characteristics using **Librosa**, and predicts emotional categories (**Happy**, **Sad**, **Angry**, **Neutral**, **Excited**, **Fearful**) with confidence scores, dynamic visualizations, and a 2D Valence-Arousal emotion plane.

![Voice Sentiment Analyzer](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Librosa](https://img.shields.io/badge/Librosa-1.0.0-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-red.svg)

---

## 🌟 Key Features

- **Microphone Live Recording**: Built-in HTML5 Web Audio API & `MediaRecorder` with real-time frequency spectrum & waveform canvas oscilloscope.
- **Preset Test Audio Clips**: 1-click sample audio clips (**Happy**, **Sad**, **Angry**, **Neutral**, **Excited**, **Fearful**) for immediate testing.
- **Drag & Drop File Upload**: Supports WAV, MP3, WEBM, OGG, and FLAC audio formats.
- **Librosa Acoustic Feature Extractor**:
  - **MFCCs**: 20 Mel-Frequency Cepstral Coefficients.
  - **Pitch ($F_0$)**: Average pitch, pitch variability (tone swings), pitch range.
  - **Loudness (RMS)**: Intensity & dynamic energy levels.
  - **Zero Crossing Rate (ZCR)**: Articulation sharpness & hiss detection.
  - **Spectral Centroid**: Timbre brightness & resonance.
  - **Speech Tempo (BPM)**: Rhythm cadence & onset strength.
- **ML Sentiment Classifier**:
  - Random Forest + MLP Neural Network ensemble classifier.
  - Acoustic rule-based domain calibration.
  - Predicts emotional state, confidence score %, sentiment polarity (Positive / Negative / Neutral), and 2D Valence-Arousal plane coordinates.
- **Glassmorphic Web Dashboard**: Modern responsive UI with glowing audio visualizers, emotion probability distribution bars, and speech metrics cards.

---

## 📁 Project Structure

```
Voice-Sentiment-Analyzer/
├── backend/
│   ├── app.py                # FastAPI Web Server & API Routes
│   ├── classifier.py         # ML Emotion Classifier & Heuristics
│   ├── feature_extractor.py  # Librosa Feature Extraction Pipeline
│   ├── sample_generator.py   # Audio Sample Generator
│   └── static/               # Generated Audio Samples
├── frontend/
│   ├── index.html            # Main HTML5 Dashboard Layout
│   ├── styles.css            # Glassmorphic Modern CSS Design
│   └── app.js                # Web Audio API, Recorder & UI Logic
├── tests/
│   └── test_analyzer.py      # Automated Pytest Suite
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- `pip`

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/balazoro08/Voice-Sentiment-Analyzer.git
cd Voice-Sentiment-Analyzer
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Application

Start the FastAPI server:
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🧪 Running Tests

Run unit & integration tests:
```bash
pytest tests/test_analyzer.py
```

---

## 📜 License
MIT License
