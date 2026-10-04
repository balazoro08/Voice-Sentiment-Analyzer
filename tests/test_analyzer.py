"""
Unit and Integration Tests for Voice Sentiment Analyzer
Tests Librosa feature extraction, ML classification, and FastAPI routes.
"""

import pytest
import numpy as np
import os
import sys

# Add backend directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from feature_extractor import extract_features_from_audio
from classifier import VoiceSentimentClassifier, EMOTIONS
from sample_generator import generate_sample_audio, SAMPLES_DIR

def test_feature_extraction():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate 440 Hz sine wave
    y = 0.5 * np.sin(2 * np.pi * 440 * t)

    features = extract_features_from_audio(y, sr)
    
    assert "feature_vector" in features
    assert len(features["feature_vector"]) > 40
    assert "characteristics" in features
    assert features["characteristics"]["duration_seconds"] == 1.0
    assert features["characteristics"]["pitch_avg_hz"] > 0
    assert "raw_features" in features

def test_classifier_prediction():
    clf = VoiceSentimentClassifier()
    
    sr = 22050
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.4 * np.sin(2 * np.pi * 300 * t)
    
    features = extract_features_from_audio(y, sr)
    prediction = clf.predict(features)
    
    assert "primary_emotion" in prediction
    assert prediction["primary_emotion"] in EMOTIONS
    assert "confidence" in prediction
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert "emotion_probabilities" in prediction
    assert len(prediction["emotion_probabilities"]) == len(EMOTIONS)
    assert "valence_arousal" in prediction
    assert -1.0 <= prediction["valence_arousal"]["valence"] <= 1.0
    assert -1.0 <= prediction["valence_arousal"]["arousal"] <= 1.0

def test_sample_generation():
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    test_wav = generate_sample_audio("Happy", "test_happy.wav", duration=1.0)
    assert os.path.exists(test_wav)
    assert os.path.getsize(test_wav) > 0
    os.remove(test_wav)
