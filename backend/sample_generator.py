"""
Synthetic Audio Sample Generator
Generates clean, realistic WAV audio sample files for test emotions:
Happy, Sad, Angry, Neutral, Excited, Fearful.
"""

import numpy as np
import scipy.io.wavfile as wavfile
import os

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "static", "samples")

def generate_sample_audio(emotion: str, filename: str, sr=22050, duration=3.0):
    """
    Generates synthetic speech-like acoustic audio for a specific emotion.
    """
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    filepath = os.path.join(SAMPLES_DIR, filename)

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Base amplitude envelope with speech pause cadence
    cadence = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
    
    if emotion == "Happy":
        # Dynamic upward pitch contour, bright harmonics
        f0 = 230 + 40 * np.sin(2 * np.pi * 2.5 * t)
        signal = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * f0 * 2 * t) + 0.1 * np.sin(2 * np.pi * f0 * 3 * t)
        signal = signal * (0.3 + 0.7 * cadence)

    elif emotion == "Sad":
        # Slow downward pitch drift, soft volume, warm low harmonics
        f0 = 130 - 15 * (t / duration)
        signal = 0.15 * np.sin(2 * np.pi * f0 * t) + 0.05 * np.sin(2 * np.pi * f0 * 2 * t)
        signal = signal * (0.2 + 0.3 * cadence)

    elif emotion == "Angry":
        # Sharp high intensity, high pitch with rapid distortion & harsh harmonics
        f0 = 260 + 50 * np.sin(2 * np.pi * 5.0 * t)
        signal = 0.6 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.square(np.sin(2 * np.pi * f0 * 1.5 * t))
        noise = np.random.normal(0, 0.05, len(t))
        signal = (signal + noise) * (0.5 + 0.5 * cadence)

    elif emotion == "Neutral":
        # Steady pitch, moderate volume, simple clean vocal formant
        f0 = 165 + 5 * np.sin(2 * np.pi * 1.0 * t)
        signal = 0.25 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * f0 * 2 * t)
        signal = signal * (0.4 + 0.6 * cadence)

    elif emotion == "Excited":
        # Very high pitch, rapid pitch modulation, high energy pulse
        f0 = 310 + 70 * np.sin(2 * np.pi * 6.0 * t)
        signal = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * f0 * 2 * t) + 0.15 * np.sin(2 * np.pi * f0 * 4 * t)
        signal = signal * (0.4 + 0.6 * (1.0 + np.sin(2 * np.pi * 5.0 * t)))

    elif emotion == "Fearful":
        # Tremor / jitter in pitch, fluctuating volume, airy noise component
        f0 = 240 + 30 * np.sin(2 * np.pi * 8.0 * t) + np.random.normal(0, 5, len(t))
        signal = 0.2 * np.sin(2 * np.pi * f0 * t) + 0.08 * np.sin(2 * np.pi * f0 * 2 * t)
        noise = np.random.normal(0, 0.03, len(t))
        signal = (signal + noise) * (0.2 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)))

    else:
        signal = 0.2 * np.sin(2 * np.pi * 200 * t)

    # Normalize audio to prevent clipping
    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        signal = signal / max_amp * 0.85

    # Convert to 16-bit PCM WAV
    audio_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(filepath, sr, audio_int16)
    return filepath

def ensure_sample_files_exist():
    """
    Generates all sample files if they do not exist.
    """
    samples = [
        ("Happy", "sample_happy.wav"),
        ("Sad", "sample_sad.wav"),
        ("Angry", "sample_angry.wav"),
        ("Neutral", "sample_neutral.wav"),
        ("Excited", "sample_excited.wav"),
        ("Fearful", "sample_fearful.wav")
    ]
    created = []
    for emotion, filename in samples:
        path = os.path.join(SAMPLES_DIR, filename)
        if not os.path.exists(path):
            generate_sample_audio(emotion, filename)
            created.append(filename)
    return created

if __name__ == "__main__":
    ensure_sample_files_exist()
    print("Sample audio files successfully generated!")
