"""
Audio Feature Extractor using Librosa
Extracts speech characteristics, acoustic parameters, and spectral features.
"""

import numpy as np
import librosa
import soundfile as sf
import os
import io

def extract_features_from_audio(y: np.ndarray, sr: int) -> dict:
    """
    Extract comprehensive acoustic features from raw audio signal y and sample rate sr.
    """
    # Ensure audio is not completely silent or empty
    if len(y) == 0:
        raise ValueError("Audio signal is empty")
    
    # Normalize audio volume
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    duration = float(len(y) / sr)

    # 1. RMS Energy (Loudness / Intensity)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms))

    # 2. Pitch / Fundamental Frequency (F0)
    # Use pyin or yin for pitch tracking
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sr
    )
    # Clean F0 values (remove NaNs where unvoiced)
    valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    
    if len(valid_f0) > 0:
        pitch_mean = float(np.mean(valid_f0))
        pitch_std = float(np.std(valid_f0))
        pitch_max = float(np.max(valid_f0))
        pitch_min = float(np.min(valid_f0))
    else:
        # Fallback if no voiced pitch detected
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_max = 0.0
        pitch_min = 0.0

    # 3. Spectral Features (Timbre, Brightness, Noise)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = float(np.mean(rolloff))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_mean = float(np.mean(bandwidth))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = float(np.mean(contrast))

    # 4. Zero Crossing Rate (Articulation / Perceived Sharpness / Hiss)
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # 5. MFCCs (Mel-Frequency Cepstral Coefficients - 20 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = [float(val) for val in np.mean(mfccs, axis=1)]
    mfccs_std = [float(val) for val in np.std(mfccs, axis=1)]

    # 6. Chroma STFT (Harmonic Content)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = float(np.mean(chroma))

    # 7. Tempo / Rhythm estimation
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo_val = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
    except Exception:
        tempo_val = 120.0  # Default fallback tempo

    # Compile feature vector for ML model
    # Combine mfccs_mean, mfccs_std, rms, pitch, spectral features, zcr
    feature_vector = np.array(
        mfccs_mean + mfccs_std + [
            rms_mean, rms_std, rms_max,
            pitch_mean, pitch_std,
            centroid_mean, centroid_std,
            rolloff_mean, bandwidth_mean, contrast_mean,
            zcr_mean, zcr_std, chroma_mean, tempo_val
        ]
    )

    # Human-readable Speech Characteristics
    speech_characteristics = {
        "duration_seconds": round(duration, 2),
        "pitch_avg_hz": round(pitch_mean, 1),
        "pitch_variation_hz": round(pitch_std, 1),
        "loudness_rms": round(rms_mean, 4),
        "loudness_peak": round(rms_max, 4),
        "tempo_bpm": round(tempo_val, 1),
        "spectral_centroid_hz": round(centroid_mean, 1),
        "zero_crossing_rate": round(zcr_mean, 4),
        "articulation_sharpness": "High" if zcr_mean > 0.08 else ("Moderate" if zcr_mean > 0.04 else "Soft"),
        "pitch_range": "High Pitch" if pitch_mean > 220 else ("Medium Pitch" if pitch_mean > 130 else ("Low Pitch" if pitch_mean > 0 else "Unpitched/Whisper")),
        "energy_level": "High Dynamic" if rms_mean > 0.15 else ("Moderate" if rms_mean > 0.05 else "Low Dynamic")
    }

    return {
        "feature_vector": feature_vector.tolist(),
        "characteristics": speech_characteristics,
        "raw_features": {
            "mfccs_mean": mfccs_mean,
            "mfccs_std": mfccs_std,
            "rms_mean": rms_mean,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "centroid_mean": centroid_mean,
            "zcr_mean": zcr_mean,
            "chroma_mean": chroma_mean,
            "tempo_bpm": tempo_val
        }
    }


def extract_features_from_file(file_path_or_bytes) -> dict:
    """
    Load audio from file path or bytes buffer and extract features.
    """
    if isinstance(file_path_or_bytes, bytes):
        buffer = io.BytesIO(file_path_or_bytes)
        y, sr = librosa.load(buffer, sr=22050, mono=True)
    else:
        y, sr = librosa.load(file_path_or_bytes, sr=22050, mono=True)
    
    return extract_features_from_audio(y, sr)
