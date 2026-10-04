/**
 * Voice Sentiment Analyzer Frontend Application JavaScript
 * Handles MediaRecorder Web Audio API recording, live canvas visualizer,
 * sample selector, drag-and-drop file upload, API fetch, and dynamic UI dashboard rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  const btnRec = document.getElementById('btnRec');
  const btnStop = document.getElementById('btnStop');
  const recTimer = document.getElementById('recTimer');
  const audioCanvas = document.getElementById('audioCanvas');
  const canvasCtx = audioCanvas.getContext('2d');

  const samplesGrid = document.getElementById('samplesGrid');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');

  const audioPlayerBox = document.getElementById('audioPlayerBox');
  const audioSourceLabel = document.getElementById('audioSourceLabel');
  const audioDuration = document.getElementById('audioDuration');
  const audioPlayer = document.getElementById('audioPlayer');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const loadingOverlay = document.getElementById('loadingOverlay');

  const emptyState = document.getElementById('emptyState');
  const resultsContainer = document.getElementById('resultsContainer');

  // Hero Elements
  const heroEmoji = document.getElementById('heroEmoji');
  const heroEmotion = document.getElementById('heroEmotion');
  const heroSentimentTag = document.getElementById('heroSentimentTag');
  const heroDurationTag = document.getElementById('heroDurationTag');
  const heroConfidence = document.getElementById('heroConfidence');

  // Dashboard Elements
  const emotionsList = document.getElementById('emotionsList');
  const pointDot = document.getElementById('pointDot');
  const coordVal = document.getElementById('coordVal');

  // Metric Cards
  const mPitch = document.getElementById('mPitch');
  const mPitchRange = document.getElementById('mPitchRange');
  const mPitchStd = document.getElementById('mPitchStd');
  const mTempo = document.getElementById('mTempo');
  const mRms = document.getElementById('mRms');
  const mEnergy = document.getElementById('mEnergy');
  const mCentroid = document.getElementById('mCentroid');
  const mZcr = document.getElementById('mZcr');
  const mSharpness = document.getElementById('mSharpness');

  // App State Variables
  let mediaRecorder = null;
  let audioChunks = [];
  let currentAudioBlob = null;
  let currentFile = null;
  let isRecording = false;
  let timerInterval = null;
  let secondsElapsed = 0;

  let audioCtx = null;
  let analyser = null;
  let animFrameId = null;

  const EMOTION_CONFIG = {
    "Happy":   { emoji: "😊", color: "#ffd700" },
    "Sad":     { emoji: "🥺", color: "#4a90e2" },
    "Angry":   { emoji: "😠", color: "#ff4757" },
    "Neutral": { emoji: "😐", color: "#a0a8ba" },
    "Excited": { emoji: "🤩", color: "#ff6b81" },
    "Fearful": { emoji: "😨", color: "#9b59b6" }
  };

  // Resize Canvas
  function resizeCanvas() {
    audioCanvas.width = audioCanvas.parentElement.clientWidth;
    audioCanvas.height = audioCanvas.parentElement.clientHeight;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Draw Idle Canvas
  function drawIdleCanvas() {
    canvasCtx.clearRect(0, 0, audioCanvas.width, audioCanvas.height);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgba(0, 242, 254, 0.2)';
    canvasCtx.beginPath();

    const cy = audioCanvas.height / 2;
    canvasCtx.moveTo(0, cy);
    canvasCtx.lineTo(audioCanvas.width, cy);
    canvasCtx.stroke();
  }
  drawIdleCanvas();

  // 1. Tab Switching Handler
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      document.getElementById(`tab-${targetTab}`).classList.add('active');
    });
  });

  // 2. Microphone Recording Logic
  btnRec.addEventListener('click', async () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  btnStop.addEventListener('click', () => {
    if (isRecording) stopRecording();
  });

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        currentAudioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        currentFile = new File([currentAudioBlob], "recording.webm", { type: "audio/webm" });
        setAudioForAnalysis(currentFile, "Voice Recording (Live Mic)");
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      isRecording = true;
      btnRec.classList.add('recording');
      btnRec.innerHTML = '<i class="fa-solid fa-stop"></i>';
      btnStop.classList.remove('hidden');

      // Timer
      secondsElapsed = 0;
      recTimer.textContent = "00:00";
      timerInterval = setInterval(() => {
        secondsElapsed++;
        const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
        const secs = String(secondsElapsed % 60).padStart(2, '0');
        recTimer.textContent = `${mins}:${secs}`;
      }, 1000);

      // Start Audio Visualizer
      setupLiveVisualizer(stream);

    } catch (err) {
      alert("Microphone permission denied or not supported on this device. You can test using the built-in Audio Samples tab!");
      console.error(err);
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      isRecording = false;
      btnRec.classList.remove('recording');
      btnRec.innerHTML = '<i class="fa-solid fa-microphone"></i>';
      btnStop.classList.add('hidden');
      clearInterval(timerInterval);
      if (animFrameId) cancelAnimationFrame(animFrameId);
      drawIdleCanvas();
    }
  }

  // Live Audio Visualizer (Waveform & Spectrum)
  function setupLiveVisualizer(stream) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 128;

    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
      if (!isRecording) return;
      animFrameId = requestAnimationFrame(draw);

      analyser.getByteFrequencyData(dataArray);

      canvasCtx.clearRect(0, 0, audioCanvas.width, audioCanvas.height);
      const barWidth = (audioCanvas.width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * audioCanvas.height;

        const gradient = canvasCtx.createLinearGradient(0, audioCanvas.height, 0, 0);
        gradient.addColorStop(0, 'rgba(0, 242, 254, 0.4)');
        gradient.addColorStop(1, 'rgba(127, 0, 255, 0.8)');

        canvasCtx.fillStyle = gradient;
        canvasCtx.fillRect(x, audioCanvas.height - barHeight, barWidth - 2, barHeight);

        x += barWidth;
      }
    }
    draw();
  }

  // 3. Load Audio Samples
  async function loadPresetSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      samplesGrid.innerHTML = '';

      data.samples.forEach(sample => {
        const config = EMOTION_CONFIG[sample.emotion] || { emoji: "🎵", color: "#00f2fe" };
        const card = document.createElement('div');
        card.className = 'sample-card';
        card.innerHTML = `
          <div class="sample-emoji">${config.emoji}</div>
          <div class="sample-title">${sample.label}</div>
        `;

        card.addEventListener('click', async () => {
          document.querySelectorAll('.sample-card').forEach(c => c.classList.remove('active'));
          card.classList.add('active');

          // Fetch audio file blob
          const audioRes = await fetch(sample.url);
          const blob = await audioRes.blob();
          const file = new File([blob], sample.filename, { type: "audio/wav" });

          setAudioForAnalysis(file, `Sample: ${sample.label}`);
        });

        samplesGrid.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load sample clips:", err);
    }
  }
  loadPresetSamples();

  // 4. Dropzone File Upload
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));

  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleUploadedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleUploadedFile(fileInput.files[0]);
    }
  });

  function handleUploadedFile(file) {
    if (!file.type.startsWith('audio/')) {
      alert("Please upload a valid audio file (WAV, MP3, WEBM, OGG, FLAC)");
      return;
    }
    setAudioForAnalysis(file, `File: ${file.name}`);
  }

  // 5. Audio Player Setup
  function setAudioForAnalysis(file, labelText) {
    currentFile = file;
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    audioSourceLabel.textContent = labelText;
    audioPlayerBox.classList.remove('hidden');

    audioPlayer.onloadedmetadata = () => {
      audioDuration.textContent = `${audioPlayer.duration.toFixed(1)}s`;
    };
  }

  // 6. Analyze Voice Sentiment API Request
  btnAnalyze.addEventListener('click', async () => {
    if (!currentFile) {
      alert("Please record audio or select a sample file first.");
      return;
    }

    loadingOverlay.classList.remove('hidden');
    btnAnalyze.disabled = true;

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Analysis failed");
      }

      const result = await response.json();
      renderDashboard(result);

    } catch (err) {
      alert(`Error analyzing voice: ${err.message}`);
      console.error(err);
    } finally {
      loadingOverlay.classList.add('hidden');
      btnAnalyze.disabled = false;
    }
  });

  // 7. Render Analytics Dashboard UI
  function renderDashboard(data) {
    const pred = data.prediction;
    const chars = data.characteristics;
    const primary = pred.primary_emotion;
    const config = EMOTION_CONFIG[primary] || { emoji: "🎤", color: "#00f2fe" };

    // Update Hero Card
    heroEmoji.textContent = config.emoji;
    heroEmotion.textContent = primary;
    heroEmotion.style.color = config.color;
    heroSentimentTag.textContent = `${pred.sentiment_label} Sentiment`;
    heroDurationTag.textContent = `${chars.duration_seconds}s Duration`;
    heroConfidence.textContent = `${(pred.confidence * 100).toFixed(1)}%`;

    // Render Emotion Probabilities Bars
    emotionsList.innerHTML = '';
    Object.entries(pred.emotion_probabilities).forEach(([emotion, prob]) => {
      const emoConfig = EMOTION_CONFIG[emotion] || { emoji: "", color: "#00f2fe" };
      const pct = (prob * 100).toFixed(1);

      const row = document.createElement('div');
      row.className = 'emotion-row';
      row.innerHTML = `
        <div class="emotion-label-group">
          <span>${emoConfig.emoji} ${emotion}</span>
          <span style="color: ${emoConfig.color}">${pct}%</span>
        </div>
        <div class="bar-bg">
          <div class="bar-fill" style="width: ${pct}%; background-color: ${emoConfig.color};"></div>
        </div>
      `;
      emotionsList.appendChild(row);
    });

    // Update Valence-Arousal Quadrant Point
    const val = pred.valence_arousal.valence; // -1 to +1
    const aro = pred.valence_arousal.arousal; // -1 to +1

    // Map -1..+1 to percentage offsets (10% to 90%)
    const leftPct = 50 + (val * 40);
    const topPct = 50 - (aro * 40);

    pointDot.style.left = `${leftPct}%`;
    pointDot.style.top = `${topPct}%`;
    coordVal.textContent = `Valence: ${val > 0 ? '+' : ''}${val} | Arousal: ${aro > 0 ? '+' : ''}${aro}`;

    // Update Acoustic Metrics Cards
    mPitch.textContent = `${chars.pitch_avg_hz} Hz`;
    mPitchRange.textContent = chars.pitch_range;
    mPitchStd.textContent = `${chars.pitch_variation_hz} Hz`;
    mTempo.textContent = `${chars.tempo_bpm} BPM`;
    mRms.textContent = chars.loudness_rms;
    mEnergy.textContent = chars.energy_level;
    mCentroid.textContent = `${chars.spectral_centroid_hz} Hz`;
    mZcr.textContent = chars.zero_crossing_rate;
    mSharpness.textContent = `${chars.articulation_sharpness} Articulation`;

    // Switch View
    emptyState.classList.add('hidden');
    resultsContainer.classList.remove('hidden');
    resultsContainer.scrollIntoView({ behavior: 'smooth' });
  }

});
