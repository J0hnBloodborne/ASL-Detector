const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const preview = document.getElementById('preview');
const ctx = overlay.getContext('2d');
const statusEl = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const detectedLetters = document.getElementById('detected-letters');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const camControls = document.getElementById('cam-controls');
const uploadControls = document.getElementById('upload-controls');
const fileInput = document.getElementById('file-input');

let intervalId = null;
let frameCount = 0;
let lastTime = performance.now();
let mode = 'cam';

// --- Word builder state ---
let currentWord = '';
let lastPrediction = null;  // tracks top-1 from latest frame

const hiddenCanvas = document.createElement('canvas');
const hiddenCtx = hiddenCanvas.getContext('2d');

// --- Mode switching ---
function switchMode(m) {
    mode = m;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-' + m).classList.add('active');

    if (m === 'cam') {
        camControls.style.display = 'flex';
        uploadControls.style.display = 'none';
        preview.style.display = 'none';
        video.style.display = 'block';
    } else {
        stopDetection();
        camControls.style.display = 'none';
        uploadControls.style.display = 'flex';
        video.style.display = 'none';
    }
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    detectedLetters.innerHTML = '';
}

// --- Webcam ---
async function initWebcam() {
    try {
        // Support old Chrome via webkit prefix
        const gum = (navigator.mediaDevices && navigator.mediaDevices.getUserMedia)
            ? navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices)
            : (constraints) => new Promise((res, rej) => {
                const fn = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
                if (!fn) { rej(new Error('getUserMedia not supported')); return; }
                fn.call(navigator, constraints, res, rej);
              });
        const stream = await gum({ video: { width: { ideal: 640 }, height: { ideal: 640 } } });
        video.srcObject = stream;
        await video.play();
        // Always 640x640 to match backend inference size
        overlay.width = 640;
        overlay.height = 640;
        hiddenCanvas.width = 640;
        hiddenCanvas.height = 640;
        statusEl.textContent = 'Camera ready';
    } catch (e) {
        statusEl.textContent = 'Camera unavailable — use Upload tab';
        console.warn('Webcam error:', e);
    }
}

// --- Capture + predict (webcam) ---
// Applies contrast + brightness + saturation to compensate for bad webcam quality
async function captureAndPredict() {
    if (video.readyState < 2) return;
    hiddenCtx.filter = 'contrast(1.4) brightness(1.1) saturate(1.3)';
    hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    hiddenCtx.filter = 'none';
    const b64 = hiddenCanvas.toDataURL('image/jpeg', 0.92);
    await sendPredict(b64);
}

// --- Upload handler ---
async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    statusEl.textContent = 'Processing...';

    const reader = new FileReader();
    reader.onload = async function (ev) {
        const b64 = ev.target.result;
        // Show preview
        preview.src = b64;
        preview.style.display = 'block';

        // Overlay always 640x640 to match backend inference output
        overlay.width = 640;
        overlay.height = 640;
        await sendPredict(b64);
        statusEl.textContent = 'Done';
    };
    reader.readAsDataURL(file);
}

// --- Shared predict ---
async function sendPredict(base64) {
    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64 })
        });
        const data = await res.json();
        drawPredictions(data.predictions);
        updateLetters(data.predictions);
    } catch (err) {
        console.error('Predict error:', err);
    }

    frameCount++;
    const now = performance.now();
    if (now - lastTime >= 1000) {
        fpsEl.textContent = frameCount + ' FPS';
        frameCount = 0;
        lastTime = now;
    }
}

// --- Draw top-1 box on image ---
function drawPredictions(preds) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!preds.length) return;

    // Only the highest-confidence prediction
    const p = preds[0];
    const [x1, y1, x2, y2] = p.box;
    const label = p.label + ' ' + (p.confidence * 100).toFixed(0) + '%';

    // Green box
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Label tag: above box if room, otherwise inside top of box
    ctx.font = 'bold 18px sans-serif';
    const tw = ctx.measureText(label).width;
    const tagH = 26;
    // prefer above box; if it would go off top, place inside box at top
    const tagY = y1 >= tagH ? y1 : y1 + tagH;
    ctx.fillStyle = '#00e676';
    ctx.fillRect(x1, tagY - tagH + 2, tw + 10, tagH);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x1 + 5, tagY - 5);
}

// --- Update letter badges (top-1 only) ---
function updateLetters(preds) {
    if (!preds.length) { detectedLetters.innerHTML = ''; lastPrediction = null; return; }
    const p = preds[0];
    lastPrediction = p;
    detectedLetters.innerHTML =
        '<span class="letter-badge">' + p.label + ' <small>' + (p.confidence * 100).toFixed(0) + '%</small></span>';
}

// --- Word builder ---
function renderWord() {
    document.getElementById('word-display').textContent = currentWord || '_';
}

function addLetter() {
    if (!lastPrediction) return;
    currentWord += lastPrediction.label;
    renderWord();
}

function deleteLetter() {
    currentWord = currentWord.slice(0, -1);
    renderWord();
}

function addSpace() {
    currentWord += ' ';
    renderWord();
}

function clearWord() {
    currentWord = '';
    renderWord();
}

// --- Start / Stop ---
function startDetection() {
    if (intervalId) return;
    intervalId = setInterval(captureAndPredict, 300);
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = 'Detecting...';
}

function stopDetection() {
    if (intervalId) { clearInterval(intervalId); intervalId = null; }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.textContent = 'Stopped';
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

document.addEventListener('DOMContentLoaded', initWebcam);
