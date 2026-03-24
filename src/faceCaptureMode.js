/**
 * โหมด Real/Spoof แบบต่อเนื่อง + ครบมุมหมุนหน้า + จับภาพใบหน้า (แสดงผลหน้าบ้านเท่านั้น)
 */
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { analyzePad, getPadCanvasSize } from './padLayer.js';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const NOSE_TIP = 1;
const LEFT_FACE = 234;
const RIGHT_FACE = 454;
const CHIN = 152;
const FOREHEAD = 10;

const MOTION_MIN = 0.85;
/** ต้องมี PAD ตกเบื้องต้นติดกันกี่เฟรมถึงจะถือว่า spoof — โหมดแคปไม่ใช้ micro-motion เป็นสปูฟ (คนถือนิ่งได้) */
const CAPTURE_PAD_CONSECUTIVE = 5;
const SMOOTH_FRAMES = 5;
const PROGRESS_LERP_BASE = 0.88;
const YAW_CENTER = 0.07;
const YAW_SIDE = 0.12;
const PITCH_UP = -0.035;
const PITCH_DOWN = 0.038;

/** หน้าตรงก่อนจับภาพ (คมกว่าเกณฑ์ “กลางซ้ายขวา”) */
const FRONTAL_YAW_MAX = 0.12;
/**
 * p (pitchNorm) = (จมูก.y − กลางดวงตา.y) / faceH — faceH = |คาง−หน้าผาก|
 *    ค่า depend มุมกล้อง/สัดส่วนหน้า; หน้าตรงบางคน ~0.2–0.3 ไม่ใช่ช่วงแคบรอบ 0.05
 * r = (คาง.y−จมูก.y) / (จมูก.y−กลางตา.y) — เงยมากจมูกใกล้ตา → r พุ่งสูง
 */
const FRONTAL_PITCH_MIN = 0.16;
const FRONTAL_PITCH_MAX = 0.34;
/** เงยแรงมัก r สูงกว่าหน้าตรง (เช่น >2.5–3); จับจากอุปกรณ์จริงหน้าตรง r~1.5–2.2 */
const FRONTAL_NC_TO_NE_RATIO_MAX = 3.1;
const FRONTAL_HOLD_FRAMES = 14;
const FRONTAL_WAIT_TIMEOUT_MS = 22000;

const video = /** @type {HTMLVideoElement} */ (document.querySelector('#videoCap'));
const canvas = /** @type {HTMLCanvasElement} */ (document.querySelector('#canvasCap'));
const ctx = canvas.getContext('2d');
const btnStart = document.querySelector('#btnCapStart');
const btnStop = document.querySelector('#btnCapStop');
const capModelStatus = document.querySelector('#capModelStatus');
const capHint = document.querySelector('#capHint');
const capRotationHint = document.querySelector('#capRotationHint');
const rotationProgressFill = document.querySelector('#rotationProgressFill');
const rotationPercent = document.querySelector('#rotationPercent');
const capResult = document.querySelector('#capResult');
const capturePreviewImg = /** @type {HTMLImageElement} */ (document.querySelector('#capturePreviewImg'));
const capHud = document.querySelector('#capHud');

const analyzeCanvas = document.createElement('canvas');
const actx = analyzeCanvas.getContext('2d', { willReadFrequently: true });
const padCanvas = document.createElement('canvas');
const padCtx = padCanvas.getContext('2d', { willReadFrequently: true });

/** @type {FaceLandmarker | null} */
let faceLandmarker = null;
let stream = null;
let running = false;
let rafId = 0;
/** @type {Uint8Array | null} */
let prevFaceGray = null;

let spoofStreak = 0;
let realStreak = 0;
/** @type {'real' | 'spoof'} */
let displayMode = 'spoof';

/** @type {{ center: boolean; left: boolean; right: boolean; up: boolean; down: boolean }} */
const coverage = {
  center: false,
  left: false,
  right: false,
  up: false,
  down: false,
};

const COVERAGE_KEYS = ['center', 'left', 'right', 'up', 'down'];
const COVERAGE_LABEL_TH = {
  center: 'ตรง',
  left: 'ซ้าย',
  right: 'ขวา',
  up: 'เงย',
  down: 'ก้ม',
};
let capturedDone = false;
/** ครบมุมแล้ว — รอหน้าตรงก่อนแคป */
let awaitingNeutralCapture = false;
let neutralHoldFrames = 0;
let neutralPhaseStartedAt = 0;
/** เก็บกรอบ + เฟรม snapshot ตอนคะแนน frontal ดีที่สุด (ใช้เมื่อรอนานเกิน timeout) */
let bestFrontal = {
  score: Infinity,
  /** @type {{ sx: number; sy: number; sw: number; sh: number } | null} */
  box: null,
};
/** @type {HTMLCanvasElement | null} */
let bestFrontalCanvas = null;

/** หยุดค้างเมื่อ spoof ระหว่างจับภาพ — เก็บ landmark snapshot + ขนาดเฟรม */
/** @type {{ lm: { x: number; y: number; z?: number }[]; w: number; h: number; ts: number } | null} */
let spoofFreeze = null;

let padSpoofStreak = 0;

let capListenersBound = false;

/** progress วาดบนวิดีโอ — smooth lerp */
let smoothProgressPct = 0;
let progressLerpLastTs = 0;

function landmarkPoint(lm, idx, w, h) {
  const p = lm[idx];
  return { x: p.x * w, y: p.y * h };
}

function faceBoundingBoxPixels(lm, w, h) {
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (let i = 0; i < lm.length; i++) {
    const p = lm[i];
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const pw = (maxX - minX) * w;
  const ph = (maxY - minY) * h;
  const padX = Math.min(pw * 0.14, w * 0.09);
  const padY = Math.min(ph * 0.16, h * 0.11);
  let sx = minX * w - padX;
  let sy = minY * h - padY;
  let sw = pw + padX * 2;
  let sh = ph + padY * 2;
  sx = Math.max(0, sx);
  sy = Math.max(0, sy);
  sw = Math.min(w - sx, sw);
  sh = Math.min(h - sy, sh);
  return { sx, sy, sw, sh };
}

function yawFromLandmarks(lm, w) {
  const nose = landmarkPoint(lm, NOSE_TIP, w, 1);
  const left = landmarkPoint(lm, LEFT_FACE, w, 1);
  const right = landmarkPoint(lm, RIGHT_FACE, w, 1);
  const midX = (left.x + right.x) / 2;
  const faceW = Math.abs(right.x - left.x);
  if (faceW < 1e-6) return 0;
  return (nose.x - midX) / faceW;
}

function pitchNormFromLandmarks(lm) {
  const nose = lm[NOSE_TIP];
  const le = lm[33];
  const re = lm[263];
  const eyeMidY = (le.y + re.y) / 2;
  const faceH = Math.abs(lm[CHIN].y - lm[FOREHEAD].y) + 1e-6;
  return (nose.y - eyeMidY) / faceH;
}

/**
 * @returns {{ ratio: number; dNoseEye: number; dChinNose: number; faceH: number }}
 */
function frontalNoseChinEyeMetrics(lm) {
  const nose = lm[NOSE_TIP];
  const le = lm[33];
  const re = lm[263];
  const chin = lm[CHIN];
  const eyeMidY = (le.y + re.y) / 2;
  const faceH = Math.abs(lm[CHIN].y - lm[FOREHEAD].y) + 1e-6;
  const dNoseEye = nose.y - eyeMidY;
  const dChinNose = chin.y - nose.y;
  const ratio = dNoseEye > 1e-8 ? dChinNose / dNoseEye : 99;
  return { ratio, dNoseEye, dChinNose, faceH };
}

function updateFaceMotionMean(videoEl, box) {
  const W = 96;
  analyzeCanvas.width = W;
  analyzeCanvas.height = W;
  const { sx, sy, sw, sh } = box;
  if (sw < 12 || sh < 12) return { meanDiff: MOTION_MIN + 1 };
  actx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, W, W);
  const img = actx.getImageData(0, 0, W, W);
  const curr = new Uint8Array(W * W);
  for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
    curr[j] = 0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
  }
  let meanDiff = MOTION_MIN + 2;
  if (prevFaceGray) {
    let s = 0;
    for (let i = 0; i < curr.length; i++) s += Math.abs(curr[i] - prevFaceGray[i]);
    meanDiff = s / curr.length;
  }
  prevFaceGray = curr;
  return { meanDiff };
}

function padLooksBad(padResult) {
  return !!(
    padResult.printLike ||
    padResult.screenLike ||
    padResult.videoBlockLike ||
    padResult.maskSuspect
  );
}

function updateCoverage(mirroredYaw, pitchN) {
  if (Math.abs(mirroredYaw) < YAW_CENTER) coverage.center = true;
  if (mirroredYaw > YAW_SIDE) coverage.left = true;
  if (mirroredYaw < -YAW_SIDE) coverage.right = true;
  if (pitchN < PITCH_UP) coverage.up = true;
  if (pitchN > PITCH_DOWN) coverage.down = true;
}

function coveragePercent() {
  let n = 0;
  for (const k of COVERAGE_KEYS) {
    if (coverage[k]) n++;
  }
  return Math.round((n / COVERAGE_KEYS.length) * 100);
}

function coverageMissingCount() {
  let n = 0;
  for (const k of COVERAGE_KEYS) {
    if (!coverage[k]) n++;
  }
  return n;
}

function resetCoverage() {
  for (const k of COVERAGE_KEYS) coverage[k] = false;
}

function resetCaptureFlow() {
  capturedDone = false;
  awaitingNeutralCapture = false;
  neutralHoldFrames = 0;
  neutralPhaseStartedAt = 0;
  bestFrontal = { score: Infinity, box: null };
  bestFrontalCanvas = null;
  if (capturePreviewImg) {
    capturePreviewImg.removeAttribute('src');
    capturePreviewImg.hidden = true;
  }
  if (capResult) {
    capResult.textContent = '';
    capResult.className = 'result';
  }
}

function abortIfSpoof() {
  resetCaptureFlow();
  resetCoverage();
}

function snapshotLandmarks(lm) {
  return lm.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0 }));
}

/**
 * Spoof ระหว่างกำลังจับภาพ — รีเซ็ตโฟลว์ + หยุดวิดีโอ + ค้าง overlay เฟรมนี้
 */
function abortSpoofCapture(lm, w, h, nowMs) {
  spoofFreeze = { lm: snapshotLandmarks(lm), w, h, ts: nowMs };
  resetCaptureFlow();
  resetCoverage();
  displayMode = 'spoof';
  spoofStreak = SMOOTH_FRAMES;
  realStreak = 0;
  smoothProgressPct = 0;
  progressLerpLastTs = nowMs;
  video.pause().catch(() => {});
}

function clearSpoofFreeze() {
  spoofFreeze = null;
  if (video.srcObject && stream) {
    video.play().catch(() => {});
  }
}

function frontalPitchPenalty(pitchN) {
  if (pitchN < FRONTAL_PITCH_MIN) {
    return (FRONTAL_PITCH_MIN - pitchN) * 2.4;
  }
  if (pitchN > FRONTAL_PITCH_MAX) {
    return (pitchN - FRONTAL_PITCH_MAX) * 2.4;
  }
  const mid = (FRONTAL_PITCH_MIN + FRONTAL_PITCH_MAX) / 2;
  return Math.abs(pitchN - mid) * 0.4;
}

function frontalRatioPenalty(lm) {
  const { ratio } = frontalNoseChinEyeMetrics(lm);
  if (ratio <= FRONTAL_NC_TO_NE_RATIO_MAX) return 0;
  return (ratio - FRONTAL_NC_TO_NE_RATIO_MAX) * 0.42;
}

function frontalPoseScore(mirroredYaw, pitchN, lm) {
  return (
    Math.abs(mirroredYaw) +
    frontalPitchPenalty(pitchN) +
    frontalRatioPenalty(lm)
  );
}

/**
 * @param {{ ratio: number; dNoseEye: number; dChinNose: number; faceH: number }} [metrics]
 */
function isFrontalPose(mirroredYaw, pitchN, lm, metrics) {
  const m = metrics ?? frontalNoseChinEyeMetrics(lm);
  if (m.ratio > FRONTAL_NC_TO_NE_RATIO_MAX) return false;
  if (m.dChinNose < m.faceH * 0.085) return false;
  return (
    Math.abs(mirroredYaw) <= FRONTAL_YAW_MAX &&
    pitchN >= FRONTAL_PITCH_MIN &&
    pitchN <= FRONTAL_PITCH_MAX
  );
}

/**
 * @param {{ sx: number; sy: number; sw: number; sh: number }} box
 * @param {{ fallbackBest?: boolean }} [opts]
 */
function performCapture(box, opts = {}) {
  const capCanvas = document.createElement('canvas');
  const cw = Math.min(512, box.sw);
  const ch = Math.min(512, box.sh);
  capCanvas.width = cw;
  capCanvas.height = ch;
  const cctx = capCanvas.getContext('2d');
  cctx.drawImage(video, box.sx, box.sy, box.sw, box.sh, 0, 0, cw, ch);
  const dataUrl = capCanvas.toDataURL('image/jpeg', 0.92);
  if (capturePreviewImg) {
    capturePreviewImg.src = dataUrl;
    capturePreviewImg.hidden = false;
  }
  if (capResult) {
    capResult.textContent = opts.fallbackBest
      ? 'จับภาพแล้ว — ใช้เฟรมที่ใกล้หน้าตรงที่สุดขณะรอ (หากหันหน้าตรงไม่ทันเวลา) — แสดงในเบราว์เซอร์อย่างเดียว'
      : 'จับภาพหน้าตรงแล้ว (แสดงในเบราว์เซอร์เท่านั้น — ยังไม่ส่งเซิร์ฟเวอร์)';
    capResult.className = 'result ok';
  }
  capHint.textContent = 'เสร็จสิ้น — คุณสามารถหยุดหรือเริ่มใหม่ได้';
  capRotationHint.textContent = '';
  capturedDone = true;
  awaitingNeutralCapture = false;
  neutralHoldFrames = 0;
  bestFrontalCanvas = null;
}

/**
 * @param {{ fallbackBest?: boolean }} [opts]
 */
function performCaptureFromBestCanvas(opts = {}) {
  if (!bestFrontalCanvas) return;
  const dataUrl = bestFrontalCanvas.toDataURL('image/jpeg', 0.92);
  if (capturePreviewImg) {
    capturePreviewImg.src = dataUrl;
    capturePreviewImg.hidden = false;
  }
  if (capResult) {
    capResult.textContent = opts.fallbackBest
      ? 'จับภาพแล้ว — ใช้เฟรมที่ใกล้หน้าตรงที่สุดขณะรอ (หากหันหน้าตรงไม่ทันเวลา) — แสดงในเบราว์เซอร์อย่างเดียว'
      : 'จับภาพหน้าตรงแล้ว (แสดงในเบราว์เซอร์เท่านั้น — ยังไม่ส่งเซิร์ฟเวอร์)';
    capResult.className = 'result ok';
  }
  capHint.textContent = 'เสร็จสิ้น — คุณสามารถหยุดหรือเริ่มใหม่ได้';
  capRotationHint.textContent = '';
  capturedDone = true;
  awaitingNeutralCapture = false;
  neutralHoldFrames = 0;
  bestFrontalCanvas = null;
}

function resizeCanvasCap() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
}

/**
 * @param {number} targetPct 0–100 ความครบมุม (เฉพาะโหมด real)
 * @param {number} nowMs performance.now()
 */
function drawFaceOverlay(w, h, lm, mode, label, targetPct, nowMs) {
  ctx.save();
  ctx.clearRect(0, 0, w, h);

  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  for (let i = 0; i < lm.length; i++) {
    const p = lm[i];
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const cx = ((minX + maxX) / 2) * w;
  const cy = ((minY + maxY) / 2) * h;
  const rx = ((maxX - minX) * w) / 2 + 0.04 * w;
  const ry = ((maxY - minY) * h) / 2 + 0.05 * h;
  const rFace = Math.max(rx, ry);

  const isReal = mode === 'real';
  const goal = isReal ? (capturedDone ? 100 : targetPct) : 0;
  const dt = progressLerpLastTs ? Math.min(64, nowMs - progressLerpLastTs) : 16;
  progressLerpLastTs = nowMs;
  const t = 1 - Math.pow(PROGRESS_LERP_BASE, dt / 16);
  smoothProgressPct += (goal - smoothProgressPct) * t;
  if (smoothProgressPct < 0.08) smoothProgressPct = 0;
  if (smoothProgressPct > 99.95) smoothProgressPct = 100;

  const rRing = rFace + Math.max(18, Math.min(28, w * 0.022));
  const pulse = isReal ? 1 + Math.sin(nowMs * 0.0035) * 0.04 : 1;

  ctx.lineWidth = isReal ? 5 : 6;
  ctx.strokeStyle = isReal ? 'rgba(255,255,255,0.16)' : 'rgba(255, 160, 160, 0.45)';
  ctx.beginPath();
  ctx.arc(cx, cy, rRing * pulse, 0, Math.PI * 2);
  ctx.stroke();

  if (smoothProgressPct > 0.35 && isReal) {
    const start = -Math.PI / 2;
    const sweep = (smoothProgressPct / 100) * Math.PI * 2;
    ctx.lineWidth = 6.5;
    ctx.lineCap = 'round';
    const gx0 = cx - rRing;
    const gy0 = cy - rRing;
    const grad = ctx.createLinearGradient(gx0, gy0, cx + rRing, cy + rRing);
    grad.addColorStop(0, '#1a8f82');
    grad.addColorStop(0.45, '#3ecf8e');
    grad.addColorStop(1, '#7ef5c5');
    ctx.strokeStyle = grad;
    ctx.shadowColor = 'rgba(62, 207, 142, 0.45)';
    ctx.shadowBlur = 18;
    ctx.beginPath();
    ctx.arc(cx, cy, rRing * pulse, start, start + sweep);
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (sweep > 0.12) {
      const ax = cx + rRing * pulse * Math.cos(start + sweep);
      const ay = cy + rRing * pulse * Math.sin(start + sweep);
      const capG = ctx.createRadialGradient(ax, ay, 0, ax, ay, 10);
      capG.addColorStop(0, 'rgba(255,255,255,0.95)');
      capG.addColorStop(0.5, 'rgba(126, 245, 197, 0.85)');
      capG.addColorStop(1, 'rgba(62, 207, 142, 0)');
      ctx.fillStyle = capG;
      ctx.beginPath();
      ctx.arc(ax, ay, 9, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.lineWidth = isReal ? 4 : 6;
  ctx.strokeStyle = isReal ? 'rgba(62, 207, 142, 0.95)' : 'rgba(255, 55, 55, 1)';
  ctx.shadowColor = isReal ? 'transparent' : 'rgba(255, 40, 40, 0.55)';
  ctx.shadowBlur = isReal ? 0 : 10;
  ctx.beginPath();
  ctx.arc(cx, cy, rFace, 0, Math.PI * 2);
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.font = 'bold 22px Sarabun, Segoe UI, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = isReal ? 'rgba(62, 207, 142, 0.98)' : 'rgba(255, 230, 230, 1)';
  ctx.shadowColor = 'rgba(0,0,0,0.85)';
  ctx.shadowBlur = isReal ? 6 : 8;
  ctx.fillText(label, cx, cy - rFace - 22);
  ctx.shadowBlur = 0;

  if (isReal && smoothProgressPct > 2) {
    ctx.font = '600 18px Sarabun, Segoe UI, sans-serif';
    ctx.fillStyle = 'rgba(200, 245, 220, 0.95)';
    ctx.fillText(Math.round(smoothProgressPct) + '%', cx, cy + rFace + 26);
  }

  ctx.restore();
}

async function loadModel() {
  capModelStatus.textContent = 'กำลังโหลดโมเดล…';
  const fileset = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm',
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
  capModelStatus.textContent = 'โมเดลพร้อม';
}

async function getUserMediaStream(constraints) {
  if (navigator.mediaDevices?.getUserMedia) {
    return navigator.mediaDevices.getUserMedia(constraints);
  }
  throw new Error('ไม่มี getUserMedia');
}

async function startCamera() {
  stream = await getUserMediaStream({
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(rafId);
  prevFaceGray = null;
  smoothProgressPct = 0;
  progressLerpLastTs = 0;
  clearSpoofFreeze();
  resetCaptureFlow();
  resetCoverage();
  spoofStreak = 0;
  realStreak = 0;
  padSpoofStreak = 0;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  video.load();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function loop() {
  if (!running || !faceLandmarker) return;

  const now = performance.now();
  if (video.readyState >= 2) {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w && h) {
      if (spoofFreeze) {
        resizeCanvasCap();
        const { lm: flm, w: fw, h: fh, ts } = spoofFreeze;
        drawFaceOverlay(fw, fh, flm, 'spoof', 'Spoof', 0, ts);
        capHint.textContent =
          'ตรวจพบ Spoof — หยุดการจับภาพแล้ว (เฟรมนี้ถูกหยุดไว้) — กด «เริ่มกล้อง» เพื่อลองใหม่';
        capRotationHint.textContent =
          'หลีกเลี่ยงรูปพิมพ์/จอ/วิดีโอ หรือถือกล้องให้นิ่งหลังใบหน้าจริง';
        capHud.textContent = '';
        rafId = requestAnimationFrame(loop);
        return;
      }

      const results = faceLandmarker.detectForVideo(video, now);
      resizeCanvasCap();

      if (!results.faceLandmarks?.length) {
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.restore();
        capHint.textContent = 'ไม่พบใบหน้า — ให้อยู่ในกรอบ';
        capHud.textContent = '';
        rafId = requestAnimationFrame(loop);
        return;
      }

      const lm = results.faceLandmarks[0];
      const box = faceBoundingBoxPixels(lm, w, h);
      const { meanDiff } = updateFaceMotionMean(video, box);

      const ps = getPadCanvasSize();
      padCanvas.width = ps;
      padCanvas.height = ps;
      padCtx.drawImage(video, box.sx, box.sy, box.sw, box.sh, 0, 0, ps, ps);
      const padImg = padCtx.getImageData(0, 0, ps, ps);
      const padResult = analyzePad(padImg);

      const covPctEarly = coveragePercent();
      if (padLooksBad(padResult)) padSpoofStreak++;
      else padSpoofStreak = 0;
      const padTriggersSpoof = padSpoofStreak >= CAPTURE_PAD_CONSECUTIVE;
      const spoofFrame = padTriggersSpoof;
      const padBad = padLooksBad(padResult);

      const captureActive =
        awaitingNeutralCapture || (!capturedDone && covPctEarly > 0);

      if (spoofFrame && captureActive) {
        abortSpoofCapture(lm, w, h, now);
        drawFaceOverlay(w, h, lm, 'spoof', 'Spoof', 0, now);
        capHint.textContent =
          'ตรวจพบ Spoof — หยุดการจับภาพแล้ว (เฟรมนี้ถูกหยุดไว้) — กด «เริ่มกล้อง» เพื่อลองใหม่';
        capRotationHint.textContent =
          'หลีกเลี่ยงรูปพิมพ์/จอ/วิดีโอ หรือถือกล้องให้นิ่งหลังใบหน้าจริง';
        if (rotationPercent) rotationPercent.textContent = '0%';
        capHud.textContent =
          'PAD+' +
          padSpoofStreak +
          '/' +
          CAPTURE_PAD_CONSECUTIVE +
          ' | m' +
          meanDiff.toFixed(1);
        rafId = requestAnimationFrame(loop);
        return;
      }

      if (spoofFrame) {
        spoofStreak++;
        realStreak = 0;
      } else {
        realStreak++;
        spoofStreak = 0;
      }
      if (realStreak >= SMOOTH_FRAMES) displayMode = 'real';
      else if (spoofStreak >= SMOOTH_FRAMES) displayMode = 'spoof';

      if (displayMode === 'spoof' && !captureActive) {
        abortIfSpoof();
      }

      const yaw = yawFromLandmarks(lm, w);
      const mirroredYaw = -yaw;
      const pitchN = pitchNormFromLandmarks(lm);
      const frontalM = frontalNoseChinEyeMetrics(lm);

      if (displayMode === 'real' && !capturedDone && !spoofFrame) {
        updateCoverage(mirroredYaw, pitchN);
      }

      const covPct = coveragePercent();

      if (
        displayMode === 'real' &&
        !capturedDone &&
        covPct >= 100 &&
        !spoofFrame &&
        isFrontalPose(mirroredYaw, pitchN, lm, frontalM)
      ) {
        const ps = frontalPoseScore(mirroredYaw, pitchN, lm);
        if (ps < bestFrontal.score) {
          bestFrontal = { score: ps, box: { ...box } };
          const bw = Math.min(512, box.sw);
          const bh = Math.min(512, box.sh);
          if (!bestFrontalCanvas) bestFrontalCanvas = document.createElement('canvas');
          bestFrontalCanvas.width = bw;
          bestFrontalCanvas.height = bh;
          const bx = bestFrontalCanvas.getContext('2d');
          if (bx) bx.drawImage(video, box.sx, box.sy, box.sw, box.sh, 0, 0, bw, bh);
        }
      }

      if (
        displayMode === 'real' &&
        !capturedDone &&
        covPct >= 100 &&
        !awaitingNeutralCapture &&
        !spoofFrame
      ) {
        awaitingNeutralCapture = true;
        neutralPhaseStartedAt = now;
        neutralHoldFrames = 0;
      }

      /** วาดวงแดงทันทีเมื่อเฟรมนี้ถือว่า spoof — ไม่ต้องรอ smooth spoof 5 เฟรม */
      const overlayIsSpoof = displayMode === 'spoof' || spoofFrame;
      const overlayMode = overlayIsSpoof ? 'spoof' : 'real';
      const targetPct =
        displayMode === 'real' && !overlayIsSpoof ? covPct : 0;
      const label = overlayIsSpoof ? 'Spoof' : 'Real';
      drawFaceOverlay(w, h, lm, overlayMode, label, targetPct, now);

      const bar = rotationProgressFill?.closest('.rotation-progress-bar');
      if (bar) bar.setAttribute('aria-valuenow', String(Math.round(smoothProgressPct)));
      if (rotationPercent) {
        rotationPercent.textContent =
          (displayMode === 'real' && !overlayIsSpoof
            ? Math.round(smoothProgressPct)
            : 0) + '%';
      }

      if (displayMode === 'real' && !capturedDone && awaitingNeutralCapture) {
        const frontalOk = isFrontalPose(mirroredYaw, pitchN, lm, frontalM);
        if (frontalOk) neutralHoldFrames++;
        else neutralHoldFrames = 0;

        if (neutralHoldFrames >= FRONTAL_HOLD_FRAMES && frontalOk) {
          performCapture(box);
        } else if (
          now - neutralPhaseStartedAt > FRONTAL_WAIT_TIMEOUT_MS &&
          bestFrontalCanvas
        ) {
          performCaptureFromBestCanvas({ fallbackBest: true });
        }

        capHint.textContent = 'ครบทุกมุมแล้ว — กลับมาหน้าตรงเพื่อจับภาพ';
        capRotationHint.textContent = frontalOk
          ? 'ถือหน้าตรงนิ่งๆ (' + neutralHoldFrames + '/' + FRONTAL_HOLD_FRAMES + ')'
          : frontalM.ratio > FRONTAL_NC_TO_NE_RATIO_MAX
            ? 'เงยมองเกินไป — ก้มคางลงเล็กน้อยให้จมูกห่างจากตา'
            : 'หันหน้าให้ตรงกลางกล้อง (ไม่เงย / ไม่ก้มมาก)';
      } else if (displayMode === 'real' && !capturedDone) {
        capHint.textContent =
          'ใบหน้าจริง — หมุนหน้าให้ครบทุกมุม (ตรง / ซ้าย / ขวา / เงย / ก้ม)';
        const missing = COVERAGE_KEYS.filter((k) => !coverage[k]);
        capRotationHint.textContent =
          missing.length === 0
            ? 'ครบมุมแล้ว — เตรียมหน้าตรง…'
            : 'ยังขาด: ' + missing.map((k) => COVERAGE_LABEL_TH[k]).join(' · ');
      } else if (overlayIsSpoof) {
        capHint.textContent =
          'สถานะ Spoof — ระบบยังไม่ยืนยันว่าเป็นหน้าจริงหน้ากล้อง (ดูค่า L/gV/ch ใน HUD)';
        capRotationHint.textContent = padTriggersSpoof
          ? 'PAD ติดเงื่อนไขหลายเฟรมติดกัน — ลองแสงสว่างขึ้น ถอยจากกล้อง หรือหลีกเลี่ยงแสงสะท้อน'
          : 'รอสถานะนิ่งหลายเฟรม — หลีกเลี่ยงรูปพิมพ์/หน้าจอ';
      }

      capHud.textContent =
        'p=pitch r=(คาง−จมูก)/(จมูก−ตา) | L' +
        padResult.debug.lapVar +
        ' gV' +
        padResult.debug.grayVar +
        ' ch' +
        padResult.debug.chromaSp +
        ' | m' +
        meanDiff.toFixed(1) +
        ' | y' +
        mirroredYaw.toFixed(2) +
        ' p' +
        pitchN.toFixed(2) +
        ' r' +
        frontalM.ratio.toFixed(2);

      rafId = requestAnimationFrame(loop);
      return;
    }
  }
  rafId = requestAnimationFrame(loop);
}

export function initCaptureMode() {
  if (capListenersBound) return;
  capListenersBound = true;
  btnStart?.addEventListener('click', onCapStart);
  btnStop?.addEventListener('click', onCapStop);
}

function onCapStart() {
  if (!btnStart || !btnStop) return;
  clearSpoofFreeze();
  padSpoofStreak = 0;
  resetCaptureFlow();
  resetCoverage();
  smoothProgressPct = 0;
  progressLerpLastTs = 0;
  if (rotationPercent) rotationPercent.textContent = '0%';
  displayMode = 'spoof';
  spoofStreak = 0;
  realStreak = 0;
  prevFaceGray = null;

  (async () => {
    try {
      if (!faceLandmarker) await loadModel();
      await startCamera();
      running = true;
      btnStart.disabled = true;
      btnStop.disabled = false;
      capHint.textContent = 'กำลังตรวจ…';
      requestAnimationFrame(loop);
    } catch (e) {
      console.error(e);
      capModelStatus.textContent = String(e?.message || e);
      capHint.textContent = 'เปิดกล้องไม่ได้';
    }
  })();
}

function onCapStop() {
  stopCamera();
  if (btnStart) btnStart.disabled = false;
  if (btnStop) btnStop.disabled = true;
  capHint.textContent = 'กดเริ่มกล้องเพื่อลองใหม่';
  capRotationHint.textContent = '';
  capHud.textContent = '';
}

export function destroyCaptureMode() {
  stopCamera();
  if (btnStart) btnStart.disabled = false;
  if (btnStop) btnStop.disabled = true;
  if (capListenersBound) {
    btnStart?.removeEventListener('click', onCapStart);
    btnStop?.removeEventListener('click', onCapStop);
    capListenersBound = false;
  }
}
