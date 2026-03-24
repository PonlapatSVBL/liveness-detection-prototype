import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 159];
const LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380];
const NOSE_TIP = 1;
const LEFT_FACE = 234;
const RIGHT_FACE = 454;

const video = document.querySelector('#video');
const canvas = document.querySelector('#overlay');
const ctx = canvas.getContext('2d');
const btnStart = document.querySelector('#btnStart');
const btnStop = document.querySelector('#btnStop');
const btnFileTest = document.querySelector('#btnFileTest');
const fileVideo = document.querySelector('#fileVideo');
const modelStatus = document.querySelector('#modelStatus');
const challengeText = document.querySelector('#challengeText');
const challengeHint = document.querySelector('#challengeHint');
const progressEl = document.querySelector('#progress');
const resultEl = document.querySelector('#result');
const hud = document.querySelector('#hud');

/** @type {FaceLandmarker | null} */
let faceLandmarker = null;
let stream = null;
/** @type {string | null} */
let fileObjectUrl = null;
let running = false;
let rafId = 0;

function revokeFileUrl() {
  if (fileObjectUrl) {
    URL.revokeObjectURL(fileObjectUrl);
    fileObjectUrl = null;
  }
}

const BLINK_EAR_THRESHOLD = 0.21;
const BLINK_REQUIRED = 2;
const BLINK_MIN_GAP_MS = 220;
const BLINK_MAX_STEP_MS = 12000;

const YAW_NEUTRAL = 0.08;
const YAW_TURN = 0.14;
const YAW_HOLD_MS = 450;

const MIN_FACE_WIDTH_NORM = 0.14;

/** การเคลื่อนไหวเฉลี่ยระดับเทา (0–255) บนพatche ใบหน้า — ต่ำต่อเนื่อง = ภาพนิ่ง */
const ANALYZE_SIZE = 96;
const MOTION_MIN = 0.9;
const LOW_MOTION_FAIL_MS = 3000;
const SESSION_WARMUP_MS = 1600;

/** ระหว่างตาหย่อน ต้องมี blendshape กระพริบถึงเกณฑ์ (โมเดลมองเห็นการหลับตา 3D) */
const MIN_BLINK_BLENDSHAPE = 0.12;

const SMILE_SCORE = 0.34;
const SMILE_HOLD_MS = 550;
const SMILE_MAX_STEP_MS = 14000;

const ChallengeType = {
  BLINK: 'blink',
  SMILE: 'smile',
  HEAD_LEFT: 'head_left',
  HEAD_RIGHT: 'head_right',
};

const analyzeCanvas = document.createElement('canvas');
const actx = analyzeCanvas.getContext('2d', { willReadFrequently: true });

/** @type {Uint8Array | null} */
let prevFaceGray = null;
let lowMotionMs = 0;
let prevAntiSpoofT = 0;
let liveSessionStartedAt = 0;
let blinkMaxBlendDuringClose = 0;

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function buildChallengeSequence() {
  const base = [
    ChallengeType.BLINK,
    ChallengeType.SMILE,
    ChallengeType.HEAD_LEFT,
    ChallengeType.HEAD_RIGHT,
  ];
  return shuffle(base);
}

function resetAntiSpoof() {
  prevFaceGray = null;
  lowMotionMs = 0;
  prevAntiSpoofT = 0;
  blinkMaxBlendDuringClose = 0;
  liveSessionStartedAt = performance.now();
}

/** @type {{ type: string, steps: string[], stepIndex: number, blinkCount: number, lastBlinkAt: number, stepStarted: number, yawHoldStart: number | null, lastYawOk: boolean }} */
const session = {
  type: '',
  steps: [],
  stepIndex: 0,
  blinkCount: 0,
  lastBlinkAt: 0,
  stepStarted: 0,
  yawHoldStart: null,
  lastYawOk: false,
};

function landmarkPoint(lm, idx, w, h) {
  const p = lm[idx];
  return { x: p.x * w, y: p.y * h, z: p.z * w };
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function eyeAspectRatio(lm, indices, w, h) {
  const p = indices.map((i) => landmarkPoint(lm, i, w, h));
  const [p1, p2, p3, p4, p5, p6] = p;
  const ver1 = dist(p2, p6);
  const ver2 = dist(p3, p5);
  const hor = dist(p1, p4);
  if (hor < 1e-6) return 1;
  return (ver1 + ver2) / (2 * hor);
}

function meanEAR(lm, w, h) {
  return (eyeAspectRatio(lm, RIGHT_EYE_IDX, w, h) + eyeAspectRatio(lm, LEFT_EYE_IDX, w, h)) / 2;
}

function yawFromLandmarks(lm, w, _h) {
  const nose = landmarkPoint(lm, NOSE_TIP, w, 1);
  const left = landmarkPoint(lm, LEFT_FACE, w, 1);
  const right = landmarkPoint(lm, RIGHT_FACE, w, 1);
  const midX = (left.x + right.x) / 2;
  const faceW = Math.abs(right.x - left.x);
  if (faceW < 1e-6) return 0;
  return (nose.x - midX) / faceW;
}

function faceWidthNorm(lm, w) {
  const left = landmarkPoint(lm, LEFT_FACE, w, 1);
  const right = landmarkPoint(lm, RIGHT_FACE, w, 1);
  return Math.abs(right.x - left.x) / w;
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

/**
 * @param {{ faceBlendshapes?: { categories: { categoryName: string; score: number }[] }[] }} results
 * @param {number} faceIdx
 */
function blendshapeMap(results, faceIdx = 0) {
  const cats = results.faceBlendshapes?.[faceIdx]?.categories;
  if (!cats?.length) return null;
  /** @type {Record<string, number>} */
  const m = {};
  for (const c of cats) {
    m[c.categoryName] = c.score;
  }
  return m;
}

/**
 * @returns {{ meanDiff: number }}
 */
function updateFaceMotionMean(videoEl, box) {
  const W = ANALYZE_SIZE;
  analyzeCanvas.width = W;
  analyzeCanvas.height = W;
  const { sx, sy, sw, sh } = box;
  if (sw < 12 || sh < 12) {
    return { meanDiff: MOTION_MIN + 1 };
  }
  actx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, W, W);
  const img = actx.getImageData(0, 0, W, W);
  const curr = new Uint8Array(W * W);
  for (let i = 0, j = 0; i < img.data.length; i += 4, j++) {
    curr[j] =
      0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2];
  }
  let meanDiff = MOTION_MIN + 2;
  if (prevFaceGray) {
    let s = 0;
    for (let i = 0; i < curr.length; i++) {
      s += Math.abs(curr[i] - prevFaceGray[i]);
    }
    meanDiff = s / curr.length;
  }
  prevFaceGray = curr;
  return { meanDiff };
}

function tickAntiSpoof(now, motionMean) {
  if (!running || !session.type) return;
  if (now - liveSessionStartedAt < SESSION_WARMUP_MS) {
    lowMotionMs = 0;
    prevAntiSpoofT = now;
    return;
  }
  const dt = prevAntiSpoofT ? Math.min(130, now - prevAntiSpoofT) : 16;
  prevAntiSpoofT = now;
  if (motionMean < MOTION_MIN) lowMotionMs += dt;
  else lowMotionMs = Math.max(0, lowMotionMs - dt * 0.85);
  if (lowMotionMs >= LOW_MOTION_FAIL_MS) {
    failSession(
      'ไม่พบการเคลื่อนไหวบนใบหน้าอย่างต่อเนื่อง — อาจถือรูปนิ่ง/หน้าจอนิ่ง หรืออยู่ไกลเกินไป',
    );
  }
}

function challengeLabel(type) {
  switch (type) {
    case ChallengeType.BLINK:
      return 'กระพริบตา ' + BLINK_REQUIRED + ' ครั้ง';
    case ChallengeType.SMILE:
      return 'ยิ้มให้เห็นฟันหรือแก้มยกชัดๆ ค้างสักครู่';
    case ChallengeType.HEAD_LEFT:
      return 'หันหน้าไปทางซ้ายของคุณ (ดูจากในกล้อง)';
    case ChallengeType.HEAD_RIGHT:
      return 'หันหน้าไปทางขวาของคุณ (ดูจากในกล้อง)';
    default:
      return '';
  }
}

function challengeHintText(type) {
  switch (type) {
    case ChallengeType.BLINK:
      return 'กระพริบให้ชัด — ต้องเป็นหน้าจริงหน้ากล้อง (ระบบตรวจทั้งรูปตาและสัญญาณ 3D จากโมเดล)';
    case ChallengeType.SMILE:
      return 'ยิ้มจริงๆ — รูปนิ่งมักไม่สร้างรูปแบบยิ้มที่โมเดลยอมรับ';
    case ChallengeType.HEAD_LEFT:
      return 'หันค่อยๆ ค้างสักครู่จนเห็นวงขอบเขียว';
    case ChallengeType.HEAD_RIGHT:
      return 'หันค่อยๆ ค้างสักครู่จนเห็นวงขอบเขียว';
    default:
      return '';
  }
}

function resetSession() {
  session.steps = [];
  session.stepIndex = 0;
  session.type = '';
  session.blinkCount = 0;
  session.lastBlinkAt = 0;
  session.stepStarted = 0;
  session.yawHoldStart = null;
  session.lastYawOk = false;
}

function startNewSequence() {
  session.steps = buildChallengeSequence();
  session.stepIndex = 0;
  session.type = session.steps[0];
  session.blinkCount = 0;
  session.lastBlinkAt = 0;
  session.stepStarted = performance.now();
  session.yawHoldStart = null;
  session.lastYawOk = false;
  blinkState.below = false;
  blinkMaxBlendDuringClose = 0;
  challengeText.textContent = challengeLabel(session.type);
  challengeHint.textContent = challengeHintText(session.type);
  progressEl.textContent = 'ขั้นตอน 1 / ' + session.steps.length;
  resultEl.textContent = '';
  resultEl.className = 'result';
}

function advanceStep() {
  session.stepIndex++;
  if (session.stepIndex >= session.steps.length) {
    session.type = '';
    challengeText.textContent = 'ผ่านการตรวจสอบ';
    challengeHint.textContent = 'คุณสามารถนำ token/ผลลัพธ์นี้ไปยืนยันต่อในฝั่งเซิร์ฟเวอร์ได้ (โปรโตไทป์นี้ยังไม่มี backend)';
    progressEl.textContent = 'เสร็จสิ้น';
    resultEl.textContent = 'ผลลัพธ์: LIVENESS_PASS (จำลอง — ประมวลผลบนอุปกรณ์)';
    resultEl.className = 'result ok';
    running = false;
    hud.textContent = 'สถานะ: สำเร็จ';
    return;
  }
  session.type = session.steps[session.stepIndex];
  session.blinkCount = 0;
  session.lastBlinkAt = 0;
  session.stepStarted = performance.now();
  session.yawHoldStart = null;
  session.lastYawOk = false;
  blinkState.below = false;
  blinkMaxBlendDuringClose = 0;
  challengeText.textContent = challengeLabel(session.type);
  challengeHint.textContent = challengeHintText(session.type);
  progressEl.textContent = 'ขั้นตอน ' + (session.stepIndex + 1) + ' / ' + session.steps.length;
}

function failSession(reason) {
  running = false;
  session.type = '';
  challengeText.textContent = 'ไม่ผ่าน — ลองใหม่';
  challengeHint.textContent = reason;
  resultEl.textContent = 'ผลลัพธ์: LIVENESS_FAIL — ' + reason;
  resultEl.className = 'result fail';
  hud.textContent = 'สถานะ: ล้มเหลว';
  btnStart.disabled = false;
  btnFileTest.disabled = false;
}

/** @type {{ below: boolean }} */
const blinkState = { below: false };

/**
 * @param {Record<string, number> | null} scores
 */
function processBlinkStep(now, ear, scores) {
  if (now - session.stepStarted > BLINK_MAX_STEP_MS) {
    failSession('หมดเวลาสำหรับการกระพริบตา');
    return;
  }
  if (ear < BLINK_EAR_THRESHOLD) {
    if (!blinkState.below) {
      blinkState.below = true;
    }
    if (scores) {
      const bl = Math.max(scores.eyeBlinkLeft ?? 0, scores.eyeBlinkRight ?? 0);
      blinkMaxBlendDuringClose = Math.max(blinkMaxBlendDuringClose, bl);
    }
  } else if (blinkState.below) {
    blinkState.below = false;
    if (scores && blinkMaxBlendDuringClose < MIN_BLINK_BLENDSHAPE) {
      blinkMaxBlendDuringClose = 0;
      return;
    }
    blinkMaxBlendDuringClose = 0;
    if (now - session.lastBlinkAt >= BLINK_MIN_GAP_MS) {
      session.blinkCount++;
      session.lastBlinkAt = now;
      if (session.blinkCount >= BLINK_REQUIRED) {
        advanceStep();
      }
    }
  }
}

/**
 * @param {Record<string, number> | null} scores
 */
function processSmileStep(now, scores) {
  if (now - session.stepStarted > SMILE_MAX_STEP_MS) {
    failSession('หมดเวลาสำหรับการยิ้ม — ลองยิ้มให้ชัดขึ้น');
    return;
  }
  if (!scores) return;
  const sm = ((scores.mouthSmileLeft ?? 0) + (scores.mouthSmileRight ?? 0)) / 2;
  const ok = sm > SMILE_SCORE;
  if (ok) {
    if (session.yawHoldStart === null) session.yawHoldStart = now;
    else if (now - session.yawHoldStart >= SMILE_HOLD_MS) advanceStep();
  } else {
    session.yawHoldStart = null;
  }
}

function processHeadStep(now, yaw, wantLeft) {
  const needPositive = wantLeft;
  const ok = needPositive ? yaw > YAW_TURN : yaw < -YAW_TURN;
  const neutral = Math.abs(yaw) < YAW_NEUTRAL;

  if (neutral) {
    session.yawHoldStart = null;
    session.lastYawOk = false;
  }

  if (ok) {
    if (session.yawHoldStart === null) session.yawHoldStart = now;
    else if (now - session.yawHoldStart >= YAW_HOLD_MS) {
      advanceStep();
    }
    session.lastYawOk = true;
  } else {
    if (!session.lastYawOk) session.yawHoldStart = null;
  }

  if (now - session.stepStarted > 15000) {
    failSession('หมดเวลาสำหรับการหันหน้า');
  }
}

function drawGuide(w, h, yaw, stepType) {
  ctx.save();
  ctx.clearRect(0, 0, w, h);
  const cx = w * 0.5;
  const cy = h * 0.42;
  const rx = w * 0.32;
  const ry = h * 0.4;
  ctx.strokeStyle = 'rgba(61, 214, 198, 0.45)';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.stroke();

  if (stepType === ChallengeType.HEAD_LEFT || stepType === ChallengeType.HEAD_RIGHT) {
    const wantLeft = stepType === ChallengeType.HEAD_LEFT;
    const ok = wantLeft ? yaw > YAW_TURN : yaw < -YAW_TURN;
    const hue = ok ? 'rgba(62, 207, 142, 0.85)' : 'rgba(240, 180, 41, 0.9)';
    ctx.strokeStyle = hue;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx - 6, ry - 6, 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

async function loadModel() {
  modelStatus.textContent = 'กำลังโหลดโมเดล (ดาวน์โหลดครั้งแรกอาจใช้เวลาสักครู่)…';
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
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
  });
  modelStatus.textContent = 'โมเดลพร้อม — ประมวลผลบนอุปกรณ์';
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  const w = video.videoWidth || rect.width;
  const h = video.videoHeight || rect.height;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
}

function loop() {
  if (!running || !faceLandmarker) return;

  const now = performance.now();
  if (video.readyState >= 2) {
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (w && h) {
      const results = faceLandmarker.detectForVideo(video, now);
      resizeCanvas();

      if (!results.faceLandmarks || results.faceLandmarks.length === 0) {
        drawGuide(w, h, 0, session.type);
        hud.textContent = 'ไม่พบใบหน้า — ให้อยู่ในกรอบและหันหน้าตรง';
        rafId = requestAnimationFrame(loop);
        return;
      }

      const lm = results.faceLandmarks[0];
      const fw = faceWidthNorm(lm, w);
      if (fw < MIN_FACE_WIDTH_NORM) {
        failSession('ใบหน้าเล็กเกินไป — อาจเป็นภาพเล็กหรืออยู่ไกลเกินไป');
        rafId = requestAnimationFrame(loop);
        return;
      }

      const box = faceBoundingBoxPixels(lm, w, h);
      const { meanDiff } = updateFaceMotionMean(video, box);
      tickAntiSpoof(now, meanDiff);
      if (!running) {
        rafId = requestAnimationFrame(loop);
        return;
      }

      const scores = blendshapeMap(results, 0);
      const ear = meanEAR(lm, w, h);
      const yaw = yawFromLandmarks(lm, w, h);
      const mirroredYaw = -yaw;

      drawGuide(w, h, mirroredYaw, session.type);

      if (session.type === ChallengeType.BLINK) {
        processBlinkStep(now, ear, scores);
        const bhint = scores
          ? Math.max(scores.eyeBlinkLeft ?? 0, scores.eyeBlinkRight ?? 0).toFixed(2)
          : '—';
        hud.textContent =
          'm ε~' +
          meanDiff.toFixed(1) +
          ' | EAR ~' +
          ear.toFixed(2) +
          ' | blinkBS ' +
          bhint +
          ' | กระพริบ ' +
          session.blinkCount +
          '/' +
          BLINK_REQUIRED;
      } else if (session.type === ChallengeType.SMILE) {
        processSmileStep(now, scores);
        const sm = scores
          ? (((scores.mouthSmileLeft ?? 0) + (scores.mouthSmileRight ?? 0)) / 2).toFixed(2)
          : '—';
        hud.textContent = 'm ε~' + meanDiff.toFixed(1) + ' | ยิ้ม~' + sm + ' (เกณฑ์ > ' + SMILE_SCORE + ')';
      } else if (session.type === ChallengeType.HEAD_LEFT) {
        processHeadStep(now, mirroredYaw, true);
        hud.textContent =
          'm ε~' +
          meanDiff.toFixed(1) +
          ' | yaw ~' +
          mirroredYaw.toFixed(2) +
          ' | ค้าง ' +
          Math.round(YAW_HOLD_MS) +
          ' ms';
      } else if (session.type === ChallengeType.HEAD_RIGHT) {
        processHeadStep(now, mirroredYaw, false);
        hud.textContent =
          'm ε~' +
          meanDiff.toFixed(1) +
          ' | yaw ~' +
          mirroredYaw.toFixed(2) +
          ' | ค้าง ' +
          Math.round(YAW_HOLD_MS) +
          ' ms';
      } else {
        hud.textContent = 'เสร็จแล้ว';
      }
    }
  }

  rafId = requestAnimationFrame(loop);
}

/**
 * @param {MediaStreamConstraints} constraints
 * @returns {Promise<MediaStream>}
 */
function explainNoGetUserMedia() {
  const { protocol, hostname, href } = location;
  const isLocalhost =
    hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]' || hostname === '';

  if (protocol === 'file:') {
    return 'อย่าเปิดไฟล์ด้วย file:// — รัน npm run dev แล้วเปิด https://localhost:5173';
  }

  if (protocol === 'http:' && !isLocalhost) {
    return (
      'คุณเปิดผ่าน ' +
      hostname +
      ' บน HTTP — เบราว์เซอร์ส่วนใหญ่ (Chrome) จะไม่เปิด navigator.mediaDevices บนเครือข่ายที่ไม่เข้ารหัส แก้ได้ 2 ทาง: ' +
      '(ก) บนคอมเดียวกันให้ใช้ https://localhost:5173 เท่านั้น ' +
      '(ข) ถ้า dev ใช้ HTTPS แล้ว ให้เปิด ' +
      href.replace(/^http:/, 'https:').replace(/\/$/, '') +
      ' แล้วกดยอมรับใบรับรองชั่วคราว — หรือใช้ปุ่ม «ทดสอบจากไฟล์วิดีโอ»'
    );
  }

  const secure =
    typeof window.isSecureContext === 'boolean'
      ? window.isSecureContext
      : protocol === 'https:' || isLocalhost;

  if (secure) {
    return (
      'บริบทนี้เป็น secure แต่ไม่มี getUserMedia — ลองแท็บ/Chrome ปกติ (ไม่ใช้พรีวิวใน IDE) อนุญาตกล้อง ' +
      'หรือใช้ปุ่ม «ทดสอบจากไฟล์วิดีโอ»'
    );
  }

  return 'เปิดผ่าน https://localhost:5173 (หลังรัน npm run dev — เซิร์ฟเวอร์ใช้ HTTPS อัตโนมัติ) ห้ามใช้ file://';
}

async function getUserMediaStream(constraints) {
  if (navigator.mediaDevices?.getUserMedia) {
    return navigator.mediaDevices.getUserMedia(constraints);
  }
  const legacy =
    navigator.getUserMedia ||
    /** @type {typeof navigator.getUserMedia | undefined} */ (navigator.webkitGetUserMedia) ||
    /** @type {typeof navigator.getUserMedia | undefined} */ (navigator.mozGetUserMedia);
  if (legacy) {
    return new Promise((resolve, reject) => {
      legacy.call(navigator, constraints, resolve, reject);
    });
  }
  throw new Error('ไม่มี getUserMedia ในบริบทนี้ — ' + explainNoGetUserMedia());
}

async function startCamera() {
  revokeFileUrl();
  video.removeAttribute('src');
  stream = await getUserMediaStream({
    video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

/**
 * @param {File} file
 */
async function startWithVideoFile(file) {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  revokeFileUrl();
  fileObjectUrl = URL.createObjectURL(file);
  video.src = fileObjectUrl;
  video.loop = true;
  video.muted = true;
  await new Promise((resolve, reject) => {
    const onOk = () => {
      video.removeEventListener('loadedmetadata', onOk);
      video.removeEventListener('error', onErr);
      resolve();
    };
    const onErr = () => {
      video.removeEventListener('loadedmetadata', onOk);
      video.removeEventListener('error', onErr);
      reject(new Error('โหลดไฟล์วิดีโอไม่สำเร็จ'));
    };
    video.addEventListener('loadedmetadata', onOk);
    video.addEventListener('error', onErr);
  });
  await video.play();
}

function beginSession() {
  resetAntiSpoof();
  resetSession();
  startNewSequence();
  running = true;
  blinkState.below = false;
  btnStart.disabled = true;
  btnFileTest.disabled = true;
  btnStop.disabled = false;
  requestAnimationFrame(loop);
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(rafId);
  prevFaceGray = null;
  lowMotionMs = 0;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  revokeFileUrl();
  video.removeAttribute('src');
  video.load();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

btnStart.addEventListener('click', async () => {
  resultEl.textContent = '';
  resultEl.className = 'result';
  try {
    if (!faceLandmarker) await loadModel();
    await startCamera();
    beginSession();
  } catch (e) {
    console.error(e);
    modelStatus.textContent = 'ข้อผิดพลาด: ' + (e?.message || String(e));
    failSession('ไม่สามารถเปิดกล้องหรือโหลดโมเดลได้');
    btnStart.disabled = false;
    btnFileTest.disabled = false;
    btnStop.disabled = true;
  }
});

btnFileTest.addEventListener('click', () => fileVideo.click());

fileVideo.addEventListener('change', async () => {
  const file = fileVideo.files?.[0];
  if (!file) return;
  resultEl.textContent = '';
  resultEl.className = 'result';
  try {
    if (!faceLandmarker) await loadModel();
    await startWithVideoFile(file);
    beginSession();
  } catch (e) {
    console.error(e);
    modelStatus.textContent = 'ข้อผิดพลาด: ' + (e?.message || String(e));
    failSession('เปิดไฟล์วิดีโอไม่ได้ — ลองฟอร์แมตอื่น (เช่น MP4 H.264)');
    btnStart.disabled = false;
    btnFileTest.disabled = false;
    btnStop.disabled = true;
  }
  fileVideo.value = '';
});

btnStop.addEventListener('click', () => {
  stopCamera();
  btnStart.disabled = false;
  btnFileTest.disabled = false;
  btnStop.disabled = true;
  challengeText.textContent = import.meta.env.PROD
    ? 'กด «เริ่มด้วยกล้อง» — ต้องเป็นหน้าจริงหน้ากล้อง'
    : 'กด «เริ่มด้วยกล้อง» หรือ «ทดสอบจากไฟล์วิดีโอ»';
  challengeHint.textContent = '';
  progressEl.textContent = '';
  hud.textContent = '';
  resetSession();
});

loadModel().catch((e) => {
  console.error(e);
  modelStatus.textContent = 'โหลดโมเดลไม่สำเร็จ: ' + (e?.message || String(e));
});

if (import.meta.env.PROD) {
  document.querySelector('.controls-alt')?.classList.add('hidden');
}

window.addEventListener('resize', () => {
  if (running) resizeCanvas();
});
