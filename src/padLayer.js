/**
 * Passive Presentation Attack Detection (PAD) — heuristics on RGB face crop.
 * ไม่ใช่โมเดล CNN — ใช้สัญญาณทางสถิติ; หน้ากากซิลิโคน RGB อย่างเดียวยากมาก ต้องยอมรับ false positive/negative
 */

const PAD_SIZE = 128;

/** ต้องตรวจพบแบบเดียวกันติดกันกี่เฟรมก่อน fail (ลดสะดุ้งจากแสง) */
export const PAD_CONSECUTIVE_FRAMES = 7;

/** Laplacian variance ต่ำมาก = พื้นผิวแบน (กระดาษ/พิมพ์) */
const LAP_VAR_PRINT_MAX = 18;
const LOCAL_NOISE_PRINT_MAX = 2.2;

/** แพตเทิร์นจอ: แถบแนวตั้ง / ความสม่ำเสมอของแถว */
const ROW_UNIFORMITY_SCREEN = 0.92;
const SCREEN_BLUE_BIAS_FRAC = 0.38;

/** บล็อกแบบ codec วิดีโอ */
const BLOCK_GRID = 8;
const BLOCK_MEAN_VAR_MIN = 28;
const INTRA_BLOCK_VAR_MAX = 65;

/** หน้ากาก: ศูนย์หน้าเรียบผิดธรรมชาติ + ขอบไฮไฟรุนแรง */
const CENTER_PATCH_STD_MAX = 6.5;
const EDGE_TO_CENTER_GRAD_RATIO_MIN = 2.4;

/**
 * @param {ImageData} imageData
 */
export function analyzePad(imageData) {
  const w = imageData.width;
  const h = imageData.height;
  const d = imageData.data;
  const n = w * h;
  const gray = new Float32Array(n);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    gray[j] = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
  }

  const lap = laplacianAbs(gray, w, h);
  const lapVar = variance(lap);
  const lapMean = mean(lap);

  const localNoise = estimateLocalNoise(gray, w, h);

  const rowMeans = rowMeansArr(gray, w, h);
  const rowUniformity = rowMeans.length > 2 ? 1 - coefficientOfVariation(rowMeans) : 0;

  let blueBiasFrac = 0;
  for (let i = 0, j = 0; i < d.length; i += 4, j++) {
    const r = d[i];
    const g = d[i + 1];
    const b = d[i + 2];
    const s = r + g + b + 1e-6;
    if ((b - r) / s > 0.1 && b > r) blueBiasFrac++;
  }
  blueBiasFrac /= n;

  const { blockMeanVar, intraAvg } = blockinessStats(gray, w, h);

  const { centerStd, edgeGradRatio } = maskSkinSignals(gray, w, h);

  const printLike =
    lapVar < LAP_VAR_PRINT_MAX && localNoise < LOCAL_NOISE_PRINT_MAX && lapMean < 6;

  const screenLike =
    (rowUniformity > ROW_UNIFORMITY_SCREEN && blueBiasFrac > SCREEN_BLUE_BIAS_FRAC) ||
    (rowUniformity > 0.88 && blueBiasFrac > 0.45);

  const videoBlockLike =
    blockMeanVar > BLOCK_MEAN_VAR_MIN && intraAvg < INTRA_BLOCK_VAR_MAX && lapVar < 120;

  const maskSuspect =
    centerStd < CENTER_PATCH_STD_MAX &&
    edgeGradRatio > EDGE_TO_CENTER_GRAD_RATIO_MIN &&
    lapVar > 25 &&
    lapVar < 200;

  return {
    printLike,
    screenLike,
    videoBlockLike,
    maskSuspect,
    debug: {
      lapVar: round2(lapVar),
      localNoise: round2(localNoise),
      rowUniformity: round2(rowUniformity),
      blueBiasFrac: round2(blueBiasFrac),
      blockMeanVar: round2(blockMeanVar),
      intraAvg: round2(intraAvg),
      centerStd: round2(centerStd),
      edgeGradRatio: round2(edgeGradRatio),
    },
  };
}

export function getPadCanvasSize() {
  return PAD_SIZE;
}

function round2(x) {
  return Math.round(x * 100) / 100;
}

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function variance(arr) {
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const t = arr[i] - m;
    s += t * t;
  }
  return s / arr.length;
}

function coefficientOfVariation(arr) {
  const m = mean(arr);
  if (m < 1e-6) return 1;
  return Math.sqrt(variance(arr)) / m;
}

/** |∇²I| แบบประมาณ 3×3 */
function laplacianAbs(gray, w, h) {
  const out = new Float32Array(gray.length);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const v =
        -4 * gray[i] +
        gray[i - 1] +
        gray[i + 1] +
        gray[i - w] +
        gray[i + w];
      out[i] = Math.abs(v);
    }
  }
  return out;
}

function estimateLocalNoise(gray, w, h) {
  let s = 0;
  let c = 0;
  for (let y = 2; y < h - 2; y += 2) {
    for (let x = 2; x < w - 2; x += 2) {
      const i = y * w + x;
      const neigh =
        (gray[i - 2] +
          gray[i + 2] +
          gray[i - 2 * w] +
          gray[i + 2 * w]) /
        4;
      s += Math.abs(gray[i] - neigh);
      c++;
    }
  }
  return c ? s / c : 0;
}

function rowMeansArr(gray, w, h) {
  const rows = [];
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = 0; x < w; x++) s += gray[y * w + x];
    rows.push(s / w);
  }
  return rows;
}

function blockinessStats(gray, w, h) {
  const bw = Math.floor(w / BLOCK_GRID);
  const bh = Math.floor(h / BLOCK_GRID);
  const means = [];
  let intraSum = 0;
  let intraC = 0;
  for (let by = 0; by < BLOCK_GRID; by++) {
    for (let bx = 0; bx < BLOCK_GRID; bx++) {
      let sum = 0;
      let c = 0;
      const x0 = bx * bw;
      const y0 = by * bh;
      for (let y = y0; y < y0 + bh && y < h; y++) {
        for (let x = x0; x < x0 + bw && x < w; x++) {
          sum += gray[y * w + x];
          c++;
        }
      }
      const m = c ? sum / c : 0;
      means.push(m);
      let v = 0;
      for (let y = y0; y < y0 + bh && y < h; y++) {
        for (let x = x0; x < x0 + bw && x < w; x++) {
          const t = gray[y * w + x] - m;
          v += t * t;
        }
      }
      intraSum += c ? v / c : 0;
      intraC++;
    }
  }
  const blockMeanVar = variance(means);
  const intraAvg = intraC ? intraSum / intraC : 0;
  return { blockMeanVar, intraAvg };
}

/** สัญญาณเสี่ยงหน้ากาก: กลางหน้าเรียบ + ขอบไฮไฟแรง */
function maskSkinSignals(gray, w, h) {
  const cx = Math.floor(w * 0.5);
  const cy = Math.floor(h * 0.45);
  const r = Math.floor(Math.min(w, h) * 0.18);
  const patches = [];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const px = cx + dx * 8;
      const py = cy + dy * 8;
      let s = 0;
      let c = 0;
      for (let y = py - 4; y <= py + 4; y++) {
        for (let x = px - 4; x <= px + 4; x++) {
          if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) continue;
          s += gray[y * w + x];
          c++;
        }
      }
      if (c) patches.push(s / c);
    }
  }
  const centerStd = patches.length > 1 ? Math.sqrt(variance(patches)) : 20;

  let edgeG = 0;
  let cG = 0;
  for (let y = 2; y < h - 2; y++) {
    for (let x = 2; x < w - 2; x++) {
      const i = y * w + x;
      const g =
        Math.abs(gray[i + 1] - gray[i - 1]) + Math.abs(gray[i + w] - gray[i - w]);
      const dx = x / w - 0.5;
      const dy = y / h - 0.45;
      const rd = Math.sqrt(dx * dx + dy * dy);
      if (rd > 0.32 && rd < 0.48) edgeG += g;
      if (rd < 0.18) cG += g;
    }
  }
  const edgeGradRatio = cG > 8 ? edgeG / cG : 1;
  return { centerStd, edgeGradRatio };
}

/**
 * @param {{ onFail: (msg: string) => void }} opts
 */
export function createPadAccumulator(opts) {
  let cPrint = 0;
  let cScreen = 0;
  let cVideo = 0;
  let cMask = 0;

  return {
    reset() {
      cPrint = 0;
      cScreen = 0;
      cVideo = 0;
      cMask = 0;
    },
    /**
     * @param {ReturnType<typeof analyzePad>} r
     * @returns {boolean} true if session should stop (failed)
     */
    feed(r) {
      cPrint = r.printLike ? cPrint + 1 : 0;
      cScreen = r.screenLike ? cScreen + 1 : 0;
      cVideo = r.videoBlockLike ? cVideo + 1 : 0;
      cMask = r.maskSuspect ? cMask + 1 : 0;

      if (cPrint >= PAD_CONSECUTIVE_FRAMES) {
        opts.onFail(
          'ตรวจพบลักษณะภาพพิมพ์/กระดาษ (พื้นผิวแบนเกินไป) — ใช้ใบหน้าจริงหน้ากล้องเท่านั้น',
        );
        return true;
      }
      if (cScreen >= PAD_CONSECUTIVE_FRAMES) {
        opts.onFail(
          'ตรวจพบลักษณะจอแสดงผล/มือถือ (แพตเทิร์นแสงและสีไม่เป็นธรรมชาติ) — ห้ามถ่ายจากหน้าจอ',
        );
        return true;
      }
      if (cVideo >= PAD_CONSECUTIVE_FRAMES) {
        opts.onFail(
          'ตรวจพบลักษณะวิดีโอบีบอัด/บล็อก — ห้ามเล่นคลิปใส่กล้อง',
        );
        return true;
      }
      if (cMask >= PAD_CONSECUTIVE_FRAMES + 2) {
        opts.onFail(
          'ตรวจพบลักษณะผิดปกติที่ขอบใบหน้า/ผิว (อาจเป็นหน้ากากหรือวัตถุปลอม) — ใช้ใบหน้าเปล่าเท่านั้น',
        );
        return true;
      }
      return false;
    },
  };
}
