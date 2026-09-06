export interface MotionSample {
  ax: number;
  ay: number;
  az: number;
  wx: number;
  wy: number;
  wz: number;
}

type PermissionFn = () => Promise<"granted" | "denied">;

function requestFn(ctor: unknown): PermissionFn | null {
  const f = (ctor as { requestPermission?: unknown })?.requestPermission;
  return typeof f === "function" ? (f as PermissionFn).bind(ctor) : null;
}

/** iOS gates motion behind a prompt that must come from a user gesture; every other
 *  engine has no such call and starts delivering events immediately. */
export async function requestMotionPermission(): Promise<boolean> {
  const ask = requestFn(window.DeviceMotionEvent);
  if (!ask) return true;
  try {
    return (await ask()) === "granted";
  } catch {
    return false;
  }
}

export async function requestOrientationPermission(): Promise<boolean> {
  const ask = requestFn(window.DeviceOrientationEvent);
  if (!ask) return true;
  try {
    return (await ask()) === "granted";
  } catch {
    return false;
  }
}

/**
 * Records `devicemotion` until `n` samples land or `timeoutMs` passes. Rejects rather
 * than returning a short buffer, since a partial capture would silently weaken the fit.
 */
export function captureMotion(n: number, timeoutMs = 15000): Promise<MotionSample[]> {
  return new Promise((resolve, reject) => {
    const out: MotionSample[] = [];

    const onMotion = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity;
      const r = e.rotationRate;
      if (!a || a.x === null || a.y === null || a.z === null) return;
      out.push({
        ax: a.x,
        ay: a.y,
        az: a.z,
        wx: r?.alpha ?? 0,
        wy: r?.beta ?? 0,
        wz: r?.gamma ?? 0,
      });
      if (out.length >= n) done();
    };

    const timer = window.setTimeout(() => {
      cleanup();
      reject(
        new Error(
          `only ${out.length}/${n} samples in ${timeoutMs} ms — devicemotion may be unavailable here`,
        ),
      );
    }, timeoutMs);

    function cleanup() {
      window.removeEventListener("devicemotion", onMotion);
      window.clearTimeout(timer);
    }
    function done() {
      cleanup();
      resolve(out);
    }

    window.addEventListener("devicemotion", onMotion);
  });
}

/** Which of the six faces is pointing down, from the dominant gravity axis. Two
 *  readings in the same bucket carry the same information about the offsets. */
function face(s: MotionSample): number {
  const v = [s.ax, s.ay, s.az];
  let k = 0;
  for (let i = 1; i < 3; i++) if (Math.abs(v[i]!) > Math.abs(v[k]!)) k = i;
  return v[k]! >= 0 ? k : k + 3;
}

export interface PoseProgress {
  faces: number;
  needed: number;
  kept: number;
}

/**
 * Captures for the accelerometer fit, which needs several orientations *and* each of
 * them stationary. A plain count fills from one face in under two seconds, and a
 * moving phone reports linear acceleration on top of gravity — so this takes a quota
 * per face, and only while `rotationRate` says the phone is at rest.
 */
export function capturePoses(
  perFace: number,
  minFaces: number,
  timeoutMs: number,
  onProgress?: (p: PoseProgress) => void,
): Promise<MotionSample[]> {
  return new Promise((resolve, reject) => {
    const STILL_DEG_PER_S = 3;
    const buckets = new Map<number, MotionSample[]>();

    const filled = () => [...buckets.values()].filter((b) => b.length >= perFace).length;
    const kept = () => [...buckets.values()].reduce((a, b) => a + Math.min(b.length, perFace), 0);

    const onMotion = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity;
      const r = e.rotationRate;
      if (!a || a.x === null || a.y === null || a.z === null) return;
      if (Math.hypot(r?.alpha ?? 0, r?.beta ?? 0, r?.gamma ?? 0) > STILL_DEG_PER_S) return;

      const s: MotionSample = {
        ax: a.x,
        ay: a.y,
        az: a.z,
        wx: r?.alpha ?? 0,
        wy: r?.beta ?? 0,
        wz: r?.gamma ?? 0,
      };
      const f = face(s);
      const b = buckets.get(f) ?? [];
      if (b.length >= perFace) return;
      b.push(s);
      buckets.set(f, b);
      onProgress?.({ faces: filled(), needed: minFaces, kept: kept() });
      if (filled() >= minFaces) done();
    };

    const timer = window.setTimeout(() => {
      cleanup();
      reject(
        new Error(
          `only ${filled()}/${minFaces} faces held still in ${timeoutMs / 1000}s — rest the phone on a different side and wait`,
        ),
      );
    }, timeoutMs);

    function cleanup() {
      window.removeEventListener("devicemotion", onMotion);
      window.clearTimeout(timer);
    }
    function done() {
      cleanup();
      resolve([...buckets.values()].flatMap((b) => b.slice(0, perFace)));
    }

    window.addEventListener("devicemotion", onMotion);
  });
}

/** iOS exposes a fused compass heading and no raw magnetometer; the Generic Sensor
 *  API that would give the raw axes is not implemented there at all. */
export function captureHeading(n: number, timeoutMs = 15000): Promise<number[]> {
  return new Promise((resolve, reject) => {
    const out: number[] = [];

    const onOrient = (e: DeviceOrientationEvent) => {
      const webkit = (e as DeviceOrientationEvent & { webkitCompassHeading?: number })
        .webkitCompassHeading;
      const deg = webkit ?? (e.alpha === null ? null : 360 - e.alpha);
      if (deg === null || Number.isNaN(deg)) return;
      out.push((deg * Math.PI) / 180);
      if (out.length >= n) done();
    };

    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error(`only ${out.length}/${n} heading samples in ${timeoutMs} ms`));
    }, timeoutMs);

    function cleanup() {
      window.removeEventListener("deviceorientation", onOrient);
      window.clearTimeout(timer);
    }
    function done() {
      cleanup();
      resolve(out);
    }

    window.addEventListener("deviceorientation", onOrient);
  });
}

export function deviceLabel(): string {
  const dpr = window.devicePixelRatio ?? 1;
  return `${navigator.userAgent} · ${screen.width}x${screen.height}@${dpr} · ${navigator.hardwareConcurrency ?? "?"} cores`;
}
