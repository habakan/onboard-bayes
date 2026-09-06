import { Rbpf, DEFAULTS, type Pose } from "./rbpf/filter";
import { detectSteps, toOdometry, STEP_DEFAULTS, type MotionSample } from "./rbpf/pdr";
import { requestMotionPermission } from "./sensors";
import { captureField, isNative } from "./magnetometer";

export interface TrackState {
  pose: Pose;
  path: [number, number][];
  steps: number;
  ess: number;
  /** Field readings folded in so far. */
  updates: number;
}

export interface TrackOptions {
  /** Weinberg constant. Calibrate it — see the sensitivity table in the README. */
  strideK: number;
  /** Half-extent of the mapped box, metres. */
  extent: number;
  particles: number;
  m: number;
}

export const TRACK_DEFAULTS: TrackOptions = {
  strideK: 0.45,
  extent: 15,
  particles: 100,
  m: 64,
};

/**
 * Drives the filter from live sensors: buffer motion, turn each newly detected step
 * into odometry, and fold in a field reading at every step.
 *
 * Steps rather than a fixed clock, because odometry only exists per step — predicting
 * between them would add noise without adding information.
 */
export class Tracker {
  private filter: Rbpf;
  private buffer: MotionSample[] = [];
  /** Timestamp of the last step already fed to the filter. Detection reruns over the
   *  whole buffer, so identity has to come from the step itself — a count breaks the
   *  moment the buffer is trimmed. */
  private lastStepT = -Infinity;
  private draining = false;
  private running = false;
  private state: TrackState;

  constructor(
    private readonly opts: TrackOptions,
    private readonly onUpdate: (s: TrackState) => void,
    /** Fires once per detected step, so the walker can see detection happen rather
     *  than trying to reconcile a total afterwards. */
    private readonly onStep?: (n: number) => void,
  ) {
    const e = opts.extent;
    this.filter = new Rbpf(
      {
        ...DEFAULTS,
        domain: { L: [e, e, 3] },
        particles: opts.particles,
        m: opts.m,
        noise: 2.0,
        sigma: 30,
        ell: 1.5,
        // Heading noise per stride, not a safety margin: at 0.05 the particles turn
        // more than the gyro does and nothing pulls them back. See degradation-check.
        odomNoise: { pos: 0.05, yaw: 0.01 },
        seed: 1,
      },
      { x: [0, 0, 0], yaw: 0 },
    );
    this.state = {
      pose: { x: [0, 0, 0], yaw: 0 },
      path: [[0, 0]],
      steps: 0,
      ess: opts.particles,
      updates: 0,
    };
  }

  async start(): Promise<void> {
    if (this.running) return;
    if (!(await requestMotionPermission())) throw new Error("motion permission denied");
    this.running = true;
    window.addEventListener("devicemotion", this.onMotion);
  }

  stop(): void {
    this.running = false;
    window.removeEventListener("devicemotion", this.onMotion);
  }

  get snapshot(): TrackState {
    return this.state;
  }

  private onMotion = (e: DeviceMotionEvent) => {
    const a = e.accelerationIncludingGravity;
    if (!a || a.x === null || a.y === null || a.z === null) return;
    // WebKit puts rotationRate's alpha/beta/gamma on x/y/z, not the spec's z/x/y, in
    // deg/s, against an acceleration whose sign is inverted. Identified by replaying a
    // recorded walk: this is the one of twelve axis/sign choices the compass agrees with.
    const r = e.rotationRate;
    const k = -Math.PI / 180;
    this.buffer.push({
      ax: a.x,
      ay: a.y,
      az: a.z,
      wx: (r?.alpha ?? 0) * k,
      wy: (r?.beta ?? 0) * k,
      wz: (r?.gamma ?? 0) * k,
      t: performance.now(),
    });
    if (this.buffer.length % 10 !== 0) return;
    void this.drain();
  };

  private async drain(): Promise<void> {
    // `drain` awaits a field reading, and motion events keep arriving meanwhile —
    // without this a second pass would replay steps the first is still working through.
    if (this.draining) return;
    this.draining = true;
    try {
      await this.drainOnce();
    } finally {
      this.draining = false;
    }
  }

  private async drainOnce(): Promise<void> {
    const steps = detectSteps(this.buffer, STEP_DEFAULTS);
    const fresh = steps.filter((s) => s.t > this.lastStepT);
    if (fresh.length === 0) return;
    this.lastStepT = fresh[fresh.length - 1]!.t;

    for (const o of toOdometry(fresh, this.opts.strideK)) {
      this.filter.predict(o);
      this.state.steps++;
      this.onStep?.(this.state.steps);
    }

    // One field reading per batch of steps. Raw axes come from the native plugin;
    // without it there is nothing to condition on and this is dead reckoning.
    if (isNative()) {
      try {
        const rs = await captureField(1, 2000);
        if (rs[0]) {
          this.filter.update([rs[0].x, rs[0].y, rs[0].z]);
          this.state.updates++;
        }
      } catch {
        // A dropped reading is not fatal; the next step will try again.
      }
    }

    const p = this.filter.estimate();
    this.state.pose = p;
    this.state.path.push([p.x[0]!, p.x[1]!]);
    if (this.state.path.length > 2000) this.state.path.shift();
    this.state.ess = this.filter.ess();
    this.onUpdate(this.state);

    // Keep only what the next detection window needs. `lastStepT` is absolute, so
    // trimming cannot make an old step look new.
    if (this.buffer.length > 600) this.buffer = this.buffer.slice(this.buffer.length - 300);
  }
}

/** Fits the path into the canvas with a margin, so a walk of any size stays visible. */
export function drawPath(
  canvas: HTMLCanvasElement,
  path: readonly [number, number][],
  pose: Pose,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  let minX = -1;
  let maxX = 1;
  let minY = -1;
  let maxY = 1;
  for (const [x, y] of path) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const pad = 20;
  const span = Math.max(maxX - minX, maxY - minY, 2);
  const scale = (Math.min(w, h) - 2 * pad) / span;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const px = (x: number) => w / 2 + (x - cx) * scale;
  const py = (y: number) => h / 2 - (y - cy) * scale;

  ctx.strokeStyle = "#e5e5e5";
  ctx.beginPath();
  ctx.moveTo(px(cx - span / 2), py(cy));
  ctx.lineTo(px(cx + span / 2), py(cy));
  ctx.moveTo(px(cx), py(cy - span / 2));
  ctx.lineTo(px(cx), py(cy + span / 2));
  ctx.stroke();

  ctx.strokeStyle = "#c2410c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  path.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(px(x), py(y)) : ctx.lineTo(px(x), py(y))));
  ctx.stroke();

  // Current pose, with a tick for heading.
  const hx = px(pose.x[0]!);
  const hy = py(pose.x[1]!);
  ctx.fillStyle = "#1a1a1a";
  ctx.beginPath();
  ctx.arc(hx, hy, 5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.strokeStyle = "#1a1a1a";
  ctx.beginPath();
  ctx.moveTo(hx, hy);
  ctx.lineTo(hx + 14 * Math.cos(pose.yaw), hy - 14 * Math.sin(pose.yaw));
  ctx.stroke();

  ctx.fillStyle = "#888";
  ctx.font = "11px -apple-system, sans-serif";
  ctx.fillText(`${span.toFixed(1)} m across`, 8, h - 8);
}
