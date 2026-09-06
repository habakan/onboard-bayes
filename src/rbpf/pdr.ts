import type { Odometry } from "./filter";

/**
 * Pedestrian dead reckoning from `devicemotion`: detect steps, estimate how long each
 * one was, and turn the pair into the odometry the filter propagates on.
 *
 * Double-integrating acceleration is not an option here — the bias the accelerometer
 * model estimates is around 0.07 m/s², which integrates to about 3.5 m of position
 * error after 10 seconds. Counting steps replaces that drift with a per-step error
 * that does not compound the same way.
 */

export interface MotionSample {
  ax: number;
  ay: number;
  az: number;
  /** Angular rate about the device axes, rad/s. Absent means zero, which is what the
   *  synthetic gaits that only turn about z supply. */
  wx?: number;
  wy?: number;
  wz: number;
  /** Milliseconds. */
  t: number;
}

export interface Step {
  t: number;
  /** Peak-to-trough acceleration over the step, which sets its length. */
  amplitude: number;
  /** Heading change accumulated since the previous step, in radians. */
  dyaw: number;
}

export interface StepOptions {
  /** Ignore peaks closer together than this; a fast walk is about 2.5 steps/s. */
  minIntervalMs: number;
  /** Peak must exceed the gravity baseline by this, in m/s². Smoothing takes the top
   *  off a peak, so this sits well under the raw swing it is derived from. */
  threshold: number;
  /** Low-pass smoothing on the magnitude, 0..1; higher follows the signal faster. */
  alpha: number;
}

/// Swept against synthetic gaits from 1.5 to 2.5 steps/s: 0.3 counts every one of
/// them exactly, including a weak 0.6 m/s² signal, and finds nothing while stationary.
export const STEP_DEFAULTS: StepOptions = {
  minIntervalMs: 300,
  threshold: 0.3,
  alpha: 0.25,
};

/**
 * Peaks in the smoothed acceleration magnitude. Walking shows one clear peak per
 * step, so this counts zero-crossings of the derivative above a threshold rather than
 * trying to model the gait.
 */
export function detectSteps(
  samples: readonly MotionSample[],
  opts: StepOptions = STEP_DEFAULTS,
): Step[] {
  const steps: Step[] = [];
  if (samples.length < 3) return steps;

  let smooth = magnitude(samples[0]!);
  // Tracks gravity, not the gait: fast enough to follow the phone's attitude, slow
  // enough that a step's own peak does not pull the reference up with it.
  let baseline = smooth;
  let prev = smooth;
  let rising = false;
  let lastT = -Infinity;
  let peak = smooth;
  let trough = smooth;
  let yawAcc = 0;
  let lastSampleT = samples[0]!.t;
  // Yaw is rotation about vertical, and only a phone held flat has that on its own z.
  // Projecting onto measured gravity makes the carry angle stop mattering.
  let gx = samples[0]!.ax;
  let gy = samples[0]!.ay;
  let gz = samples[0]!.az;

  for (const s of samples) {
    const mag = magnitude(s);
    smooth += opts.alpha * (mag - smooth);
    baseline += 0.01 * (mag - baseline);

    gx += 0.02 * (s.ax - gx);
    gy += 0.02 * (s.ay - gy);
    gz += 0.02 * (s.az - gz);
    const gn = Math.hypot(gx, gy, gz) || 1;
    const yawRate = ((s.wx ?? 0) * gx + (s.wy ?? 0) * gy + s.wz * gz) / gn;
    yawAcc += yawRate * ((s.t - lastSampleT) / 1000);
    lastSampleT = s.t;

    if (smooth > peak) peak = smooth;
    if (smooth < trough) trough = smooth;

    const goingUp = smooth > prev;
    // A step is where the smoothed magnitude turns over, far enough above gravity.
    if (rising && !goingUp && prev - baseline > opts.threshold && s.t - lastT >= opts.minIntervalMs) {
      steps.push({ t: s.t, amplitude: Math.max(peak - trough, 1e-6), dyaw: yawAcc });
      lastT = s.t;
      peak = smooth;
      trough = smooth;
      yawAcc = 0;
    }
    rising = goingUp;
    prev = smooth;
  }
  return steps;
}

function magnitude(s: MotionSample): number {
  return Math.hypot(s.ax, s.ay, s.az);
}

/**
 * Weinberg's stride estimate, `k · amplitude^{1/4}`. `k` is per-person and per-phone,
 * which is exactly the kind of constant the hierarchical Stan model is for — a
 * calibrated `k` is what connects the two halves of this app.
 */
export function strideLength(amplitude: number, k: number): number {
  return k * Math.pow(amplitude, 0.25);
}

/** Steps to odometry, walking forward along the body's x axis. */
export function toOdometry(steps: readonly Step[], k: number): Odometry[] {
  return steps.map((s) => ({
    ds: [strideLength(s.amplitude, k), 0, 0] as [number, number, number],
    dyaw: s.dyaw,
  }));
}
