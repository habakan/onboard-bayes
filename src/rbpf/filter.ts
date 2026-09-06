import { makeBasis, type Domain } from "./basis";
import { MagMap } from "./map";

export interface Pose {
  /** Position in the map's box. */
  x: [number, number, number];
  /** Heading about z, in radians. Dead reckoning drifts here first. */
  yaw: number;
}

export interface Odometry {
  /** Step in the body frame, from the pedometer or double-integrated motion. */
  ds: [number, number, number];
  /** Turn since the last step, from the gyroscope. */
  dyaw: number;
}

export interface FilterOptions {
  particles: number;
  m: number;
  domain: Domain;
  sigma: number;
  ell: number;
  /** Field measurement noise, in the same units as the readings. */
  noise: number;
  /** Per-step odometry noise: metres on position, radians on heading. */
  odomNoise: { pos: number; yaw: number };
  /** Resample once the effective sample size drops below this fraction. */
  resampleAt: number;
  seed: number;
}

export const DEFAULTS: Omit<FilterOptions, "domain"> = {
  particles: 200,
  m: 128,
  sigma: 1,
  ell: 0.7,
  noise: 0.05,
  odomNoise: { pos: 0.02, yaw: 0.01 },
  resampleAt: 0.5,
  seed: 1,
};

/** Deterministic PRNG, so a run can be replayed exactly when a fit misbehaves. */
function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}

function gauss(u: () => number): number {
  // Box-Muller. The log's argument is kept off zero.
  const a = Math.max(u(), 1e-12);
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * u());
}

/**
 * Rao-Blackwellised particle filter for magnetic-field SLAM. Particles carry only the
 * pose; each one's map is a Gaussian updated in closed form, which is what keeps the
 * particle count manageable — sampling an m-dimensional field directly would not.
 */
export class Rbpf {
  readonly opts: FilterOptions;
  poses: Pose[] = [];
  maps: MagMap[] = [];
  logW: Float64Array;
  private readonly u: () => number;

  constructor(opts: FilterOptions, start: Pose) {
    this.opts = opts;
    this.u = rng(opts.seed);
    const basis = makeBasis(opts.domain, opts.m);
    for (let i = 0; i < opts.particles; i++) {
      this.poses.push({ x: [...start.x], yaw: start.yaw });
      this.maps.push(new MagMap(basis, opts.sigma, opts.ell));
    }
    // Uniform and normalised from the start. Zeros would leave every weight at 1, so
    // `estimate` would sum the particles instead of averaging them and `ess` would
    // read 0 — which is what happens when there is no field to update against.
    this.logW = new Float64Array(opts.particles).fill(-Math.log(opts.particles));
  }

  /** Propagates every pose through the odometry, with noise. */
  predict(o: Odometry): void {
    const { pos, yaw } = this.opts.odomNoise;
    for (const p of this.poses) {
      p.yaw += o.dyaw + yaw * gauss(this.u);
      const c = Math.cos(p.yaw);
      const s = Math.sin(p.yaw);
      // Body-frame step rotated into the map frame; z is not turned.
      p.x[0] += c * o.ds[0] - s * o.ds[1] + pos * gauss(this.u);
      p.x[1] += s * o.ds[0] + c * o.ds[1] + pos * gauss(this.u);
      p.x[2] += o.ds[2] + pos * gauss(this.u);
    }
  }

  /** Conditions each map on the reading and reweights by how well it predicted it. */
  update(y: readonly number[]): void {
    const nv = this.opts.noise * this.opts.noise;
    for (let i = 0; i < this.poses.length; i++) {
      this.logW[i] = this.logW[i]! + this.maps[i]!.update(this.poses[i]!.x, y, nv);
    }
    this.normalise();
    if (this.ess() < this.opts.resampleAt * this.poses.length) this.resample();
  }

  private normalise(): void {
    let max = -Infinity;
    for (const w of this.logW) if (w > max) max = w;
    let sum = 0;
    for (let i = 0; i < this.logW.length; i++) sum += Math.exp(this.logW[i]! - max);
    const off = max + Math.log(sum);
    for (let i = 0; i < this.logW.length; i++) this.logW[i] = this.logW[i]! - off;
  }

  /** Effective sample size, `1 / Σ w²`. */
  ess(): number {
    let s = 0;
    for (const lw of this.logW) {
      const w = Math.exp(lw);
      s += w * w;
    }
    return 1 / s;
  }

  /** Systematic resampling: one uniform draw, lower variance than multinomial. */
  private resample(): void {
    const n = this.poses.length;
    const step = 1 / n;
    let target = this.u() * step;
    let acc = 0;
    let src = 0;
    const poses: Pose[] = [];
    const maps: MagMap[] = [];
    for (let i = 0; i < n; i++) {
      acc += Math.exp(this.logW[i]!);
      while (target < acc && poses.length < n) {
        poses.push({ x: [...this.poses[i]!.x], yaw: this.poses[i]!.yaw });
        maps.push(this.maps[i]!.clone());
        target += step;
      }
      src++;
    }
    // Numerical shortfall: top up from the last particle rather than return short.
    while (poses.length < n) {
      const last = Math.min(src, n) - 1;
      poses.push({ x: [...this.poses[last]!.x], yaw: this.poses[last]!.yaw });
      maps.push(this.maps[last]!.clone());
    }
    this.poses = poses;
    this.maps = maps;
    this.logW.fill(-Math.log(n));
  }

  /** Weighted mean pose. */
  estimate(): Pose {
    const out: Pose = { x: [0, 0, 0], yaw: 0 };
    let sy = 0;
    let cy = 0;
    for (let i = 0; i < this.poses.length; i++) {
      const w = Math.exp(this.logW[i]!);
      for (let d = 0; d < 3; d++) out.x[d] += w * this.poses[i]!.x[d]!;
      sy += w * Math.sin(this.poses[i]!.yaw);
      cy += w * Math.cos(this.poses[i]!.yaw);
    }
    out.yaw = Math.atan2(sy, cy);
    return out;
  }

  /** The map of the heaviest particle, for prediction or display. */
  bestMap(): MagMap {
    let best = 0;
    for (let i = 1; i < this.logW.length; i++) if (this.logW[i]! > this.logW[best]!) best = i;
    return this.maps[best]!;
  }
}
