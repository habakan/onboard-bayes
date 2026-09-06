import { fieldDesign, priorVariance, type Basis } from "./basis";

/**
 * The Gaussian posterior over the reduced-rank weights that one particle carries.
 * Conditional on a pose trajectory the field model is linear-Gaussian, so this is a
 * Kalman update rather than more particles — the Rao-Blackwellisation.
 *
 * `cov` is dense and m×m, which is what sets both the memory and the per-step cost.
 */
export class MagMap {
  readonly m: number;
  readonly mean: Float64Array;
  readonly cov: Float64Array;

  private readonly H: Float64Array;
  private readonly Ph: Float64Array;

  constructor(
    readonly basis: Basis,
    sigma: number,
    ell: number,
  ) {
    this.m = basis.m;
    this.mean = new Float64Array(this.m);
    this.cov = new Float64Array(this.m * this.m);
    const s = priorVariance(basis, sigma, ell);
    for (let i = 0; i < this.m; i++) this.cov[i * this.m + i] = s[i]!;
    this.H = new Float64Array(3 * this.m);
    this.Ph = new Float64Array(this.m);
  }

  clone(): MagMap {
    const c = Object.create(MagMap.prototype) as MagMap;
    Object.assign(c, {
      basis: this.basis,
      m: this.m,
      mean: this.mean.slice(),
      cov: this.cov.slice(),
      H: new Float64Array(3 * this.m),
      Ph: new Float64Array(this.m),
    });
    return c;
  }

  /**
   * Conditions on a 3-axis field reading at `x`, one axis at a time so each is a
   * rank-1 update. Returns the log density of the reading under the predictive
   * distribution, which is the particle's weight increment.
   */
  update(x: readonly number[], y: readonly number[], noiseVar: number): number {
    fieldDesign(this.basis, x, this.H);
    let logp = 0;
    for (let d = 0; d < 3; d++) {
      logp += this.updateRow(this.H.subarray(d * this.m, (d + 1) * this.m), y[d]!, noiseVar);
    }
    return logp;
  }

  /** `P ← P − (Ph)(Ph)ᵀ / s`, `mean ← mean + (Ph) r / s`. The hot loop. */
  private updateRow(h: Float64Array, y: number, noiseVar: number): number {
    const { m, cov, mean, Ph } = this;
    for (let i = 0; i < m; i++) {
      let acc = 0;
      const row = i * m;
      for (let j = 0; j < m; j++) acc += cov[row + j]! * h[j]!;
      Ph[i] = acc;
    }
    let hPh = 0;
    let pred = 0;
    for (let i = 0; i < m; i++) {
      hPh += h[i]! * Ph[i]!;
      pred += h[i]! * mean[i]!;
    }
    const s = hPh + noiseVar;
    const inv = 1 / s;
    const r = y - pred;

    for (let i = 0; i < m; i++) {
      const gi = Ph[i]! * inv;
      mean[i] = mean[i]! + gi * r;
      const row = i * m;
      for (let j = 0; j < m; j++) cov[row + j] = cov[row + j]! - gi * Ph[j]!;
    }
    return -0.5 * (Math.log(2 * Math.PI * s) + (r * r) * inv);
  }

  /** Posterior mean field at `x`. */
  predict(x: readonly number[]): [number, number, number] {
    fieldDesign(this.basis, x, this.H);
    const out: [number, number, number] = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      let acc = 0;
      const off = d * this.m;
      for (let j = 0; j < this.m; j++) acc += this.H[off + j]! * this.mean[j]!;
      out[d] = acc;
    }
    return out;
  }
}
