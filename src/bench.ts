import { StanModel } from "stanwasm";
import type { SensorModel } from "./models";

export interface RunResult {
  key: string;
  n: number;
  compileMs: number;
  sampleMs: number;
  msPerDraw: number;
  means: Record<string, number>;
  /** Strongest posterior correlation between two parameters. Near ±1 means only
   *  their combination was identified, not either one on its own. */
  worstPair: { a: string; b: string; corr: number } | null;
}

export interface RunOptions {
  warmup: number;
  draws: number;
  seed: bigint;
}

export const DEFAULTS: RunOptions = { warmup: 500, draws: 500, seed: 42n };

/**
 * One fit, timed. `compileMs` is parse plus the initial trace, which a periodic
 * refit pays every time the data changes; `sampleMs` is NUTS alone.
 */
export function run(
  model: SensorModel,
  data: Record<string, number | number[]>,
  opts: RunOptions = DEFAULTS,
): RunResult {
  const t0 = performance.now();
  const compiled = new StanModel(model.stan, JSON.stringify(data));
  const compileMs = performance.now() - t0;

  const t1 = performance.now();
  const draws = compiled.sample(
    new Float64Array(model.init),
    opts.warmup,
    opts.draws,
    opts.seed,
  );
  const sampleMs = performance.now() - t1;

  const nP = compiled.n_params;
  const names = compiled.paramNames();
  const means: Record<string, number> = {};

  // sample() returns unconstrained draws, so read each one back through
  // constrainDraw to report sigma on its natural scale rather than log.
  const kept: number[][] = [];
  for (let i = 0; i < nP; i++) means[names[i] ?? `p${i}`] = 0;
  for (let d = opts.warmup; d < opts.warmup + opts.draws; d++) {
    const row = Array.from(compiled.constrainDraw(draws.slice(d * nP, (d + 1) * nP)));
    kept.push(row);
    for (let i = 0; i < row.length; i++) {
      const name = names[i] ?? `p${i}`;
      means[name] = (means[name] ?? 0) + row[i]! / opts.draws;
    }
  }

  compiled.free();
  const n = typeof data.N === "number" ? data.N : 0;
  return {
    key: model.key,
    n,
    compileMs: Math.round(compileMs),
    sampleMs: Math.round(sampleMs),
    msPerDraw: +(sampleMs / (opts.warmup + opts.draws)).toFixed(3),
    means,
    worstPair: strongestCorrelation(kept, names),
  };
}

/** The most correlated pair of parameters in the posterior. A hard-iron fit taken
 *  from one orientation puts `bz` and `radius` at −1.000: their sum is pinned and
 *  neither of them is, which reads as a confident wrong answer. */
function strongestCorrelation(
  draws: number[][],
  names: string[],
): { a: string; b: string; corr: number } | null {
  const k = draws[0]?.length ?? 0;
  if (draws.length < 20 || k < 2) return null;

  const mean = (i: number) => draws.reduce((acc, r) => acc + r[i]!, 0) / draws.length;
  const mu = Array.from({ length: k }, (_, i) => mean(i));
  const sd = Array.from({ length: k }, (_, i) =>
    Math.sqrt(draws.reduce((acc, r) => acc + (r[i]! - mu[i]!) ** 2, 0) / draws.length),
  );

  let best: { a: string; b: string; corr: number } | null = null;
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      if (sd[i]! < 1e-12 || sd[j]! < 1e-12) continue;
      const cov =
        draws.reduce((acc, r) => acc + (r[i]! - mu[i]!) * (r[j]! - mu[j]!), 0) / draws.length;
      const c = cov / (sd[i]! * sd[j]!);
      if (!best || Math.abs(c) > Math.abs(best.corr)) {
        best = { a: names[i] ?? `p${i}`, b: names[j] ?? `p${j}`, corr: +c.toFixed(3) };
      }
    }
  }
  return best;
}

/** Same model over growing N, which is what decides whether a refit fits inside a
 *  sensor duty cycle — cost is per leapfrog step, and N sets the step's size. */
export function sweep(
  model: SensorModel,
  build: (n: number) => Record<string, number | number[]>,
  sizes: number[],
  opts: RunOptions = DEFAULTS,
): RunResult[] {
  return sizes.map((n) => run(model, build(n), opts));
}
