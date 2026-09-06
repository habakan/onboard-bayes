/**
 * Trajectories and fields for the localization benchmark. Each scenario stresses one
 * thing: a loop tests whether the map closes it, a corridor tests drift over distance,
 * a featureless field tests that the filter degrades to dead reckoning rather than
 * inventing corrections.
 */
import { makeBasis, fieldDesign } from "./basis.ts";

/** Deterministic, so a regression is a real change and not a reseed. */
export function rng(seed) {
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

export function gauss(u) {
  return Math.sqrt(-2 * Math.log(Math.max(u(), 1e-12))) * Math.cos(2 * Math.PI * u());
}

export const TRAJECTORIES = {
  /** Returns to the start, so the closure error is meaningful. */
  loop: (n) =>
    Array.from({ length: n }, (_, i) => {
      const t = (2 * Math.PI * i) / n;
      return { x: [4 * Math.sin(t), 4 * (1 - Math.cos(t)) - 4, 0], yaw: t };
    }),

  /** Two loops, so the second lap can be corrected by the map built on the first. */
  figure8: (n) =>
    Array.from({ length: n }, (_, i) => {
      const t = (4 * Math.PI * i) / n;
      return { x: [4 * Math.sin(t), 3 * Math.sin(2 * t), 0], yaw: t };
    }),

  /** Never revisits anywhere, which is the hardest case for a map. */
  corridor: (n) =>
    Array.from({ length: n }, (_, i) => {
      const s = (14 * i) / n - 7;
      return { x: [s, 0.4 * Math.sin(s * 0.7), 0], yaw: 0 };
    }),

  /** Back and forth over the same ground: revisits without closing a loop. */
  scan: (n) =>
    Array.from({ length: n }, (_, i) => {
      const u = (6 * i) / n;
      const lane = Math.floor(u);
      const along = (u - lane) * 8 - 4;
      const dir = lane % 2 === 0 ? 1 : -1;
      return { x: [dir * along, lane * 1.2 - 3, 0], yaw: dir > 0 ? 0 : Math.PI };
    }),
};

/**
 * A curl-free field built from the same basis the filter uses, so the benchmark tests
 * inference rather than model mismatch. `richness` scales the higher-order terms:
 * near zero the field is almost uniform and carries no positional information.
 */
export function makeField(domain, m, richness, seed) {
  const basis = makeBasis(domain, m);
  const u = rng(seed);
  const w = Float64Array.from({ length: m }, (_, j) => gauss(u) / (1 + j * 0.2));
  const H = new Float64Array(3 * m);
  const raw = (p) => {
    fieldDesign(basis, p, H);
    return [0, 1, 2].map((d) => {
      let a = 0;
      for (let j = 0; j < m; j++) a += H[d * m + j] * w[j];
      return a;
    });
  };

  // Scaled to what an indoor anomaly actually looks like: a 45 µT ambient field with
  // roughly `richness × 15 µT` of spatial variation on top. The first attempt left the
  // whole field at 1.7 µT, below the noise it was being measured with, and every
  // benchmark row was then reading noise.
  const probe = [];
  for (let i = 0; i < 200; i++) {
    const t = (2 * Math.PI * i) / 200;
    probe.push(raw([4 * Math.sin(t), 4 * Math.cos(t), 0]));
  }
  const mean = probe.reduce((a, v) => a + Math.hypot(...v), 0) / probe.length;
  const gain = mean > 1e-9 ? (richness * 15) / mean : 0;
  const AMBIENT = [22, -8, 38];
  return (p) => {
    const v = raw(p);
    return [0, 1, 2].map((d) => AMBIENT[d] + gain * v[d]);
  };
}

/** Odometry between consecutive poses, with a scale bias and per-step noise. */
export function odometryFrom(truth, { scaleBias = 0, posNoise = 0, yawNoise = 0, seed = 1 } = {}) {
  const u = rng(seed);
  const out = [];
  for (let i = 1; i < truth.length; i++) {
    const dx = truth[i].x[0] - truth[i - 1].x[0];
    const dy = truth[i].x[1] - truth[i - 1].x[1];
    const yaw = truth[i - 1].yaw;
    const c = Math.cos(-yaw);
    const s = Math.sin(-yaw);
    const k = 1 + scaleBias;
    out.push({
      ds: [k * (c * dx - s * dy) + posNoise * gauss(u), k * (s * dx + c * dy) + posNoise * gauss(u), 0],
      dyaw: truth[i].yaw - truth[i - 1].yaw + yawNoise * gauss(u),
    });
  }
  return out;
}
