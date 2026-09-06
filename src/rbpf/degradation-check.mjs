/**
 * Why the filter is worse than dead reckoning when the field carries no information.
 *
 * With `richness = 0` the field is uniform, so no weighting can improve a pose and the
 * filter should degrade to dead reckoning. It scores three times worse instead. Two
 * things could do that — resampling on weights that are noise, or the odometry noise
 * injected into every particle — so each is switched off in turn.
 */
import { Rbpf, DEFAULTS } from "./filter.ts";
import { TRAJECTORIES, makeField, odometryFrom, rng, gauss } from "./scenarios.mjs";

const DOMAIN = { L: [10, 10, 2] };
const dist = (a, b) => Math.hypot(a.x[0] - b.x[0], a.x[1] - b.x[1]);
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

function deadReckon(start, odo) {
  const out = [{ x: [...start.x], yaw: start.yaw }];
  let p = { x: [...start.x], yaw: start.yaw };
  for (const o of odo) {
    const yaw = p.yaw + o.dyaw;
    const c = Math.cos(yaw), s = Math.sin(yaw);
    p = { x: [p.x[0] + c * o.ds[0] - s * o.ds[1], p.x[1] + s * o.ds[0] + c * o.ds[1], 0], yaw };
    out.push(p);
  }
  return out;
}
const ate = (e, t) => Math.sqrt(e.reduce((s, _, i) => s + dist(e[i], t[i]) ** 2, 0) / e.length);

function run({ traj = "loop", richness = 0, seed, resampleAt = 0.5, odomPos = 0.03, odomYaw = 0.01,
              steps = 300, truePos = 0.01, trueYaw = 0.004 }) {
  const truth = TRAJECTORIES[traj](steps);
  const field = makeField(DOMAIN, 128, richness, 42);
  const odo = odometryFrom(truth, { scaleBias: 0.03, posNoise: truePos, yawNoise: trueYaw, seed });
  const f = new Rbpf(
    { ...DEFAULTS, domain: DOMAIN, particles: 150, m: 96, noise: 0.5, sigma: 30, ell: 1.2,
      odomNoise: { pos: odomPos, yaw: odomYaw }, resampleAt, seed },
    truth[0],
  );
  const u = rng(seed + 991);
  const est = [{ x: [...truth[0].x], yaw: truth[0].yaw }];
  const N = f.poses.length;
  const essTrace = [];
  let resamples = 0;
  for (let i = 0; i < odo.length; i++) {
    f.predict(odo[i]);
    f.update(field(truth[i + 1].x).map((v) => v + 0.5 * gauss(u)));
    const e = f.ess();
    // A reset to exactly N is the resample's signature.
    if (e > N - 0.001 && essTrace.length && essTrace[essTrace.length - 1] < N - 0.001) resamples++;
    essTrace.push(e);
    est.push(f.estimate());
  }
  const dr = deadReckon(truth[0], odo);
  return { ate: ate(est, truth), ateDR: ate(dr, truth), resamples,
           essMin: Math.min(...essTrace), essMean: mean(essTrace),
           driftFromDR: mean(est.map((e, i) => dist(e, dr[i]))) };
}

const SEEDS = [5, 11, 23, 41];
const f2 = (v) => v.toFixed(2).padStart(6);

const CASES = [
  ["as shipped", {}],
  ["never resample", { resampleAt: 0 }],
  ["no odometry noise", { odomPos: 0, odomYaw: 0 }],
  ["no yaw noise only", { odomYaw: 0 }],
  ["no position noise only", { odomPos: 0 }],
  ["odometry noise matched", { odomPos: 0.01, odomYaw: 0.004 }],
];

/** One simulation per seed, then every metric off the same runs. */
function summarise(cfg) {
  const rs = SEEDS.map((seed) => run({ ...cfg, seed }));
  const of = (k) => mean(rs.map((r) => r[k]));
  return { ate: of("ate"), ateDR: of("ateDR"), drift: of("driftFromDR"),
           resamples: of("resamples"), ess: of("essMean") };
}

console.log("Uniform field, so nothing the weights say can be information. 4 seeds, loop.\n");
console.log("configuration                 ATE     DR    drift from DR  resamples  mean ESS");
for (const [name, cfg] of CASES) {
  const r = summarise({ ...cfg, richness: 0 });
  console.log(
    `${name.padEnd(28)} ${f2(r.ate)} ${f2(r.ateDR)}   ${f2(r.drift)}` +
      `     ${r.resamples.toFixed(0).padStart(4)}     ${r.ess.toFixed(0).padStart(4)}`,
  );
}

console.log("\nField back on (richness 1), to check a fix would not just be turning the");
console.log("filter off. figure8, the trajectory that revisits.\n");
console.log("configuration                 ATE     DR");
for (const [name, cfg] of CASES) {
  const r = summarise({ ...cfg, richness: 1, traj: "figure8" });
  console.log(`${name.padEnd(28)} ${f2(r.ate)} ${f2(r.ateDR)}`);
}

// The app walks in 0.7 m strides, not the 8 cm the rows above step in, so its process
// noise cannot be read off them directly. Same loop, 40 strides, app-scale odometry.
const STRIDE = { steps: 40, truePos: 0.05, trueYaw: 0.01, traj: "loop" };
console.log("\nApp scale: 40 strides of 0.63 m, odometry noise 0.05 m and 0.01 rad.\n");
console.log("assumed pos, yaw            uniform field   with field    DR");
for (const [odomPos, odomYaw] of [[0.05, 0.05], [0.05, 0.01], [0.05, 0.004], [0.02, 0.01]]) {
  const a = summarise({ ...STRIDE, odomPos, odomYaw, richness: 0 });
  const b = summarise({ ...STRIDE, odomPos, odomYaw, richness: 1 });
  const tag = `${odomPos}, ${odomYaw}${odomPos === 0.05 && odomYaw === 0.05 ? "  (as shipped)" : ""}`;
  console.log(`${tag.padEnd(28)} ${f2(a.ate)}        ${f2(b.ate)}  ${f2(b.ateDR)}`);
}
