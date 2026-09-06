/**
 * Localization benchmark: several trajectories and field conditions, scored with the
 * metrics the SLAM literature uses rather than one mean error.
 *
 * - ATE: absolute trajectory error, RMSE against the true positions. Global accuracy.
 * - RPE: relative pose error over a 10-step window. Local consistency, which ATE hides
 *   — a path that is right in shape but offset scores badly on ATE and well on RPE.
 * - Loop closure: distance between the estimate at the start and at the end, on
 *   trajectories that return. The one number a map is supposed to improve.
 *
 * Every row is averaged over several seeds. Run to run the ATE moves by more than the
 * settings being compared do, so a single-seed table ranks noise.
 *
 * Usage: npm run bench:loc            (the standard set)
 *        npm run bench:loc -- sweep   (particles and basis size)
 */
import { Rbpf, DEFAULTS } from "./filter.ts";
import { TRAJECTORIES, makeField, odometryFrom, rng, gauss } from "./scenarios.mjs";

const DOMAIN = { L: [10, 10, 2] };

function deadReckon(start, odo) {
  const out = [{ x: [...start.x], yaw: start.yaw }];
  let p = { x: [...start.x], yaw: start.yaw };
  for (const o of odo) {
    const yaw = p.yaw + o.dyaw;
    const c = Math.cos(yaw);
    const s = Math.sin(yaw);
    p = { x: [p.x[0] + c * o.ds[0] - s * o.ds[1], p.x[1] + s * o.ds[0] + c * o.ds[1], 0], yaw };
    out.push(p);
  }
  return out;
}

const dist = (a, b) => Math.hypot(a.x[0] - b.x[0], a.x[1] - b.x[1]);

function ate(est, truth) {
  let s = 0;
  for (let i = 0; i < est.length; i++) s += dist(est[i], truth[i]) ** 2;
  return Math.sqrt(s / est.length);
}

/** RMSE of how wrong each 10-step displacement is: local drift, offset removed. */
function rpe(est, truth, gap = 10) {
  let s = 0;
  let n = 0;
  for (let i = 0; i + gap < est.length; i++) {
    const de = [est[i + gap].x[0] - est[i].x[0], est[i + gap].x[1] - est[i].x[1]];
    const dt = [truth[i + gap].x[0] - truth[i].x[0], truth[i + gap].x[1] - truth[i].x[1]];
    s += (de[0] - dt[0]) ** 2 + (de[1] - dt[1]) ** 2;
    n++;
  }
  return n ? Math.sqrt(s / n) : NaN;
}

/** Only meaningful where the truth returns to where it started. */
function closure(est, truth) {
  const trueGap = dist(truth[0], truth[truth.length - 1]);
  if (trueGap > 0.5) return null;
  return dist(est[0], est[est.length - 1]);
}

function runOne({ traj, steps, richness, scaleBias, fieldNoise, particles, m, seed }) {
  const truth = TRAJECTORIES[traj](steps);
  const field = makeField(DOMAIN, 128, richness, 42);
  const odo = odometryFrom(truth, { scaleBias, posNoise: 0.01, yawNoise: 0.004, seed });

  const f = new Rbpf(
    {
      ...DEFAULTS,
      domain: DOMAIN,
      particles,
      m,
      noise: Math.max(fieldNoise, 0.05),
      sigma: 30,
      ell: 1.2,
      // Matched to the odometry generated above. Inflating it is not the safe choice it
      // looks like: the extra spread is a random walk nothing corrects where the map cannot.
      odomNoise: { pos: 0.01, yaw: 0.004 },
      seed,
    },
    truth[0],
  );

  const u = rng(seed + 991);
  const est = [{ x: [...truth[0].x], yaw: truth[0].yaw }];
  const t0 = performance.now();
  for (let i = 0; i < odo.length; i++) {
    f.predict(odo[i]);
    f.update(field(truth[i + 1].x).map((v) => v + fieldNoise * gauss(u)));
    est.push(f.estimate());
  }
  const ms = performance.now() - t0;
  const dr = deadReckon(truth[0], odo);

  return {
    ate: ate(est, truth),
    ateDR: ate(dr, truth),
    rpe: rpe(est, truth),
    rpeDR: rpe(dr, truth),
    closure: closure(est, truth),
    closureDR: closure(dr, truth),
    msPerStep: ms / odo.length,
    ess: f.ess(),
  };
}

const BASE = {
  steps: 300,
  richness: 1,
  scaleBias: 0.03,
  fieldNoise: 0.5,
  particles: 150,
  m: 96,
  seed: 5,
};

const SEEDS = [5, 11, 23, 41, 67, 89, 103, 137];

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => Math.sqrt(mean(a.map((v) => (v - mean(a)) ** 2)));

/** Runs every seed and reports the spread, plus how often the map actually helped. */
function over(cfg) {
  const rs = SEEDS.map((seed) => runOne({ ...BASE, ...cfg, seed }));
  const num = (k) => rs.map((r) => r[k]).filter((v) => v !== null && !Number.isNaN(v));
  return {
    ate: mean(num("ate")),
    ateSd: sd(num("ate")),
    ateDR: mean(num("ateDR")),
    rpe: mean(num("rpe")),
    rpeDR: mean(num("rpeDR")),
    closure: num("closure").length ? mean(num("closure")) : null,
    closureDR: num("closureDR").length ? mean(num("closureDR")) : null,
    wins: rs.filter((r) => r.ate < r.ateDR).length,
    msPerStep: mean(num("msPerStep")),
  };
}

const fmt = (v, d = 2) => (v === null || Number.isNaN(v) ? "  —  " : v.toFixed(d).padStart(5));

function standard() {
  console.log(`Each row is ${SEEDS.length} seeds. "wins" counts seeds where the filter beat dead reckoning.`);
  console.log("\nTrajectory   ATE ± sd      (DR)    RPE (DR)      closure (DR)   wins  ms/step");
  for (const traj of ["loop", "figure8", "corridor", "scan"]) {
    const r = over({ traj });
    console.log(
      `${traj.padEnd(11)} ${fmt(r.ate)} ±${fmt(r.ateSd)} (${fmt(r.ateDR)})  ${fmt(r.rpe)} (${fmt(r.rpeDR)})  ` +
        `${fmt(r.closure)} (${fmt(r.closureDR)})  ${String(r.wins).padStart(2)}/${SEEDS.length}  ${fmt(r.msPerStep, 1)}`,
    );
  }

  // The split is the point: a map can only correct a pose where it has been before, so
  // the two trajectories that revisit win and the two that never do lose.
  console.log("\nField richness — a uniform field carries no position information");
  console.log("richness   ATE (DR)      closure (DR)   wins");
  for (const richness of [0, 0.1, 0.3, 1, 3]) {
    const r = over({ traj: "figure8", richness });
    console.log(
      `${String(richness).padEnd(9)} ${fmt(r.ate)} (${fmt(r.ateDR)})  ${fmt(r.closure)} (${fmt(r.closureDR)})  ${String(r.wins).padStart(2)}/${SEEDS.length}`,
    );
  }

  console.log("\nField noise, µT");
  console.log("noise      ATE (DR)      wins");
  for (const fieldNoise of [0.1, 0.5, 2, 5, 15]) {
    const r = over({ traj: "figure8", fieldNoise });
    console.log(`${String(fieldNoise).padEnd(9)} ${fmt(r.ate)} (${fmt(r.ateDR)})  ${String(r.wins).padStart(2)}/${SEEDS.length}`);
  }
}

function sweep() {
  console.log("Cost against accuracy, on the loop");
  console.log("particles    m    ATE     closure   ms/step   MB");
  for (const particles of [50, 100, 200, 400]) {
    for (const m of [48, 96, 192]) {
      const r = runOne({ ...BASE, traj: "loop", particles, m });
      const mb = (particles * m * m * 8) / 1048576;
      console.log(
        `${String(particles).padStart(9)}  ${String(m).padStart(3)}  ${fmt(r.ate)}   ` +
          `${fmt(r.closure)}    ${fmt(r.msPerStep, 1)}   ${mb.toFixed(0)}`,
      );
    }
  }
}

if (process.argv[2] === "sweep") sweep();
else standard();
