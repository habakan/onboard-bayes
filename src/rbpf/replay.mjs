/**
 * Replays a recording from `recordings/` through step detection and sweeps the
 * detection settings against the step count noted at capture time.
 *
 * Walking once and replaying is the only way to tune this at any speed; the
 * alternative is one walk per parameter change.
 *
 * Usage: npm run replay -- recordings/<file>.json
 *        npm run replay                       (newest recording)
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { detectSteps, STEP_DEFAULTS } from "./pdr.ts";

function newest(dir) {
  const files = readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
  if (files.length === 0) throw new Error(`no recordings in ${dir}`);
  return path.join(dir, files[files.length - 1]);
}

const file = process.argv[2] ?? newest("recordings");
const rec = JSON.parse(readFileSync(file, "utf8"));
// The columns are named wx,wy,wz but hold alpha,beta,gamma; see tracking.ts for why
// those land on x,y,z with a sign flip.
const K = -Math.PI / 180;
const samples = rec.motion.map(([t, ax, ay, az, al, be, ga]) => ({
  ax, ay, az, wx: al * K, wy: be * K, wz: ga * K, t,
}));
const truth = rec.trueSteps || 0;

const dur = (samples[samples.length - 1].t - samples[0].t) / 1000;
console.log(file);
console.log(`  ${samples.length} samples over ${dur.toFixed(1)} s  (${(samples.length / dur).toFixed(0)} Hz)`);
console.log(`  device: ${(rec.device ?? "?").slice(0, 70)}`);
console.log(`  steps counted at capture: ${truth || "not recorded"}\n`);

const rows = [];
for (const threshold of [0.15, 0.2, 0.3, 0.4, 0.5, 0.7]) {
  for (const minIntervalMs of [250, 300, 350]) {
    for (const alpha of [0.15, 0.25, 0.4]) {
      const n = detectSteps(samples, { threshold, minIntervalMs, alpha }).length;
      rows.push({
        threshold,
        minIntervalMs,
        alpha,
        n,
        err: truth ? (100 * (n - truth)) / truth : NaN,
        isDefault:
          threshold === STEP_DEFAULTS.threshold &&
          minIntervalMs === STEP_DEFAULTS.minIntervalMs &&
          alpha === STEP_DEFAULTS.alpha,
      });
    }
  }
}

const fmt = (r) =>
  `  thr ${r.threshold.toFixed(2)}  gap ${r.minIntervalMs}  alpha ${r.alpha.toFixed(2)}  ` +
  `${String(r.n).padStart(4)} steps  ${truth ? `${r.err >= 0 ? "+" : ""}${r.err.toFixed(1)}%` : "—"}`;

const current = rows.find((r) => r.isDefault);
if (current) console.log(`current default:\n${fmt(current)}\n`);

if (!truth) {
  console.log("No step count was recorded, so there is nothing to sweep against.");
} else {
  // How much of the grid lands inside tolerance says more than the single best cell,
  // which on one recording can just be luck.
  const ok = rows.filter((r) => Math.abs(r.err) <= 5);
  const best = rows.reduce((a, b) => (Math.abs(b.err) < Math.abs(a.err) ? b : a));
  console.log(`${ok.length}/${rows.length} settings land within 5%.`);
  console.log(`best:\n${fmt(best)}`);
  if (current) {
    console.log(
      Math.abs(current.err) <= 5
        ? "\nThe default is already inside tolerance here."
        : `\nThe default is ${current.err >= 0 ? "+" : ""}${current.err.toFixed(1)}% here — worth changing.`,
    );
  }
}
