/**
 * Measures how much a recording's magnetic environment moves between its two halves.
 *
 * The intended capture is a phone held still on the machine under test, motors off
 * for the first half and running for the second. Whatever the second half moves by is
 * the disturbance the map would have to survive.
 *
 * Raw axes are only available in the native build, so the compass heading is used as
 * the fallback: it is derived from the magnetometer, so a heading that wanders while
 * the phone is still is the same disturbance seen indirectly.
 *
 * Usage: npm run disturbance -- recordings/<file>.json
 *        npm run disturbance                        (newest)
 */
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";

function newest(dir) {
  const files = readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
  if (files.length === 0) throw new Error(`no recordings in ${dir}`);
  return path.join(dir, files[files.length - 1]);
}

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => {
  const m = mean(a);
  return Math.sqrt(mean(a.map((v) => (v - m) ** 2)));
};

/** Circular mean and spread, so 359° and 1° are 2° apart rather than 358°. */
function circular(deg) {
  const r = deg.map((d) => (d * Math.PI) / 180);
  const s = mean(r.map(Math.sin));
  const c = mean(r.map(Math.cos));
  const centre = (Math.atan2(s, c) * 180) / Math.PI;
  const spread = (Math.sqrt(-2 * Math.log(Math.min(1, Math.hypot(s, c)))) * 180) / Math.PI;
  return { centre: (centre + 360) % 360, spread };
}

function split(rows) {
  const half = rows[0][0] + (rows[rows.length - 1][0] - rows[0][0]) / 2;
  return [rows.filter((r) => r[0] < half), rows.filter((r) => r[0] >= half)];
}

const file = process.argv[2] ?? newest("recordings");
const rec = JSON.parse(readFileSync(file, "utf8"));
console.log(`${file}`);
console.log(`  device: ${(rec.device ?? "?").slice(0, 70)}\n`);

let verdict = null;

if (rec.field?.length > 20) {
  const [off, on] = split(rec.field);
  const axis = (rows, i) => rows.map((r) => r[i]);
  const magOf = (rows) => rows.map((r) => Math.hypot(r[1], r[2], r[3]));
  const shift = Math.hypot(
    mean(axis(on, 1)) - mean(axis(off, 1)),
    mean(axis(on, 2)) - mean(axis(off, 2)),
    mean(axis(on, 3)) - mean(axis(off, 3)),
  );
  console.log(`raw field, ${rec.field.length} samples`);
  console.log(`  |B| first half  ${mean(magOf(off)).toFixed(1)} µT  (sd ${sd(magOf(off)).toFixed(2)})`);
  console.log(`  |B| second half ${mean(magOf(on)).toFixed(1)} µT  (sd ${sd(magOf(on)).toFixed(2)})`);
  console.log(`  vector shift between halves: ${shift.toFixed(1)} µT`);
  verdict = shift;
} else if (rec.heading?.length > 20) {
  const [off, on] = split(rec.heading);
  const a = circular(off.map((r) => r[1]));
  const b = circular(on.map((r) => r[1]));
  let d = Math.abs(b.centre - a.centre);
  if (d > 180) d = 360 - d;
  console.log(`compass heading, ${rec.heading.length} samples (no raw axes without the native build)`);
  console.log(`  first half  ${a.centre.toFixed(1)}°  (spread ${a.spread.toFixed(1)}°)`);
  console.log(`  second half ${b.centre.toFixed(1)}°  (spread ${b.spread.toFixed(1)}°)`);
  console.log(`  heading moved ${d.toFixed(1)}°, spread went ${a.spread.toFixed(1)}° → ${b.spread.toFixed(1)}°`);

  // Only a lower bound, and a weak one. Heading is an angle in the horizontal plane,
  // so a disturbance pointing along the existing horizontal field changes its length
  // without turning it: a synthetic 25 µT added this way moves the heading 4.4°, which
  // back-converts to 3.5 µT. Never call a recording clean on this number alone.
  const moved = Math.max(d, b.spread);
  const lower = 45 * Math.tan((moved * Math.PI) / 180);
  console.log(`  at least ~${lower.toFixed(1)} µT of added field — a lower bound only`);
  console.log(`  a disturbance along the existing field barely turns the compass, so the`);
  console.log(`  true figure can be several times this. The native build measures it directly.`);

  const call =
    moved < 2
      ? "probably quiet, but confirm with raw axes before trusting it"
      : moved < 8
        ? "clearly disturbed — measure properly with the native build"
        : "badly disturbed — this mounting will not work";
  console.log(`\nheading moved ${moved.toFixed(1)}°: ${call}`);
  console.log("Hold the phone still for the whole recording, or this measures motion instead.");
  verdict = null;
} else {
  console.log("This recording has neither field nor heading samples.");
}

if (verdict !== null) {
  const call =
    verdict < 5
      ? "usable — small against a 45 µT ambient field"
      : verdict < 15
        ? "marginal — try raising the phone away from the motors, then measure again"
        : "too noisy for magnetic mapping from this mounting";
  console.log(`\n${verdict.toFixed(1)} µT: ${call}`);
  console.log("Hold the phone still for the whole recording, or this measures motion instead.");
}
