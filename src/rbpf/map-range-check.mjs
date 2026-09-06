/**
 * How far past its observations the map can still predict. This is what decides
 * whether particle weights carry position information at all: if particles differ by
 * less than this range, they all predict the same field and the likelihood ranks
 * noise instead of position.
 */
import { makeBasis } from "./basis.ts";
import { MagMap } from "./map.ts";
import { TRAJECTORIES, makeField } from "./scenarios.mjs";

const DOMAIN = { L: [10, 10, 2] };
const field = makeField(DOMAIN, 128, 1, 42);
const truth = TRAJECTORIES.loop(300);
const STEP_M = 0.084;

console.log("Prediction error against distance beyond the last observation (150 obs)");
console.log("  the field's own spatial variation is about 15 µT\n");
console.log("ell     0.1m    0.4m    0.8m    1.6m    3.2m");
for (const ell of [0.3, 0.6, 1.2, 2.5]) {
  const map = new MagMap(makeBasis(DOMAIN, 96), 30, ell);
  for (let i = 0; i < 150; i++) map.update(truth[i].x, field(truth[i].x), 0.25);
  const errs = [1, 5, 10, 19, 38].map((ahead) => {
    const p = truth[150 + ahead].x;
    const t = field(p);
    const e = map.predict(p);
    return Math.sqrt([0, 1, 2].reduce((a, d) => a + (t[d] - e[d]) ** 2, 0) / 3);
  });
  console.log(`${ell.toFixed(1).padStart(3)}  ` + errs.map((v) => v.toFixed(2).padStart(6)).join("  "));
}
console.log(`\nA step is ${(STEP_M * 100).toFixed(0)} cm. Particles that differ by less than the`);
console.log("range above cannot be told apart by their field predictions.");
