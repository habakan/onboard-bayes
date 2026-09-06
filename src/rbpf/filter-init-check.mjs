import { Rbpf, DEFAULTS } from "./filter.ts";

// Dead reckoning only, which is what a browser without the native plugin runs. The
// weights are never touched there, so an unnormalised initialisation shows up as a
// position multiplied by the particle count.
const f = new Rbpf({...DEFAULTS, domain:{L:[15,15,3]}, particles:100, m:64,
  noise:2.0, sigma:30, ell:1.5, odomNoise:{pos:0.05,yaw:0.05}, seed:1}, {x:[0,0,0],yaw:0});

const essStart = f.ess();
for (let i=0;i<18;i++) f.predict({ds:[0.55,0,0], dyaw:0});
const e = f.estimate();
const expect = 18*0.55;
const ok = Math.abs(e.x[0]-expect) < 1.0 && Math.abs(essStart-100) < 1e-6;
console.log(`predict only, no field update`);
console.log(`  initial ESS = ${essStart.toFixed(1)}  (expect 100)`);
console.log(`  x = ${e.x[0].toFixed(2)}  (expect ${expect.toFixed(2)})`);
console.log(`  ${ok ? "ok" : "FAILED"}`);
if (!ok) process.exit(1);
