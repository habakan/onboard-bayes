import { detectSteps, strideLength, STEP_DEFAULTS } from "./pdr.ts";

const G = 9.80665, HZ = 50;

/** 既知の歩数・歩調で歩行加速度を合成する */
function walk(nSteps, stepHz, ampG, turnRate = 0) {
  const out = [], dt = 1000 / HZ;
  const total = Math.ceil((nSteps / stepHz) * HZ);
  for (let i = 0; i < total; i++) {
    const t = i / HZ;
    // 歩行の主成分（上下動）+ 高調波 + ノイズ
    const gait = ampG * (Math.sin(2*Math.PI*stepHz*t) + 0.3*Math.sin(4*Math.PI*stepHz*t));
    const nz = k => 0.15 * Math.sin((i + k) * 7.31);
    out.push({ ax: nz(0), ay: nz(1), az: G + gait + nz(2), wz: turnRate, t: i * dt });
  }
  return out;
}

console.log("歩数検知");
for (const [label, n, hz, amp] of [
  ["ゆっくり 1.5歩/s", 20, 1.5, 1.2],
  ["通常   2.0歩/s", 30, 2.0, 1.5],
  ["速い   2.5歩/s", 30, 2.5, 2.0],
  ["弱い信号 (amp 0.6)", 20, 2.0, 0.6],
]) {
  const s = detectSteps(walk(n, hz, amp), STEP_DEFAULTS);
  const err = s.length - n;
  console.log(`  ${label.padEnd(20)} 真 ${String(n).padStart(2)} 歩 → 検知 ${String(s.length).padStart(2)} 歩  (${err >= 0 ? "+" : ""}${err})`);
}

console.log("\n静止時の誤検知");
const still = Array.from({length: 300}, (_, i) => ({
  ax: 0.02*Math.sin(i*7.3), ay: 0.02*Math.cos(i*5.1), az: G + 0.02*Math.sin(i*3.7), wz: 0, t: i*20 }));
console.log(`  静止 6 秒 → 検知 ${detectSteps(still, STEP_DEFAULTS).length} 歩 (0 が正しい)`);

console.log("\n旋回の積分");
const turn = detectSteps(walk(10, 2.0, 1.5, 0.5), STEP_DEFAULTS);  // 0.5 rad/s
const totalYaw = turn.reduce((a, s) => a + s.dyaw, 0);
console.log(`  0.5 rad/s × 5 秒 = 2.5 rad が期待値 → 積算 ${totalYaw.toFixed(3)} rad`);

console.log("\n歩幅 (Weinberg, k=0.45)");
for (const amp of [0.5, 1.5, 3.0, 6.0]) {
  console.log(`  振幅 ${amp.toFixed(1)} → ${strideLength(amp, 0.45).toFixed(3)} m`);
}

// --- PDR → RBPF の通し ---
import { makeBasis, fieldDesign } from "./basis.ts";
import { Rbpf, DEFAULTS } from "./filter.ts";
import { toOdometry } from "./pdr.ts";

console.log("\nPDR → RBPF 通し");
const domain = { L: [4, 4, 1.5] };
const basis = makeBasis(domain, 128);
const wTrue = Float64Array.from({length: basis.m}, (_,j)=>1.5*Math.sin(j*1.7)/(1+j*0.25));
const Ht = new Float64Array(3*basis.m);
const field = p => { fieldDesign(basis, p, Ht);
  return [0,1,2].map(d=>{let a=0;for(let j=0;j<basis.m;j++)a+=Ht[d*basis.m+j]*wTrue[j];return a;}); };

// 真の歩行: 一定歩調で歩き、途中で曲がる
const K_TRUE = 0.45, N_STEPS = 120;
const samples = [];
for (let i = 0; i < N_STEPS / 2.0 * HZ; i++) {
  const t = i / HZ;
  const turn = (t > 12 && t < 20) ? 0.35 : 0;
  const gait = 1.5*(Math.sin(2*Math.PI*2.0*t) + 0.3*Math.sin(4*Math.PI*2.0*t));
  const nz = k => 0.15*Math.sin((i+k)*7.31);
  samples.push({ ax: nz(0), ay: nz(1), az: G + gait + nz(2), wz: turn, t: i*1000/HZ });
}
const steps = detectSteps(samples, STEP_DEFAULTS);
const odo = toOdometry(steps, K_TRUE);

// 真の軌跡を同じオドメトリから作る（PDR が理想的なら一致するはず）
let tp = {x:[0,0,0], yaw:0}; const truth=[{x:[0,0,0], yaw:0}];
for (const o of odo) { tp = {x:[...tp.x], yaw: tp.yaw + o.dyaw};
  const c=Math.cos(tp.yaw), s=Math.sin(tp.yaw);
  tp.x[0]+=c*o.ds[0]-s*o.ds[1]; tp.x[1]+=s*o.ds[0]+c*o.ds[1]; truth.push(tp); }

// 歩幅係数を 12% 誤って与える（未較正の状態を模す）
const odoBad = toOdometry(steps, K_TRUE * 1.12);
const f = new Rbpf({...DEFAULTS, domain, m:128, particles:200, noise:0.03,
                    sigma:1.0, ell:0.8, odomNoise:{pos:0.03, yaw:0.01}, seed:5}, truth[0]);
let dr = {x:[0,0,0], yaw:0}; let drE=0, rbE=0;
const t0=performance.now();
for (let i=0;i<odoBad.length;i++){
  dr.yaw+=odoBad[i].dyaw; const c=Math.cos(dr.yaw),s=Math.sin(dr.yaw);
  dr.x[0]+=c*odoBad[i].ds[0]-s*odoBad[i].ds[1]; dr.x[1]+=s*odoBad[i].ds[0]+c*odoBad[i].ds[1];
  f.predict(odoBad[i]);
  f.update(field(truth[i+1].x).map(v=>v+0.03*Math.sin(v*91.7)));
  const e=f.estimate();
  drE+=Math.hypot(dr.x[0]-truth[i+1].x[0], dr.x[1]-truth[i+1].x[1]);
  rbE+=Math.hypot(e.x[0]-truth[i+1].x[0], e.x[1]-truth[i+1].x[1]);
}
const ms=performance.now()-t0;
console.log(`  検知 ${steps.length} 歩 (真 ${N_STEPS})   歩幅係数を 12% 誤設定`);
console.log(`  デッドレコニング: 平均誤差 ${(drE/odoBad.length).toFixed(3)} m`);
console.log(`  RBPF            : 平均誤差 ${(rbE/odoBad.length).toFixed(3)} m`);
console.log(`  ${ms.toFixed(0)} ms / ${odoBad.length} 歩  (${(ms/odoBad.length).toFixed(1)} ms/歩)`);
