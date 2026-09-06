import { makeBasis, fieldDesign } from "./basis.ts";
import { Rbpf, DEFAULTS } from "./filter.ts";

const domain = { L: [3, 3, 1.5] };
const basis = makeBasis(domain, 128);
const wTrue = Float64Array.from({length: basis.m}, (_,j)=>1.5*Math.sin(j*1.7)/(1+j*0.25));
const Ht = new Float64Array(3*basis.m);
const trueField = p => { fieldDesign(basis, p, Ht);
  return [0,1,2].map(d=>{let a=0;for(let j=0;j<basis.m;j++)a+=Ht[d*basis.m+j]*wTrue[j];return a;}); };

// 真の軌跡: 部屋を8の字で2周
const STEPS = 400, dt = 0.05;
const truth = [];
for (let i=0;i<STEPS;i++){ const t=i*dt;
  truth.push({x:[1.8*Math.sin(t*0.5), 1.2*Math.sin(t), 0.0], yaw: t*0.5}); }

// オドメトリにバイアスを入れる（実機のドリフトを模す）
let seed=7; const rnd=()=>{seed=(seed*1103515245+12345)&0x7fffffff;return seed/0x7fffffff;};
const g=()=>Math.sqrt(-2*Math.log(Math.max(rnd(),1e-9)))*Math.cos(2*Math.PI*rnd());
const BIAS = 0.004;
const odo=[]; 
for (let i=1;i<STEPS;i++){
  const dx=truth[i].x[0]-truth[i-1].x[0], dy=truth[i].x[1]-truth[i-1].x[1];
  const yaw=truth[i-1].yaw, c=Math.cos(-yaw), s=Math.sin(-yaw);
  odo.push({ ds:[c*dx-s*dy+BIAS+0.004*g(), s*dx+c*dy+0.004*g(), 0],
             dyaw: truth[i].yaw-truth[i-1].yaw+0.0015+0.002*g() });
}

// A: デッドレコニングのみ
let dr={x:[...truth[0].x], yaw:truth[0].yaw}; const drErr=[];
for (const o of odo){ dr.yaw+=o.dyaw; const c=Math.cos(dr.yaw),s=Math.sin(dr.yaw);
  dr.x[0]+=c*o.ds[0]-s*o.ds[1]; dr.x[1]+=s*o.ds[0]+c*o.ds[1]; drErr.push(dr); }

// B: RBPF
const f = new Rbpf({...DEFAULTS, domain, m:128, particles:200, noise:0.03, sigma:1.0, ell:0.8,
                    odomNoise:{pos:0.01, yaw:0.004}, seed:3}, truth[0]);
const t0=performance.now(); const rbErr=[];
for (let i=0;i<odo.length;i++){
  f.predict(odo[i]);
  const y = trueField(truth[i+1].x).map(v=>v+0.03*g());
  f.update(y);
  rbErr.push(f.estimate());
}
const ms=performance.now()-t0;

const err=(e,i)=>Math.hypot(e.x[0]-truth[i+1].x[0], e.x[1]-truth[i+1].x[1]);
const fin = a => err(a[a.length-1], a.length-1);
const mean = a => a.reduce((s,e,i)=>s+err(e,i),0)/a.length;
console.log(`ステップ ${odo.length}、粒子 ${f.opts.particles}、基底 ${f.opts.m}`);
console.log(`  ${ms.toFixed(0)} ms  (${(ms/odo.length).toFixed(2)} ms/step, ${(1000/(ms/odo.length)).toFixed(0)} Hz)`);
console.log(`  デッドレコニング: 平均誤差 ${mean(drErr).toFixed(3)} m   最終 ${fin(drErr).toFixed(3)} m`);
console.log(`  RBPF            : 平均誤差 ${mean(rbErr).toFixed(3)} m   最終 ${fin(rbErr).toFixed(3)} m`);
console.log(`  ESS = ${f.ess().toFixed(1)} / ${f.opts.particles}`);
