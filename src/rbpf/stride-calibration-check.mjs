import { readFile } from "node:fs/promises";
const R="/Users/habano/work/oss/onboard-bayes";
const {default:init, StanModel} = await import(`${R}/node_modules/stanwasm/index.js`);
await init({module_or_path: await readFile(`${R}/node_modules/stanwasm/pkg/stanwasm_bg.wasm`)});
const { STRIDE_K } = await import(`${R}/src/models.ts`);

const K_TRUE = 0.45, TAU_TRUE = 0.02;
// W 回の歩行。ペースが違うので k が少しばらつく
function trial(W, stepsPerWalk, seed) {
  let s=seed; const rnd=()=>{s=(s*1103515245+12345)&0x7fffffff;return s/0x7fffffff;};
  const g=()=>Math.sqrt(-2*Math.log(Math.max(rnd(),1e-9)))*Math.cos(2*Math.PI*rnd());
  const amp_sum=[], distance=[];
  for (let w=0; w<W; w++) {
    const kw = K_TRUE + TAU_TRUE*g();
    let sum=0;
    for (let i=0;i<stepsPerWalk;i++) sum += Math.pow(1.2 + 1.6*rnd(), 0.25);
    amp_sum.push(sum);
    distance.push(kw*sum + 0.05*g());   // 巻尺の測定誤差 5cm
  }
  return {W, amp_sum, distance};
}

console.log("Needs stanwasm >= 0.1.2: indexed assignment and shaped transformed parameters.");
console.log("walks   k_pop est        error   tau      sigma    time");
for (const W of [2, 3, 5, 8]) {
  const data = trial(W, 40, 11+W);
  const src = STRIDE_K.stan;
  const init = new Float64Array([-0.8, -3.0, ...Array(W).fill(0.1), -1.6]);
  const m = new StanModel(src, JSON.stringify(data));
  const t0=performance.now();
  const s = m.sample(init, 500, 500, 29n);
  const ms=performance.now()-t0, np=m.n_params, names=m.paramNames();
  const mean=name=>{const i=names.indexOf(name); let a=0;
    for(let k=500;k<1000;k++){const c=m.constrainDraw(s.slice(k*np,(k+1)*np)); a+=c[i];} return a/500;};
  const k=mean("k_pop"), err=100*Math.abs(k-K_TRUE)/K_TRUE;
  const flag = err<=5 ? "ok" : "OVER 5%";
  console.log(`   ${W}     ${k.toFixed(4)}  (真 ${K_TRUE})  ${err.toFixed(1)}% ${flag.padStart(8)}  ${mean("tau").toFixed(4)}  ${mean("sigma").toFixed(4)}  ${ms.toFixed(0)}ms`);
  m.free();
}
