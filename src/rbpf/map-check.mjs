import { makeBasis, fieldDesign } from "./basis.ts";
import { MagMap } from "./map.ts";

const dom = { L: [2, 2, 2] };
const basis = makeBasis(dom, 128);
// 真の場: 同じ基底に決定的な重みを入れて作る（curl-free が保証される）
const wTrue = Float64Array.from({length: basis.m}, (_, j) => 2.0 * Math.sin(j * 1.7) / (1 + j * 0.3));
const Htmp = new Float64Array(3 * basis.m);
function trueField(p) {
  fieldDesign(basis, p, Htmp);
  return [0,1,2].map(d => { let a=0; for (let j=0;j<basis.m;j++) a += Htmp[d*basis.m+j]*wTrue[j]; return a; });
}

const map = new MagMap(basis, 1.0, 0.7);
const noise = 0.01;
// 軌跡: 箱の中を歩き回る
const path = [];
for (let i = 0; i < 300; i++) {
  const t = i * 0.05;
  path.push([1.2*Math.sin(t), 1.2*Math.sin(t*0.7+1), 0.6*Math.sin(t*0.31)]);
}
let logp = 0;
const t0 = performance.now();
for (const p of path) {
  const y = trueField(p).map(v => v + noise * Math.sin(v * 91.7));
  logp += map.update(p, y, noise * noise);
}
const ms = performance.now() - t0;

// 3種類の点で誤差を測り、汎化の効く範囲を見る
function rmse(pts) {
  let se=0, sv=0;
  for (const p of pts) { const t=trueField(p), e=map.predict(p);
    for (let d=0;d<3;d++){ se+=(t[d]-e[d])**2; sv+=t[d]**2; } }
  return [Math.sqrt(se/(3*pts.length)), Math.sqrt(se/sv)*100];
}
const onPath = path.slice(0, 50);
const nearPath = path.slice(0,50).map(p => p.map(v => v + 0.05*Math.sin(v*13)));
const farPts = Array.from({length:50},(_,i)=>[1.5*Math.cos(i*0.9),1.5*Math.sin(i*1.3),0.8*Math.cos(i*0.5)]);
for (const [label, pts] of [["軌跡上", onPath], ["軌跡の近く(±0.05)", nearPath], ["離れた点", farPts]]) {
  const [r, pct] = rmse(pts);
  console.log(`  ${label.padEnd(20)} RMSE=${r.toFixed(4)}  相対 ${pct.toFixed(1)}%`);
}
// 軌跡がどれだけ箱を覆っているか
const ext = [0,1,2].map(d => { const vs=path.map(p=>p[d]); return [Math.min(...vs), Math.max(...vs)]; });
console.log(`\n軌跡の範囲: ${ext.map(e=>`[${e[0].toFixed(2)}, ${e[1].toFixed(2)}]`).join(" ")}   箱=[-2,2]^3`);
console.log(`累積対数尤度 = ${logp.toFixed(1)}`);
