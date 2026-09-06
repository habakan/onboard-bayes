import { makeBasis, priorVariance, phi, fieldDesign } from "./basis.ts";

const dom = { L: [2, 2, 2] };
const b = makeBasis(dom, 64);
const H = new Float64Array(3 * b.m);
const x = [0.31, -0.72, 0.44];

// 1. H = -∇φ を有限差分で検証
fieldDesign(b, x, H);
const h = 1e-6;
let worst = 0;
for (let j = 0; j < b.m; j++) {
  for (let d = 0; d < 3; d++) {
    const xp = [...x], xm = [...x];
    xp[d] += h; xm[d] -= h;
    const fd = -(phi(b, j, xp) - phi(b, j, xm)) / (2 * h);
    worst = Math.max(worst, Math.abs(fd - H[d * b.m + j]));
  }
}
console.log(`1. 勾配 vs 有限差分   最大誤差 = ${worst.toExponential(2)}`);

// 2. 場が回転なしか（B = -∇φ なら数学的に curl B = 0）
//    ランダムな重みで場を作り、curl を有限差分で測る
const w = Float64Array.from({length: b.m}, (_, j) => Math.sin(j * 1.7));
function field(p) {
  const Hp = new Float64Array(3 * b.m);
  fieldDesign(b, p, Hp);
  return [0,1,2].map(d => { let a=0; for (let j=0;j<b.m;j++) a += Hp[d*b.m+j]*w[j]; return a; });
}
const dh = 1e-5;
const d_ = (d, comp) => {
  const p1=[...x], p2=[...x]; p1[d]+=dh; p2[d]-=dh;
  return (field(p1)[comp]-field(p2)[comp])/(2*dh);
};
const curl = [d_(1,2)-d_(2,1), d_(2,0)-d_(0,2), d_(0,1)-d_(1,0)];
const mag = Math.hypot(...field(x));
console.log(`2. curl B (回転)      = ${curl.map(v=>v.toExponential(1)).join(", ")}   |B|=${mag.toFixed(3)}`);

// 3. 事前分散が長さスケールで単調に集中するか
for (const ell of [0.3, 1.0, 3.0]) {
  const S = priorVariance(b, 1.0, ell);
  const tot = S.reduce((a,v)=>a+v,0);
  const top8 = Array.from(S).slice(0,8).reduce((a,v)=>a+v,0);
  console.log(`3. ell=${ell}  先頭8項が全体の ${(100*top8/tot).toFixed(1)}%`);
}
console.log(`\n基底数=${b.m}  最小λ=${b.lambda[0].toFixed(3)}  最大λ=${b.lambda[b.m-1].toFixed(3)}`);
