/**
 * Why the RBPF underperforms dead reckoning while its machinery is sound.
 *
 * A particle that builds its own map from a displaced trajectory writes each
 * observation to the wrong place and is then consistent with itself. The likelihood
 * still prefers the truth, but by so little that 150 particles cannot be separated by
 * it — five orders of magnitude less than the same displacement scored against a map
 * that was already finished.
 *
 * The map is flexible enough to explain any trajectory hypothesis, so on a first visit
 * it carries no information about position. Only revisits can supply that, and this
 * implementation does not exploit them.
 */
import { makeBasis } from "./basis.ts";
import { MagMap } from "./map.ts";
import { TRAJECTORIES, makeField } from "./scenarios.mjs";
const DOMAIN={L:[10,10,2]}, M=96;
const basis=makeBasis(DOMAIN,M);
const field=makeField(DOMAIN,M,1,42);
const truth=TRAJECTORIES.loop(300);
console.log("Log-likelihood gap between the true position and a displaced one");
console.log("offset   self-built map   finished map");
// 完成マップ
const oracle=new MagMap(basis,30,1.2);
for(let pass=0;pass<3;pass++) for(let i=0;i<truth.length;i++) oracle.update(truth[i].x, field(truth[i].x), 0.01);
for (const OFF of [0.1,0.25,0.5,1.0,2.0]) {
  // 自作: ずれた位置で学習した粒子の累積尤度
  const g=new MagMap(basis,30,1.2), b=new MagMap(basis,30,1.2);
  let a=0,c=0;
  for(let i=0;i<200;i++){ const y=field(truth[i].x);
    a+=g.update(truth[i].x,y,0.25); c+=b.update([truth[i].x[0]+OFF,truth[i].x[1],0],y,0.25); }
  // 完成マップ: 同じ 200 点で、真の位置 vs ずれた位置の尤度差
  let og=0,ob=0;
  for(let i=0;i<200;i++){ const y=field(truth[i].x);
    const p1=oracle.predict(truth[i].x), p2=oracle.predict([truth[i].x[0]+OFF,truth[i].x[1],0]);
    for(let d=0;d<3;d++){ og+=-0.5*(y[d]-p1[d])**2/0.25; ob+=-0.5*(y[d]-p2[d])**2/0.25; } }
  console.log(`  ${OFF.toFixed(2)}    ${(a-c).toFixed(1).padStart(10)}  ${(og-ob).toFixed(1).padStart(12)}`);
}
