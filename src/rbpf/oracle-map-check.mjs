/**
 * Separates the two halves of the RBPF: give the particles a finished map and let them
 * only weight against it. If this tracks well, the weighting, resampling and estimation
 * are sound and any remaining error belongs to building the map online.
 *
 * It does track well — 0.012 m against dead reckoning's 0.541 m — which is what places
 * the current defect in map construction rather than in the filter machinery.
 */
import { makeBasis, fieldDesign } from "./basis.ts";
import { MagMap } from "./map.ts";
import { TRAJECTORIES, makeField, odometryFrom, rng, gauss } from "./scenarios.mjs";
const DOMAIN={L:[10,10,2]}, M=96, NP=150;
const field=makeField(DOMAIN,128,1,42);
const truth=TRAJECTORIES.loop(300);
const odo=odometryFrom(truth,{scaleBias:0.03,posNoise:0.01,yawNoise:0.004,seed:5});
const basis=makeBasis(DOMAIN,M);

const oracle=new MagMap(basis,30,1.2);
for(let pass=0;pass<3;pass++) for(let i=0;i<truth.length;i++) oracle.update(truth[i].x, field(truth[i].x), 0.01);
let se=0; for(let i=0;i<300;i+=7){const t=field(truth[i].x),e=oracle.predict(truth[i].x);
  se+=[0,1,2].reduce((a,k)=>a+(t[k]-e[k])**2,0);}
console.log(`finished map residual RMS = ${Math.sqrt(se/(3*43)).toFixed(3)} µT\n`);

// 自前の最小粒子フィルタ: マップ固定、重み付けと系統再サンプリングのみ
const u=rng(11);
let poses=Array.from({length:NP},()=>({x:[...truth[0].x],yaw:truth[0].yaw}));
let lw=new Float64Array(NP).fill(-Math.log(NP));
const H=new Float64Array(3*M);
const d=(a,b)=>Math.hypot(a.x[0]-b.x[0],a.x[1]-b.x[1]);
const NOISE=0.5, PN=0.05, YN=0.02;
let err=0, dre=0; let dr={x:[...truth[0].x],yaw:truth[0].yaw};
for(let i=0;i<odo.length;i++){
  for(const p of poses){
    p.yaw+=odo[i].dyaw+YN*gauss(u);
    const c=Math.cos(p.yaw), s=Math.sin(p.yaw);
    p.x[0]+=c*odo[i].ds[0]-s*odo[i].ds[1]+PN*gauss(u);
    p.x[1]+=s*odo[i].ds[0]+c*odo[i].ds[1]+PN*gauss(u);
  }
  const y=field(truth[i+1].x);
  for(let k=0;k<NP;k++){
    fieldDesign(basis,poses[k].x,H);
    let lp=0;
    for(let dd=0;dd<3;dd++){ let pred=0; for(let j=0;j<M;j++) pred+=H[dd*M+j]*oracle.mean[j];
      const r=y[dd]-pred; lp+=-0.5*(r*r)/(NOISE*NOISE); }
    lw[k]+=lp;
  }
  let mx=-Infinity; for(const v of lw) if(v>mx) mx=v;
  let sum=0; for(let k=0;k<NP;k++) sum+=Math.exp(lw[k]-mx);
  const off=mx+Math.log(sum); for(let k=0;k<NP;k++) lw[k]-=off;
  let s2=0; for(const v of lw){const w=Math.exp(v); s2+=w*w;}
  const ess=1/s2;
  if(ess<NP/2){
    const step=1/NP; let tgt=u()*step, acc=0; const np=[];
    for(let k=0;k<NP;k++){ acc+=Math.exp(lw[k]);
      while(tgt<acc&&np.length<NP){ np.push({x:[...poses[k].x],yaw:poses[k].yaw}); tgt+=step; } }
    while(np.length<NP) np.push({x:[...poses[NP-1].x],yaw:poses[NP-1].yaw});
    poses=np; lw.fill(-Math.log(NP));
  }
  let ex=0,ey=0; for(let k=0;k<NP;k++){const w=Math.exp(lw[k]); ex+=w*poses[k].x[0]; ey+=w*poses[k].x[1];}
  err+=Math.hypot(ex-truth[i+1].x[0], ey-truth[i+1].x[1]);
  dr.yaw+=odo[i].dyaw; const c=Math.cos(dr.yaw),s=Math.sin(dr.yaw);
  dr.x[0]+=c*odo[i].ds[0]-s*odo[i].ds[1]; dr.x[1]+=s*odo[i].ds[0]+c*odo[i].ds[1];
  dre+=d(dr,truth[i+1]);
}
console.log(`fixed map, weighting only : ${(err/odo.length).toFixed(3)} m mean error`);
console.log(`dead reckoning            : ${(dre/odo.length).toFixed(3)} m mean error`);
