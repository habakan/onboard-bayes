import { makeBasis, fieldDesign } from "./basis.ts";
import { Rbpf, DEFAULTS } from "./filter.ts";
import { detectSteps, toOdometry, STEP_DEFAULTS } from "./pdr.ts";

const G=9.80665, HZ=50, K_TRUE=0.45, N_STEPS=120;
const domain={L:[4,4,1.5]};
const basis=makeBasis(domain,128);
const wTrue=Float64Array.from({length:basis.m},(_,j)=>1.5*Math.sin(j*1.7)/(1+j*0.25));
const Ht=new Float64Array(3*basis.m);
const field=p=>{fieldDesign(basis,p,Ht);
  return [0,1,2].map(d=>{let a=0;for(let j=0;j<basis.m;j++)a+=Ht[d*basis.m+j]*wTrue[j];return a;});};

const samples=[];
for(let i=0;i<N_STEPS/2.0*HZ;i++){const t=i/HZ;
  const turn=(t>12&&t<20)?0.35:0;
  const gait=1.5*(Math.sin(2*Math.PI*2.0*t)+0.3*Math.sin(4*Math.PI*2.0*t));
  const nz=k=>0.15*Math.sin((i+k)*7.31);
  samples.push({ax:nz(0),ay:nz(1),az:G+gait+nz(2),wz:turn,t:i*1000/HZ});}
const steps=detectSteps(samples,STEP_DEFAULTS);
const odoTrue=toOdometry(steps,K_TRUE);
let tp={x:[0,0,0],yaw:0}; const truth=[{x:[0,0,0],yaw:0}];
for(const o of odoTrue){tp={x:[...tp.x],yaw:tp.yaw+o.dyaw};
  const c=Math.cos(tp.yaw),s=Math.sin(tp.yaw);
  tp.x[0]+=c*o.ds[0]-s*o.ds[1];tp.x[1]+=s*o.ds[0]+c*o.ds[1];truth.push(tp);}

console.log("How much a wrong stride constant costs, and where the map stops helping");
console.log("k err   DR mean   RBPF mean  gain    DR final  RBPF final");
for (const pct of [0, 2, 5, 12, 25]) {
  const odo=toOdometry(steps,K_TRUE*(1+pct/100));
  const f=new Rbpf({...DEFAULTS,domain,m:128,particles:200,noise:0.03,sigma:1.0,ell:0.8,
                    odomNoise:{pos:0.03,yaw:0.01},seed:5},truth[0]);
  let dr={x:[0,0,0],yaw:0},drE=0,rbE=0,drF=0,rbF=0;
  for(let i=0;i<odo.length;i++){
    dr.yaw+=odo[i].dyaw;const c=Math.cos(dr.yaw),s=Math.sin(dr.yaw);
    dr.x[0]+=c*odo[i].ds[0]-s*odo[i].ds[1];dr.x[1]+=s*odo[i].ds[0]+c*odo[i].ds[1];
    f.predict(odo[i]);
    f.update(field(truth[i+1].x).map(v=>v+0.03*Math.sin(v*91.7)));
    const e=f.estimate();
    drF=Math.hypot(dr.x[0]-truth[i+1].x[0],dr.x[1]-truth[i+1].x[1]);
    rbF=Math.hypot(e.x[0]-truth[i+1].x[0],e.x[1]-truth[i+1].x[1]);
    drE+=drF; rbE+=rbF;
  }
  const d=drE/odo.length, r=rbE/odo.length;
  console.log(`${String(pct).padStart(4)}%  ${d.toFixed(3).padStart(7)}  ${r.toFixed(3).padStart(8)}  ${(d/r).toFixed(2)}x  ${drF.toFixed(3).padStart(7)}  ${rbF.toFixed(3).padStart(8)}`);
}
