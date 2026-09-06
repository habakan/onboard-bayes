export interface SensorModel {
  key: string;
  title: string;
  what: string;
  stan: string;
  /** Unconstrained starting point. Zero gradients are rejected by nuts-rs, so no
   *  entry here is allowed to sit exactly at a flat spot. */
  init: number[];
  paramsOfInterest: string[];
}

/**
 * Accelerometer bias: every reading should have magnitude g, so the offset that
 * reconciles the observed magnitudes with g is the bias.
 *
 * The capture has to cover several orientations. From one orientation only the
 * distance from the origin is observed, every point on a sphere fits equally well,
 * and NUTS walks that ridge — 8 s for an answer that is also wrong, against 64 ms
 * and the right answer once the phone has been turned over.
 */
export const ACCEL_BIAS: SensorModel = {
  key: "accel_bias",
  title: "Accelerometer bias",
  what: "Turn the phone through several faces while this runs — one orientation is not enough to identify the offsets.",
  stan: `data {
  int<lower=0> N;
  vector[N] ax;
  vector[N] ay;
  vector[N] az;
  real g;
}
parameters {
  real bx;
  real by;
  real bz;
  real<lower=0> sigma;
}
model {
  bx ~ normal(0, 1);
  by ~ normal(0, 1);
  bz ~ normal(0, 1);
  sigma ~ exponential(1);
  for (n in 1:N)
    g ~ normal(sqrt(square(ax[n] - bx) + square(ay[n] - by) + square(az[n] - bz)), sigma);
}`,
  init: [0.01, 0.01, 0.01, 0],
  paramsOfInterest: ["bx", "by", "bz", "sigma"],
};

/** Gyroscope bias at rest: the rate reads non-zero while the phone is still, and the
 *  mean of that is the bias. Two parameters, so this is the cheap end of the range. */
export const GYRO_BIAS: SensorModel = {
  key: "gyro_bias",
  title: "Gyroscope bias",
  what: "Hold the phone still. Estimates the resting drift on one axis.",
  stan: `data {
  int<lower=0> N;
  vector[N] w;
}
parameters {
  real bias;
  real<lower=0> sigma;
}
model {
  bias ~ normal(0, 10);
  sigma ~ exponential(1);
  w ~ normal(bias, sigma);
}`,
  init: [0.01, 0],
  paramsOfInterest: ["bias", "sigma"],
};

/**
 * Heading, with the wrap handled by fitting the mean direction's components rather
 * than the angle. iOS gives a fused `webkitCompassHeading` and no raw magnetometer,
 * so hard-iron calibration is out of reach from a WebView without native code.
 */
export const HEADING: SensorModel = {
  key: "heading",
  title: "Heading",
  what: "Point the phone one way and hold. Estimates the true bearing and its spread.",
  stan: `data {
  int<lower=0> N;
  vector[N] cx;
  vector[N] cy;
}
parameters {
  real hx;
  real hy;
  real<lower=0> sigma;
}
model {
  hx ~ normal(0, 2);
  hy ~ normal(0, 2);
  sigma ~ exponential(1);
  for (n in 1:N) {
    cx[n] ~ normal(hx, sigma);
    cy[n] ~ normal(hy, sigma);
  }
}
generated quantities {
  real bearing = atan2(hy, hx);
}`,
  init: [0.1, 0.1, 0],
  paramsOfInterest: ["hx", "hy", "sigma"],
};

/**
 * Hard-iron offset: the phone's own magnetised parts shift the measured field by a
 * constant vector, so readings that should lie on a sphere about the origin lie on one
 * about `b` instead. Sweeping through orientations traces that sphere out.
 *
 * The radius is a parameter here, unlike the accelerometer's known `g` — the ambient
 * field strength depends on where you are standing.
 */
export const HARD_IRON: SensorModel = {
  key: "hard_iron",
  title: "Magnetometer hard-iron",
  what: "Sweep the phone through as many orientations as you can — figure-eights work.",
  stan: `data {
  int<lower=0> N;
  vector[N] mx;
  vector[N] my;
  vector[N] mz;
}
parameters {
  real bx;
  real by;
  real bz;
  real<lower=0> radius;
  real<lower=0> sigma;
}
model {
  bx ~ normal(0, 50);
  by ~ normal(0, 50);
  bz ~ normal(0, 50);
  radius ~ normal(45, 25);
  sigma ~ exponential(1);
  for (n in 1:N)
    radius ~ normal(sqrt(square(mx[n] - bx) + square(my[n] - by) + square(mz[n] - bz)), sigma);
}`,
  init: [0.1, 0.1, 0.1, 3.8, 0],
  paramsOfInterest: ["bx", "by", "bz", "radius"],
};

/**
 * Weinberg's stride constant `k`, from walks over a known distance. Each walk gives
 * `distance = k · Σ amplitude^{1/4}`, so `k` follows from the total.
 *
 * Hierarchical because `k` drifts with pace: a hurried walk and a stroll do not share
 * one constant, but they are not independent either. Partial pooling lets a two-walk
 * calibration borrow from the others instead of trusting each on its own — the same
 * structure as the campaign-lift demo, and the reason it matters here is that the
 * filter needs `k` inside 5% before its map is worth running.
 */
export const STRIDE_K: SensorModel = {
  key: "stride_k",
  title: "Stride constant",
  what: "Walk a measured distance a few times, at different paces.",
  stan: `data {
  int<lower=0> W;
  vector<lower=0>[W] amp_sum;
  vector<lower=0>[W] distance;
}
parameters {
  real<lower=0> k_pop;
  real<lower=0> tau;
  vector[W] z;
  real<lower=0> sigma;
}
transformed parameters {
  vector[W] k;
  for (w in 1:W) k[w] = k_pop + tau * z[w];
}
model {
  k_pop ~ lognormal(log(0.45), 0.4);
  tau ~ exponential(20);
  sigma ~ exponential(5);
  for (w in 1:W) z[w] ~ normal(0, 1);
  for (w in 1:W) distance[w] ~ normal(k[w] * amp_sum[w], sigma);
}`,
  // Placeholder: the real length depends on W, so callers use `strideInit`.
  init: [],
  paramsOfInterest: ["k_pop", "tau", "sigma"],
};

/** `k_pop`, `tau`, one offset per walk, then `sigma` — sized to the data. */
export function strideInit(walks: number): number[] {
  return [-0.8, -3.0, ...Array(walks).fill(0.1), -1.6];
}

export const MODELS = [GYRO_BIAS, ACCEL_BIAS, HEADING, HARD_IRON, STRIDE_K];
