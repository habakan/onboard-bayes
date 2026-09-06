# Findings

Every measurement behind [the README](README.md), kept in full because the numbers are
what make the claims checkable. Simulation unless a row says a phone produced it.

## What the sensor runs showed

**Speed is a property of the model, not of the device.** 500 warmup + 500 draws,
same synthetic data and seed on both. Phone is an iPhone SE (3rd gen, A15) in Safari;
laptop is an M-series Mac in Node. Both columns are stanwasm 0.1.2, which is what
makes the ratio meaningful; the phone has not been re-run on 0.3.0. `LAPTOP_MS` in
`src/synthetic.ts`, which the app compares a live phone run against, does carry the
0.3.0 laptop numbers — 25 to 30% below the column here on the identified models.

| Model | N | Phone | Laptop | Ratio |
|---|---:|---:|---:|---:|
| Gyro bias | 50 | 36 ms | 10 ms | 3.6x |
| Gyro bias | 100 | 15 ms | 13 ms | 1.2x |
| Gyro bias | 200 | 30 ms | 25 ms | 1.2x |
| Heading | 50 | 23 ms | 16 ms | 1.4x |
| Heading | 100 | 39 ms | 31 ms | 1.3x |
| Heading | 200 | 77 ms | 56 ms | 1.4x |
| Accel bias, six orientations | 100 | 88 ms | 64 ms | 1.4x |
| Accel bias, one orientation | 50 | 4.9 s | 3.9 s | 1.3x |
| Accel bias, one orientation | 100 | 10.7 s | 8.0 s | 1.3x |
| Accel bias, one orientation | 200 | 18.1 s | 15.6 s | 1.2x |

A phone is 1.2–1.4x a laptop here, not the several-fold penalty worth designing
around. An identified fit lands in 15–88 ms, so refitting on a sensor duty cycle is
comfortable rather than marginal.

The 3.6x on the very first row is warmup, not the model: the same model at N=100 —
twice the work — finishes in 15 ms right after. Run one throwaway fit at startup if
the first real one is user-visible.

**An unidentified model is slow and wrong at the same time.** Holding the phone still
gives one point on the gravity sphere, so only the distance from the origin is
observed and every offset at that distance fits equally well. NUTS walks that ridge:
8 s to return `bx=-5.5` where the answer is `0.05`. Turning the phone over a few times
during the capture makes the same model finish in 64 ms with the right answer.

That is 125x, from data collection alone, and on the phone the unidentified fit costs
18 seconds. Reach for the capture protocol before reaching for a faster sampler.

**And the fix for one assumption can break another.** Turning the phone identifies the
offsets, but `accelerationIncludingGravity` reports linear acceleration on top of
gravity, so a reading taken mid-turn has magnitude != g — and a model that assumes
`|a| = g` charges the hand movement to the bias. A first run on real hardware returned
an offset of 0.92 m/s², about 9% of g, which no modern phone actually has.

Filtering to stationary samples alone made it worse — 4.18 m/s² on the next real run.
At 60 Hz a hundred samples fill in under two seconds, so the quota was satisfied by the
first face and the capture ended before the phone was ever moved. Stillness had cost
the orientation diversity that identifies the model.

So the capture takes a quota *per face*, only while `rotationRate` is under 3 deg/s,
and will not finish until four faces are filled. Against synthetic data with 2 m/s² of
movement between poses it recovers 0.062 / -0.031 / 0.020 where the truth is
0.05 / -0.03 / 0.02, and a phone left on one face errors out at 90 seconds instead of
returning a confident wrong answer.

Both failures produced a number that looked like an answer. The count-based capture is
what turned the second one into an error message.

Every fit therefore also reports its strongest posterior correlation. Two parameters
near ±1 mean only their combination was identified: a hard-iron fit taken from one
orientation puts `bz` and `radius` at −1.000, because a small patch of a sphere is
nearly a plane and only `bz + radius` — the distance to that plane — is observable.
Its sum lands at 64.95 ± 0.06 against a true 65, while each term is off by 2.2.

| Coverage | Worst pair | Flagged at 0.95 | Error |
|---|---:|---|---:|
| All orientations | −0.686 | no | 0.0 µT |
| Hemisphere | −0.944 | no | 0.1 µT |
| Narrow cap | −1.000 | **yes** | 2.2 µT |

The threshold sits between a fit that is merely imperfect and one that is not a fit at
all. Note that the degenerate case is also 15x slower: on this model, slowness is a
symptom of the same thing.

On the phone, the three protocols read the same sensor and disagree by a factor of 56:

| Capture | Estimated offset | Fit |
|---|---:|---:|
| Rotating throughout | 0.92 m/s² (9.4% of g) | 150 ms |
| One face only | 4.18 m/s² (43% of g) | 1469 ms |
| Four faces, each still | **0.074 m/s² (0.75% of g)** | 114 ms |

Only the last is a plausible bias for a phone. Note that the wrong answers are not
noisier — the middle one is confident and 56x off.

**And the phone has to be carried flat.** Tracking takes heading from the gyroscope
alone, so the filter turns with the *device*, not with the walker. Five walks of the
same out-and-back, where the answer is a 180 degree turn:

| Carry | Pace | Attitude travel | Heading turned |
|---|---|---:|---:|
| Flat, screen up | normal | 93° | 204° |
| Flat, screen up | slow | 98° | 199° |
| Flat, screen up | fast | 200° | 306° |
| Arm at side | normal | 424° | 334° |
| Arm at side | slow | 546° | 350° |

Attitude travel is how far the gravity direction itself moved over the walk. Held flat
the turn lands within 13%; swung at the side it comes out double, and the error tracks
how much the phone rotated rather than which pace was walked. Tuning does not reach it
— sweeping the gravity filter's constant over an eighty-fold range moves the flat
walks by 6 degrees and leaves the swinging ones between 249 and 374. An arm swing
rotates the device without turning the walker, and integrating device rotation cannot
tell those apart.
## The filter

Calibration is a batch fit; tracking is not. NUTS re-runs over the whole dataset every
time and its parameter count grows with the trajectory, so pose estimation runs as a
Rao-Blackwellised particle filter in `src/rbpf/` instead, and Stan keeps the job it is
good at — the hyperparameters, every few seconds.

The field is curl-free in free space, so it is modelled as `B = -∇φ` with a
reduced-rank GP on the scalar potential: Laplace eigenfunctions on a box, weighted by
the squared-exponential spectral density. Truncating at `m` terms makes the field
linear in `m` weights, so each particle carries a pose and a Gaussian over those
weights that updates in closed form. Sampling the field directly would not be
tractable.

`npm run verify:rbpf` checks the three layers against known answers:

| Check | Result |
|---|---|
| Eigenfunction gradients vs finite differences | 2.8e-10 |
| `curl B` on a random field | 1e-10, i.e. curl-free by construction |
| Map recovers a known field, on the path | 2.0% relative error |
| Map off the path | 51.6% — a GP falls back to its prior, as it should |
| RBPF against dead reckoning | 2.19 m → **0.54 m** mean error |
| Step detection, 1.5–2.5 steps/s | exact, and nothing while stationary |
| PDR into the filter, stride constant 12% wrong | 1.92 m → 1.48 m |

Steps come from peaks in the smoothed acceleration magnitude rather than from double
integration: the bias the accelerometer model estimates is about 0.07 m/s², which
integrates to metres of error within seconds, while a step count trades that for a
per-step error that does not compound the same way.

The detection threshold is 0.3 m/s² above the gravity baseline. The first value tried,
1.2, found nothing at all — smoothing takes the top off a peak, so the swing that
survives is smaller than the raw signal suggests. Sweeping it against known step counts
is what settled it.

### Why the stride constant has to be calibrated

Weinberg's estimate carries a per-person constant `k`. Getting it wrong is a
*systematic* scale error, and that is the one thing the map cannot repair:

| `k` error | Dead reckoning | RBPF | Gain |
|---:|---:|---:|---:|
| 0% | 0.00 m | 0.14 m | — |
| 2% | 0.32 m | **0.14 m** | 2.3x |
| 5% | 0.80 m | **0.23 m** | 3.5x |
| 12% | 1.92 m | 1.48 m | 1.3x |
| 25% | 4.01 m | **4.52 m** | **0.89x, worse** |

Below 5% the map is worth 2–3x. Past 12% it stops helping, and at 25% it actively
hurts. The map is built from the trajectory it is correcting, so a stride that is
uniformly too long records the field it saw at the wrong place — map and path stay
consistent with each other while both drift, and nothing in the measurements
contradicts it.

An uncalibrated Weinberg constant is routinely 20–30% off, so this is not a refinement:
without calibration the filter is worse than not running it. That is the job the
hierarchical Stan model does, and 5% is the number it has to reach.

It reaches it. Walking a measured distance a few times, at different paces:

| Walks | `k` estimate (true 0.45) | Error |
|---:|---:|---:|
| 2 | 0.4336 | 3.7% |
| 3 | 0.4740 | 5.3% |
| 5 | 0.4498 | **0.0%** |
| 8 | 0.4558 | **1.3%** |

Five walks or more clears it comfortably; three does not, because `k` genuinely drifts
with pace and a short calibration still carries one walk's habits. Each fit takes under
25 ms, so this is a thing to redo whenever it seems off, not a setup ritual.

The model is `STRIDE_K` in `src/models.ts`. It needs stanwasm 0.1.2 or later: indexed
assignment and shaped `transformed parameters` both landed after 0.1.1.

At 200 particles and m=128 a step costs about 19 ms, so roughly 50 Hz on a laptop.
Memory is the binding constraint before speed: each particle's covariance is m×m, so
500 particles at m=256 would be 250 MB and an iOS WebView would be killed for it.

### Localization benchmark

```bash
npm run bench:loc          # four trajectories, field richness, field noise
npm run bench:loc -- sweep # particles and basis size against cost
```

Scored the way SLAM results usually are, because one mean error hides too much:

- **ATE** — RMSE against the true positions. Global accuracy.
- **RPE** — error in each 10-step displacement. Local consistency, which ATE hides: a
  path with the right shape but an offset scores badly on ATE and well on RPE.
- **Loop closure** — the gap between the first and last estimate where the truth
  returns to its start. The number a map is supposed to improve.

Trajectories are chosen to separate causes: `loop` returns, `figure8` covers the same
ground twice so the second lap can use the first lap's map, `corridor` never revisits
anywhere, and `scan` revisits without closing.

Every row averages eight seeds. A single seed is not enough to rank anything here: the
ATE moves more between runs than between the settings being compared, and an earlier
single-seed version of this table supported the flat conclusion that the filter always
loses to dead reckoning. It does not.

| Trajectory | ATE | Dead reckoning | Seeds won |
|---|---:|---:|---:|
| `loop` | 0.42 ± 0.16 | 0.37 | 4/8 |
| `figure8` | **0.20 ± 0.04** | 0.37 | 8/8 |
| `corridor` | 0.62 ± 0.28 | 0.44 | 2/8 |
| `scan` | 7.40 ± 0.59 | 7.36 | 1/8 |

The split is the one the theory predicts. `figure8` covers its ground twice and wins on
every seed by a wide margin; `corridor` never returns anywhere and cannot be helped;
`scan` drifts 7 m, far outside the 0.4 m a map predicts past its observations, so no
map can reach it.

Getting there needed one fix, and it was not in the filter. **The process noise has to
match the odometry's, and inflating it is not the conservative choice it looks like.**
The benchmark assumed 0.03 m and 0.01 rad per step against a true 0.01 and 0.004, so
every particle random-walked three times faster than the odometry did, and where the
map could not correct that walk it was pure added error. Matching the two moved
`figure8` from 0.37 m to 0.20 m and `corridor` from 2.52 m to 0.62 m.

`degradation-check.mjs` isolates it by switching each source off in turn on a uniform
field, where nothing the weights say can be information and the filter should therefore
degrade to dead reckoning exactly:

| Configuration | ATE | Drift from dead reckoning |
|---|---:|---:|
| As shipped, 0.03 / 0.01 | 0.83 | 0.79 |
| Never resample | 0.91 | 0.69 |
| No odometry noise at all | 0.44 | 0.00 |
| Matched, 0.01 / 0.004 | **0.45** | 0.17 |

With no injected noise the filter reproduces dead reckoning to 0.00 m, which is what
names the cause; with matched noise it degrades to it properly at 0.45 against 0.44.
Resampling makes the walk worse but does not cause it — turning resampling off leaves
the error and collapses the effective sample size to 5 of 150.

The app carried the same fault, larger. It walks in 0.7 m strides rather than the
benchmark's 8 cm steps, so the settings do not transfer directly, but at stride scale
its `yaw: 0.05` scored 2.43 m on a uniform field against dead reckoning's 1.07 — worse
than switching the filter off. At `yaw: 0.01` it degrades correctly (1.06) and with a
field it improves on dead reckoning (0.86). Its `pos: 0.05` was already right, being
7% of a stride. Fixed in `tracking.ts`.

Where the fault is not: `oracle-map-check.mjs` hands the particles a finished map and
lets them only weight against it. That tracks to **0.012 m** where dead reckoning is
0.541 m, so the weighting, resampling and estimation are sound and the rest belongs to
building the map online. Widening the particle spread makes things worse, which the
process-noise result above now explains.

`map-range-check.mjs` measures the other half of the picture: with `ell = 1.2`, a map
predicts to about 0.4 m past its observations and is useless by 1.6 m. Particles that
differ by less than that range predict the same field, so their weights rank noise
rather than position — and a step is 8 cm.

`identifiability-check.mjs` names the cause. A particle that builds its own map from a
displaced trajectory writes every observation to the wrong place and is then consistent
with itself. The likelihood still prefers the truth, but barely:

| Offset | Self-built map | Finished map |
|---:|---:|---:|
| 0.25 m | 0.5 | 26,456 |
| 0.50 m | 1.3 | 107,242 |
| 1.00 m | 2.3 | 402,380 |

Five orders of magnitude. A gap of 1.3 cannot separate 150 particles. The map is
flexible enough to explain any trajectory hypothesis, so on a first visit it says
nothing about position — only revisits can, and this implementation does not exploit
them.

Three hypotheses were tested and rejected on the way. Thinning the observations makes
it worse (0.79 m at every step, 2.03 m at every eighth), so it is not over-counting
correlated data. Sweeping measurement noise from 0.5 to 8 µT and particle spread from
0.03 to 0.30 m never beats dead reckoning, so it is not tuning. Matching the map's
basis size to the field's does help — 2.30 m to 0.79 m — but that is model
misspecification in the benchmark, not the defect itself.

The third was holding a block of readings back and scoring all of them against maps
that had seen none of them, on the theory that a particle absorbs each reading before
being judged on it. That theory was wrong: `MagMap.update` computes its residual from
the prior mean, so every reading was already scored against a map that had not seen it.
Blocking only delays the map by K readings, and the accuracy falls off monotonically on
exactly the trajectories that revisit — on `figure8`, 0.35 m at a block of 1 against
0.56, 0.69 and 0.77 at 4, 16 and 64. The change was reverted.

Two things were wrong with the benchmark itself before it could say even that: the
synthetic field was 1.7 µT peak, below the noise it was measured with, so every row was
reading noise; and the first version of the disturbance check trusted a heading-derived
µT figure that under-reports by 7x.
## Is a mounting magnetically quiet enough?

Before mapping from a vehicle — a robot vacuum, a trolley — measure what its motors do
to the field. Hold the phone still on it, record 30 s with the motors off for the first
half and running for the second, then:

```bash
npm run disturbance
```

It compares the halves and reports the shift, against a 45 µT ambient field: under 5 µT
is usable, 5–15 µT wants the phone raised further from the motors, above that the
mounting will not support magnetic mapping.

In a browser there are no raw axes, so it falls back to the compass heading — and that
is **a lower bound only**. Heading is an angle in the horizontal plane, so a
disturbance pointing along the existing field lengthens it without turning it: a
synthetic 25 µT added that way moves the heading 4.4°, which back-converts to 3.5 µT.
Seven times off, in the direction that would say "fine". Use the native build before
concluding a mounting is clean.