import { ACCEL_BIAS, GYRO_BIAS, HEADING, type SensorModel } from "./models";

const G = 9.80665;
const noise = (i: number) => 0.02 * Math.sin(i * 7.3);
const seq = (n: number, f: (i: number) => number) => Array.from({ length: n }, (_, i) => f(i));

/** Six faces, so the offsets are identified. The single-orientation version of this
 *  data takes 8 s and returns the wrong answer — see the README. */
const FACES = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [0, 0, 1],
  [0, 0, -1],
];

export interface SyntheticCase {
  model: SensorModel;
  label: string;
  data: (n: number) => Record<string, number | number[]>;
}

export const CASES: SyntheticCase[] = [
  {
    model: GYRO_BIAS,
    label: "gyro_bias",
    data: (n) => ({ N: n, w: seq(n, (i) => 0.012 + noise(i)) }),
  },
  {
    model: HEADING,
    label: "heading",
    data: (n) => ({
      N: n,
      cx: seq(n, (i) => Math.cos(0.7) + noise(i)),
      cy: seq(n, (i) => Math.sin(0.7) + noise(i + 1)),
    }),
  },
  {
    model: ACCEL_BIAS,
    label: "accel_bias (6 faces)",
    data: (n) => {
      const ax: number[] = [];
      const ay: number[] = [];
      const az: number[] = [];
      for (let i = 0; i < n; i++) {
        const d = FACES[i % 6]!;
        ax.push(G * d[0]! + 0.05 + noise(i));
        ay.push(G * d[1]! - 0.03 + noise(i + 1));
        az.push(G * d[2]! + 0.02 + noise(i + 2));
      }
      return { N: n, ax, ay, az, g: G };
    },
  },
  {
    model: ACCEL_BIAS,
    label: "accel_bias (1 face, unidentified)",
    data: (n) => ({
      N: n,
      ax: seq(n, (i) => 0.05 + noise(i)),
      ay: seq(n, (i) => -0.03 + noise(i + 1)),
      az: seq(n, (i) => G + 0.02 + noise(i + 2)),
      g: G,
    }),
  },
];

export const SIZES = [50, 100, 200];

/** The laptop this was written on, same seed and draw counts, for scaling the
 *  device numbers against something. Medians on stanwasm 0.3.0; the identified
 *  models run 25-30% faster than they did on 0.1.2. */
export const LAPTOP_MS: Record<string, Record<number, number>> = {
  gyro_bias: { 50: 6, 100: 9, 200: 17 },
  heading: { 50: 12, 100: 23, 200: 42 },
  "accel_bias (6 faces)": { 100: 63 },
  "accel_bias (1 face, unidentified)": { 50: 3886, 100: 8220, 200: 15368 },
};
