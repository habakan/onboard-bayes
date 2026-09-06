import { Capacitor, registerPlugin } from "@capacitor/core";

export interface MagReading {
  x: number;
  y: number;
  z: number;
  t: number;
}

interface MagnetometerPlugin {
  isAvailable(): Promise<{ available: boolean }>;
  start(options?: { hz?: number }): Promise<void>;
  stop(): Promise<void>;
  addListener(
    event: "reading",
    cb: (r: MagReading) => void,
  ): Promise<{ remove: () => Promise<void> }>;
}

/// The name must match `jsName` in MagnetometerPlugin.swift.
const Native = registerPlugin<MagnetometerPlugin>("Magnetometer");

/** The Generic Sensor API shape. Not in the DOM lib, since no browser ships it widely. */
interface XyzSensor extends EventTarget {
  start(): void;
  stop(): void;
  readonly x: number | null;
  readonly y: number | null;
  readonly z: number | null;
  readonly timestamp: number | null;
}
type XyzSensorCtor = new (opts?: { frequency?: number }) => XyzSensor;

const SensorCtor = (globalThis as { Magnetometer?: XyzSensorCtor }).Magnetometer;

export type FieldSource = "native" | "sensor" | "none";

export interface FieldSourceReport {
  source: FieldSource;
  /** Why, in a form worth putting in front of a user on an unfamiliar device. */
  detail: string;
}

export function isNative(): boolean {
  return Capacitor.isNativePlatform();
}

/**
 * Starts the Generic Sensor magnetometer and waits for a reading or an error.
 *
 * Chrome for Android has never shipped `Magnetometer` on any version, and desktop
 * Chrome keeps it behind `#enable-generic-sensor-extra-classes`, so this reports rather
 * than succeeds on a phone. Presence of the constructor decides nothing either: a denied
 * prompt or permissions policy fails only once the sensor has actually been started.
 */
function probeSensor(timeoutMs = 2000): Promise<FieldSourceReport> {
  return new Promise((resolve) => {
    if (!SensorCtor) {
      return resolve({
        source: "none",
        detail:
          "no Magnetometer constructor — absent on Chrome for Android, and behind " +
          "#enable-generic-sensor-extra-classes on desktop Chrome",
      });
    }
    if (!window.isSecureContext) {
      return resolve({ source: "none", detail: "sensors need a secure context (https)" });
    }

    let sensor: XyzSensor;
    try {
      sensor = new SensorCtor({ frequency: 10 });
    } catch (e) {
      return resolve({ source: "none", detail: `constructor threw: ${String(e)}` });
    }

    const finish = (r: FieldSourceReport) => {
      window.clearTimeout(timer);
      try {
        sensor.stop();
      } catch {
        // Already stopped or never started; nothing to recover.
      }
      resolve(r);
    };
    const timer = window.setTimeout(
      () => finish({ source: "none", detail: `no reading in ${timeoutMs} ms` }),
      timeoutMs,
    );

    sensor.addEventListener("reading", () =>
      finish({ source: "sensor", detail: "Generic Sensor API magnetometer" }),
    );
    sensor.addEventListener("error", (e) => {
      const err = (e as Event & { error?: DOMException }).error;
      const name = err?.name ?? "unknown";
      const why =
        name === "NotAllowedError"
          ? "permission denied"
          : name === "SecurityError"
            ? "blocked by permissions policy"
            : name === "NotReadableError"
              ? "no magnetometer on this device"
              : (err?.message ?? name);
      finish({ source: "none", detail: `${name}: ${why}` });
    });

    try {
      sensor.start();
    } catch (e) {
      finish({ source: "none", detail: `start threw: ${String(e)}` });
    }
  });
}

let cached: Promise<FieldSourceReport> | undefined;

/** Which backend can deliver raw field, and why not when it cannot. Probed once. */
export function fieldSource(): Promise<FieldSourceReport> {
  cached ??= (async () => {
    if (isNative()) {
      try {
        if ((await Native.isAvailable()).available) {
          return { source: "native" as const, detail: "CoreMotion magnetometer" };
        }
        return { source: "none" as const, detail: "no magnetometer on this device" };
      } catch (e) {
        // "not implemented" means the bridge never registered the plugin, which looks
        // the same as a broken one until you can see what it did register.
        const headers = (globalThis as { Capacitor?: { PluginHeaders?: { name: string }[] } })
          .Capacitor?.PluginHeaders;
        const listed = headers ? headers.map((h) => h.name).join(", ") || "none" : "no PluginHeaders";
        return { source: "none" as const, detail: `native plugin failed: ${String(e)} — bridge registered: ${listed}` };
      }
    }
    return probeSensor();
  })();
  return cached;
}

export async function magnetometerAvailable(): Promise<boolean> {
  return (await fieldSource()).source !== "none";
}

/** Streams raw readings to `cb` until the returned function is called. */
async function subscribe(cb: (r: MagReading) => void): Promise<() => Promise<void>> {
  const { source } = await fieldSource();

  if (source === "native") {
    const handle = await Native.addListener("reading", cb);
    await Native.start({ hz: 50 });
    return async () => {
      await Native.stop();
      await handle.remove();
    };
  }

  if (source === "sensor" && SensorCtor) {
    const sensor = new SensorCtor({ frequency: 50 });
    sensor.addEventListener("reading", () => {
      if (sensor.x === null || sensor.y === null || sensor.z === null) return;
      cb({ x: sensor.x, y: sensor.y, z: sensor.z, t: sensor.timestamp ?? performance.now() });
    });
    sensor.start();
    return async () => sensor.stop();
  }

  return async () => {};
}

/**
 * Records `n` raw field readings. Unlike the accelerometer, stillness is not required
 * — the field's magnitude is the same in every orientation, and rotating is what
 * identifies the offsets, so the caller is asked to sweep the phone around instead.
 */
export async function captureField(n: number, timeoutMs = 60000): Promise<MagReading[]> {
  const out: MagReading[] = [];
  const stop = await subscribe((r) => {
    if (out.length < n) out.push(r);
  });

  const started = Date.now();
  try {
    while (out.length < n) {
      if (Date.now() - started > timeoutMs) {
        throw new Error(`only ${out.length}/${n} field samples in ${timeoutMs / 1000}s`);
      }
      await new Promise((r) => setTimeout(r, 50));
    }
  } finally {
    await stop();
  }
  return out;
}

/** How many of the six coarse directions the sweep actually covered. Fewer than three
 *  and the sphere is not pinned down, which shows up as a slow, wrong fit. */
export function directionsCovered(rs: MagReading[]): number {
  const seen = new Set<number>();
  for (const r of rs) {
    const v = [r.x, r.y, r.z];
    let k = 0;
    for (let i = 1; i < 3; i++) if (Math.abs(v[i]!) > Math.abs(v[k]!)) k = i;
    seen.add(v[k]! >= 0 ? k : k + 3);
  }
  return seen.size;
}

/**
 * Streams raw field readings into `sink` until the returned function is called. A
 * no-op where no backend can deliver them, so the caller does not have to branch.
 */
export async function recordFieldInto(sink: number[][]): Promise<() => Promise<void>> {
  return subscribe((r) => sink.push([performance.now(), r.x, r.y, r.z]));
}
