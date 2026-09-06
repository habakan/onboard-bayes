import init from "stanwasm";
import wasmUrl from "stanwasm/pkg/stanwasm_bg.wasm?url";
import { ACCEL_BIAS, GYRO_BIAS, HARD_IRON, HEADING, STRIDE_K, strideInit } from "./models";
import { run, DEFAULTS as DEFAULT_RUN, type RunResult } from "./bench";
import { CASES, LAPTOP_MS, SIZES } from "./synthetic";
import {
  captureHeading,
  captureMotion,
  capturePoses,
  deviceLabel,
  requestMotionPermission,
  requestOrientationPermission,
  type MotionSample,
} from "./sensors";
import { captureField, directionsCovered, fieldSource, isNative, recordFieldInto } from "./magnetometer";
import { CapacitorHttp } from "@capacitor/core";

/** Baked in by vite from RECORD_URL; empty in a browser, where the page's own origin serves it. */
declare const __RECORD_URL__: string;

// The build bakes in a laptop address and DHCP moves it. Editing beats rebuilding: a
// recording that cannot be posted is a walk you have to do again.
const REC_URL_KEY = "onboard-bayes.recordUrl";

function recordUrl(): string {
  try {
    return localStorage.getItem(REC_URL_KEY) || __RECORD_URL__;
  } catch {
    return __RECORD_URL__;
  }
}

function setRecordUrl(v: string): void {
  try {
    localStorage.setItem(REC_URL_KEY, v.trim().replace(/\/$/, ""));
  } catch {
    // Storage disabled still records; it just forgets the address next launch.
  }
}

/**
 * A bundled app has no dev server behind its origin, and `capacitor://localhost` posting
 * to plain http is mixed content the WebView blocks. CapacitorHttp goes out natively.
 */
async function postRecording(payload: unknown): Promise<string> {
  const body = JSON.stringify(payload);
  if (isNative()) {
    const base = recordUrl();
    if (!base) throw new Error("no recorder address set — fill in the \"save to\" field");
    const res = await CapacitorHttp.post({
      url: `${base}/api/record`,
      headers: { "content-type": "application/json" },
      data: body,
    });
    if (res.status >= 300) throw new Error(`${base} answered ${res.status}`);
    return (res.data as { saved?: string })?.saved ?? "?";
  }
  const res = await fetch("/api/record", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
  });
  return ((await res.json()) as { saved?: string }).saved ?? "?";
}
import { Tracker, drawPath, TRACK_DEFAULTS } from "./tracking";
import { detectSteps, STEP_DEFAULTS, type MotionSample as PdrSample } from "./rbpf/pdr";
import "./style.css";

const G = 9.80665;

// Above this, two parameters have collapsed onto a ridge: only their combination was
// identified, so each one on its own is a confident wrong answer.
const DEGENERATE_AT = 0.95;
const app = document.querySelector<HTMLDivElement>("#app")!;

let ready = false;
const results: RunResult[] = [];

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  text = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text) node.textContent = text;
  return node;
}

function log(msg: string, kind: "info" | "error" = "info") {
  const line = el("div", { class: `log ${kind}` }, msg);
  document.querySelector("#log")!.prepend(line);
}

function reportFit(r: RunResult, label: string) {
  results.unshift(r);
  renderResults();
  const wp = r.worstPair;
  if (wp && Math.abs(wp.corr) >= DEGENERATE_AT) {
    log(
      `${label}: ${wp.a} and ${wp.b} are ${wp.corr.toFixed(3)} correlated — only their ` +
        `combination is identified, so neither value should be trusted`,
      "error",
    );
  } else {
    log(`${label} done`);
  }
}

function renderResults() {
  const table = document.querySelector("#results")!;
  table.innerHTML = "";
  if (results.length === 0) {
    table.append(el("p", { class: "muted" }, "No runs yet."));
    return;
  }
  const head = el("div", { class: "row head" });
  for (const h of ["model", "N", "compile", "sample", "worst corr", "estimate"]) {
    head.append(el("span", {}, h));
  }
  table.append(head);
  for (const r of results) {
    const row = el("div", { class: "row" });
    row.append(el("span", {}, r.key));
    row.append(el("span", { class: "num" }, String(r.n)));
    row.append(el("span", { class: "num" }, `${r.compileMs} ms`));
    row.append(el("span", { class: "num strong" }, `${r.sampleMs} ms`));
    const wp = r.worstPair;
    const degenerate = wp !== null && Math.abs(wp.corr) >= DEGENERATE_AT;
    row.append(
      el(
        "span",
        { class: degenerate ? "num warn" : "num" },
        wp ? `${wp.corr.toFixed(2)}` : "—",
      ),
    );
    const summary = Object.entries(r.means)
      .slice(0, 3)
      .map(([k, v]) => `${k}=${v.toFixed(3)}`)
      .join(" ");
    row.append(el("span", { class: degenerate ? "mono warn" : "mono" }, summary));
    table.append(row);
  }
}

async function withBusy<T>(button: HTMLButtonElement, fn: () => Promise<T>) {
  const label = button.textContent ?? "";
  button.disabled = true;
  button.textContent = "working…";
  try {
    return await fn();
  } finally {
    button.disabled = false;
    button.textContent = label;
  }
}

function gyroData(samples: MotionSample[]) {
  return { N: samples.length, w: samples.map((s) => s.wx) };
}

function accelData(samples: MotionSample[]) {
  return {
    N: samples.length,
    ax: samples.map((s) => s.ax),
    ay: samples.map((s) => s.ay),
    az: samples.map((s) => s.az),
    g: G,
  };
}

function fieldData(rs: { x: number; y: number; z: number }[]) {
  return {
    N: rs.length,
    mx: rs.map((r) => r.x),
    my: rs.map((r) => r.y),
    mz: rs.map((r) => r.z),
  };
}

function headingData(rad: number[]) {
  return { N: rad.length, cx: rad.map(Math.cos), cy: rad.map(Math.sin) };
}

/** One line per capability: what it is, whether this device has it, and why not. */
function capLine(ok: boolean, label: string, detail: string): HTMLElement {
  const li = el("li", { class: ok ? "cap ok" : "cap no" });
  li.append(el("span", { class: "dot" }, ok ? "●" : "○"));
  li.append(el("span", { class: "cap-label" }, label));
  li.append(el("span", { class: "cap-detail" }, detail));
  return li;
}

function build() {
  app.innerHTML = "";
  app.append(el("h1", {}, "onboard-bayes"));
  app.append(
    el(
      "p",
      { class: "muted" },
      "Bayesian sensor calibration, sampled on this device. Nothing is uploaded.",
    ),
  );

  const status = el("p", { class: "muted", id: "status" }, "loading wasm…");
  app.append(status);

  // Most visitors arrive on a desktop, where four of the five models have no sensor to
  // read. Saying so here beats five buttons that look broken.
  const caps = el("ul", { class: "caps", id: "caps" });
  app.append(caps);

  const controls = el("div", { class: "controls" });

  const nInput = el("input", { type: "number", value: "100", min: "10", max: "500" });
  const nLabel = el("label", {}, "samples ");
  nLabel.append(nInput);
  controls.append(nLabel);

  const benchBtn = el("button", { class: "primary" }, "Device benchmark");
  const gyroBtn = el("button", {}, "Gyro bias");
  const accelBtn = el("button", {}, "Accel bias");
  const headingBtn = el("button", {}, "Heading");
  const magBtn = el("button", {}, "Hard-iron");
  controls.append(benchBtn, gyroBtn, accelBtn, headingBtn, magBtn);
  // Shown only once fieldSource() says there is something behind it.
  magBtn.hidden = true;
  // Appended below the tracking section: the filter is what this demonstrates.

  // ---- tracking -------------------------------------------------------
  app.append(el("h2", {}, "Tracking"));
  app.append(
    el("p", { class: "muted" },
      "Steps and heading drive a particle filter. Without the raw field it is dead " +
      "reckoning, and heading comes from the gyroscope, so carry the phone flat — " +
      "swinging it at your side doubles the error."),
  );
  const canvas = el("canvas", { width: "340", height: "340", class: "map" });
  app.append(canvas);
  const trackStat = el("p", { class: "muted mono", id: "trackstat" }, "not started");
  app.append(trackStat);

  // A live counter that flashes on every detection. Reconciling a total after the
  // walk cannot tell a missed step from a doubled one; watching it can.
  const counter = el("div", { class: "counter", id: "stepcount" }, "0");
  app.append(counter);
  const truthRow = el("div", { class: "controls" });
  const trueInput = el("input", { type: "number", id: "truesteps", placeholder: "steps you counted", min: "1" });
  const checkBtn = el("button", { class: "secondary" }, "Compare");
  truthRow.append(trueInput, checkBtn);
  app.append(truthRow);

  checkBtn.addEventListener("click", () => {
    const truth = Number(trueInput.value);
    const got = tracker?.snapshot.steps ?? 0;
    if (!truth || got === 0) return log("walk first, then type what you counted", "error");
    const pct = (100 * (got - truth)) / truth;
    const verdict = Math.abs(pct) <= 5 ? "within 5%" : "off — the threshold needs work";
    log(`detected ${got} vs ${truth} counted: ${pct >= 0 ? "+" : ""}${pct.toFixed(1)}% — ${verdict}`,
        Math.abs(pct) <= 5 ? "info" : "error");
  });

  const trackRow = el("div", { class: "controls" });
  const kInput = el("input", { type: "number", id: "stridek", value: "0.45", step: "0.01", min: "0.1" });
  const kLabel = el("label", {}, "stride k ");
  kLabel.append(kInput);
  const startBtn = el("button", {}, "Start");
  const stopBtn = el("button", { class: "secondary" }, "Stop");
  stopBtn.disabled = true;
  trackRow.append(kLabel, startBtn, stopBtn);
  app.append(trackRow);

  let tracker: Tracker | null = null;
  startBtn.addEventListener("click", async () => {
    if (!ready) return log("wasm not ready", "error");
    tracker = new Tracker(
      { ...TRACK_DEFAULTS, strideK: Number(kInput.value) || 0.45 },
      (st) => {
        drawPath(canvas, st.path, st.pose);
        trackStat.textContent =
          `x=${st.pose.x[0].toFixed(2)} y=${st.pose.x[1].toFixed(2)} ` +
          `yaw=${((st.pose.yaw * 180) / Math.PI).toFixed(0)}°  ESS ${st.ess.toFixed(0)}` +
          (isNative() ? `  field ${st.updates}` : "");
      },
      (n) => {
        counter.textContent = String(n);
        counter.classList.add("tick");
        window.setTimeout(() => counter.classList.remove("tick"), 120);
      },
    );
    try {
      await tracker.start();
      startBtn.disabled = true;
      stopBtn.disabled = false;
      const fs = await fieldSource();
      log(fs.source === "none"
        ? `tracking — no raw magnetometer (${fs.detail}), so this is dead reckoning only`
        : "tracking — walk around; the map corrects the dead reckoning");
    } catch (e) {
      log(String(e), "error");
    }
  });
  stopBtn.addEventListener("click", () => {
    tracker?.stop();
    startBtn.disabled = false;
    stopBtn.disabled = true;
    log(`stopped at ${tracker?.snapshot.steps ?? 0} steps — type what you counted and compare`);
  });

  // ---- calibration ----------------------------------------------------
  app.append(el("h2", {}, "Calibration"));
  app.append(
    el("p", { class: "muted" },
      "Each button captures from a sensor and fits a Stan model to it here, on this " +
      "device. The filter above needs the stride constant and the noise these report."),
  );
  app.append(controls);

  // ---- recording ------------------------------------------------------
  // The recorder is vite middleware, so a static build has nothing behind it. The
  // bundled app posts to a dev server instead, and says so in the field below.
  if (import.meta.env.DEV || isNative()) {
    app.append(el("h2", {}, "Record for replay"));
    app.append(
      el("p", { class: "muted" },
        "Walk once with the raw stream saved, then tune thresholds against the file " +
        "instead of walking again for each change."),
    );
    const recRow = el("div", { class: "controls" });
    const recSteps = el("input", { type: "number", id: "recsteps", placeholder: "steps you will walk", min: "1" });
    const recBtn = el("button", {}, "Record 30s");
    recRow.append(recSteps, recBtn);
    app.append(recRow);

    // Only a bundled app posts elsewhere; in a browser the page's own origin serves it.
    if (isNative()) {
      const urlRow = el("div", { class: "controls" });
      const urlInput = el("input", { type: "url", id: "recurl", placeholder: "http://<laptop>:5173" });
      urlInput.value = recordUrl();
      urlInput.addEventListener("change", () => {
        setRecordUrl(urlInput.value);
        urlInput.value = recordUrl();
        log(`recordings go to ${recordUrl() || "nowhere"}`);
      });
      urlRow.append(el("span", { class: "muted" }, "save to"), urlInput);
      app.append(urlRow);
    }

    recBtn.addEventListener("click", () =>
      withBusy(recBtn, async () => {
        if (!(await requestMotionPermission())) return log("motion permission denied", "error");
        const truth = Number(recSteps.value) || 0;
        const motion: number[][] = [];
        const heading: number[][] = [];
        const field: number[][] = [];
        const on = (e: DeviceMotionEvent) => {
          const a = e.accelerationIncludingGravity;
          const r = e.rotationRate;
          if (!a || a.x === null || a.y === null || a.z === null) return;
          // Flat arrays: a 30 s recording is ~1500 samples and objects triple the size.
          motion.push([performance.now(), a.x, a.y, a.z, r?.alpha ?? 0, r?.beta ?? 0, r?.gamma ?? 0]);
        };
        // The compass is derived from the magnetometer, so a heading that wanders while
        // the phone is still is a magnetic disturbance — measurable without the native
        // build, which is the only way to get the raw axes on iOS.
        const onOrient = (e: DeviceOrientationEvent) => {
          const wk = (e as DeviceOrientationEvent & { webkitCompassHeading?: number })
            .webkitCompassHeading;
          const deg = wk ?? (e.alpha === null ? null : 360 - e.alpha);
          if (deg !== null && !Number.isNaN(deg)) heading.push([performance.now(), deg]);
        };
        await requestOrientationPermission();
        log("recording 30 s — walk now, counting your steps");
        window.addEventListener("devicemotion", on);
        window.addEventListener("deviceorientation", onOrient);
        const stopField = await recordFieldInto(field);
        await new Promise((r) => setTimeout(r, 30000));
        window.removeEventListener("devicemotion", on);
        window.removeEventListener("deviceorientation", onOrient);
        await stopField();

        const payload = {
          recorded: new Date().toISOString(),
          device: deviceLabel(),
          trueSteps: truth,
          columns: ["t", "ax", "ay", "az", "wx", "wy", "wz"],
          motion,
          headingColumns: ["t", "deg"],
          heading,
          fieldColumns: ["t", "x", "y", "z"],
          field,
        };
        try {
          const saved = await postRecording(payload);
          log(`saved ${motion.length} motion, ${heading.length} heading, ${field.length} field → ${saved}`);
        } catch (e) {
          log(`could not save: ${String(e)}`, "error");
        }
      }),
    );
  }

  // ---- stride calibration ---------------------------------------------
  app.append(el("h2", {}, "Stride calibration"));
  app.append(
    el("p", { class: "muted" },
      "Walk a measured distance a few times, at different paces. Five walks or more; " +
      "the filter needs k inside 5%."),
  );
  const calRow = el("div", { class: "controls" });
  const distInput = el("input", { type: "number", value: "20", step: "1", min: "3" });
  const distLabel = el("label", {}, "metres ");
  distLabel.append(distInput);
  const walkBtn = el("button", {}, "Record walk");
  const fitBtn = el("button", { class: "secondary" }, "Fit k");
  fitBtn.disabled = true;
  calRow.append(distLabel, walkBtn, fitBtn);
  app.append(calRow);
  const walks: { amp: number; dist: number }[] = [];
  const walkList = el("p", { class: "muted mono", id: "walks" }, "no walks yet");
  app.append(walkList);

  const renderWalks = () => {
    walkList.textContent = walks.length === 0
      ? "no walks yet"
      : walks.map((w, i) => `#${i + 1} ${w.dist}m / Σamp ${w.amp.toFixed(1)}`).join("   ");
    fitBtn.disabled = walks.length < 2;
  };

  walkBtn.addEventListener("click", () =>
    withBusy(walkBtn, async () => {
      if (!(await requestMotionPermission())) return log("motion permission denied", "error");
      const metres = Number(distInput.value) || 20;
      log(`recording — walk ${metres} m now, then this stops on its own`);
      const buf: PdrSample[] = [];
      const on = (e: DeviceMotionEvent) => {
        const a = e.accelerationIncludingGravity;
        if (!a || a.x === null || a.y === null || a.z === null) return;
        const r = e.rotationRate;
        const k = -Math.PI / 180;
        buf.push({ ax: a.x, ay: a.y, az: a.z,
          wx: (r?.alpha ?? 0) * k, wy: (r?.beta ?? 0) * k, wz: (r?.gamma ?? 0) * k,
          t: performance.now() });
      };
      window.addEventListener("devicemotion", on);
      await new Promise((r) => setTimeout(r, 20000));
      window.removeEventListener("devicemotion", on);

      const steps = detectSteps(buf, STEP_DEFAULTS);
      if (steps.length < 8) return log(`only ${steps.length} steps detected — walk further`, "error");
      const amp = steps.reduce((a, s) => a + Math.pow(s.amplitude, 0.25), 0);
      walks.push({ amp, dist: metres });
      renderWalks();
      log(`walk #${walks.length}: ${steps.length} steps over ${metres} m`);
    }),
  );

  fitBtn.addEventListener("click", () =>
    withBusy(fitBtn, async () => {
      if (!ready) return log("wasm not ready", "error");
      try {
        const data = {
          W: walks.length,
          amp_sum: walks.map((w) => w.amp),
          distance: walks.map((w) => w.dist),
        };
        // The unconstrained length grows with the number of walks, so the model's
        // own `init` cannot be a fixed array.
        const r = run({ ...STRIDE_K, init: strideInit(walks.length) }, data, {
          ...DEFAULT_RUN,
          seed: 29n,
        });
        reportFit(r, `stride k (${walks.length} walks)`);
        const k = r.means["k_pop"];
        if (k !== undefined) {
          kInput.value = k.toFixed(3);
          log(`k = ${k.toFixed(3)} — copied into tracking`);
        }
      } catch (e) {
        log(String(e), "error");
      }
    }),
  );

  app.append(el("h2", {}, "Runs"));
  app.append(el("div", { id: "results" }));
  app.append(el("h2", {}, "Log"));
  app.append(el("div", { id: "log" }));
  app.append(el("p", { class: "device mono" }, deviceLabel()));

  const n = () => Math.max(10, Math.min(500, Number(nInput.value) || 100));

  // No sensor and no permission prompt: this is engine speed on this device, on the
  // same synthetic data and seed the README's laptop column was measured with.
  benchBtn.addEventListener("click", () =>
    withBusy(benchBtn, async () => {
      if (!ready) return log("wasm not ready", "error");
      log("benchmarking — the unidentified case takes tens of seconds, that is the point");
      for (const size of SIZES) {
        for (const c of CASES) {
          if (c.label.includes("6 faces") && size !== 100) continue;
          await new Promise((r) => setTimeout(r, 0));
          try {
            const r = run(c.model, c.data(size));
            const laptop = LAPTOP_MS[c.label]?.[size];
            results.unshift({ ...r, key: c.label, n: size });
            renderResults();
            log(
              `${c.label} N=${size}: ${r.sampleMs} ms` +
                (laptop ? ` (laptop ${laptop} ms, ${(r.sampleMs / laptop).toFixed(1)}x)` : ""),
            );
          } catch (e) {
            log(`${c.label} N=${size}: ${String(e)}`, "error");
          }
        }
      }
      log("benchmark done — tap and hold the table to copy");
    }),
  );

  gyroBtn.addEventListener("click", () =>
    withBusy(gyroBtn, async () => {
      if (!ready) return log("wasm not ready", "error");
      if (!(await requestMotionPermission())) return log("motion permission denied", "error");
      log(`capturing ${n()} motion samples — hold still`);
      try {
        const s = await captureMotion(n());
        reportFit(run(GYRO_BIAS, gyroData(s)), `gyro bias (${s.length} samples)`);
      } catch (e) {
        log(String(e), "error");
      }
    }),
  );

  accelBtn.addEventListener("click", () =>
    withBusy(accelBtn, async () => {
      if (!ready) return log("wasm not ready", "error");
      if (!(await requestMotionPermission())) return log("motion permission denied", "error");
      const faces = 4;
      log(`rest the phone on ${faces} different sides, holding each still for a second`);
      try {
        let shown = -1;
        const s = await capturePoses(Math.ceil(n() / faces), faces, 90000, (pr) => {
          if (pr.faces === shown) return;
          shown = pr.faces;
          log(`faces held still: ${pr.faces}/${pr.needed}`);
        });
        reportFit(run(ACCEL_BIAS, accelData(s)), `accel bias (${s.length} samples, ${faces} faces)`);
      } catch (e) {
        log(String(e), "error");
      }
    }),
  );

  headingBtn.addEventListener("click", () =>
    withBusy(headingBtn, async () => {
      if (!ready) return log("wasm not ready", "error");
      if (!(await requestOrientationPermission()))
        return log("orientation permission denied", "error");
      log(`capturing ${n()} heading samples — point one way and hold`);
      try {
        const h = await captureHeading(n());
        reportFit(run(HEADING, headingData(h)), `heading (${h.length} samples)`);
      } catch (e) {
        log(String(e), "error");
      }
    }),
  );

  // devicemotion needs a secure context, and a WebView pointed at a plain-http dev
  // server has none — permission then resolves to true and no event ever arrives.
  const hasMotion = "DeviceMotionEvent" in window && window.isSecureContext;
  caps.append(
    capLine(
      hasMotion,
      "motion sensors",
      hasMotion
        ? "accelerometer and gyroscope — gyro bias, accel bias and stride"
        : window.isSecureContext
          ? "no devicemotion here; open this on a phone"
          : `blocked: ${location.origin} is not a secure context`,
    ),
  );

  // Raw field comes from CoreMotion natively or the Generic Sensor API in a browser.
  // Reported on load: which one is available is the thing being tested on new devices.
  magBtn.disabled = true;
  void fieldSource().then((fs) => {
    const ok = fs.source !== "none";
    magBtn.hidden = !ok;
    magBtn.disabled = !ok;
    magBtn.title = fs.detail;
    caps.append(
      capLine(
        ok,
        "raw magnetometer",
        ok
          ? `${fs.detail} — hard-iron calibration and the field map`
          : "no phone browser exposes the raw axes; heading still works, hard-iron does not",
      ),
    );
    // The compass is magnetometer-derived even where the raw axes are not reachable.
    caps.append(capLine(true, "compass heading", "fused by the OS — the heading model reads this"));
  });

  magBtn.addEventListener("click", () =>
    withBusy(magBtn, async () => {
      if (!ready) return log("wasm not ready", "error");
      log(`sweeping for ${n()} field samples — figure-eights, all orientations`);
      try {
        const rs = await captureField(n());
        const dirs = directionsCovered(rs);
        log(`captured ${rs.length} samples over ${dirs}/6 directions`);
        if (dirs < 3) {
          return log("too few orientations — the offsets will not be identified", "error");
        }
        reportFit(run(HARD_IRON, fieldData(rs)), `hard-iron (${dirs}/6 directions)`);
      } catch (e) {
        log(String(e), "error");
      }
    }),
  );

  renderResults();
}

build();

init({ module_or_path: wasmUrl })
  .then(() => {
    ready = true;
    document.querySelector("#status")!.textContent = "wasm ready — sampling runs on this device";
  })
  .catch((e: unknown) => {
    // A rejected init used to leave the page on "loading" forever, which is exactly
    // how the relaxed-SIMD failure presented on iOS before stanwasm 0.1.1.
    document.querySelector("#status")!.textContent = `wasm failed to load: ${String(e)}`;
    log(String(e), "error");
  });
