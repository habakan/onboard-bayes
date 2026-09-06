import { writeFileSync, mkdirSync } from "node:fs";
import { defineConfig, type Plugin } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";

/**
 * Saves a recorded sensor log to `recordings/`. Walking once and replaying offline is
 * the only way to iterate on detection thresholds at any speed — the alternative is a
 * walk per parameter change.
 */
function recorder(): Plugin {
  return {
    name: "recorder",
    configureServer(server) {
      server.middlewares.use("/api/record", (req, res) => {
        if (req.method !== "POST") {
          res.statusCode = 405;
          return res.end();
        }
        const chunks: Buffer[] = [];
        req.on("data", (c) => chunks.push(c as Buffer));
        req.on("end", () => {
          try {
            const body = Buffer.concat(chunks).toString("utf8");
            const name = `recordings/${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
            mkdirSync("recordings", { recursive: true });
            writeFileSync(name, body);
            res.setHeader("content-type", "application/json");
            res.end(JSON.stringify({ saved: name, bytes: body.length }));
            server.config.logger.info(`recorded ${name} (${body.length} bytes)`);
          } catch (e) {
            res.statusCode = 500;
            res.end(String(e));
          }
        });
      });
    },
  };
}

// DeviceMotionEvent.requestPermission() needs a secure context. `tailscale serve`
// supplies a real certificate, so set TS_SERVE=1 and let it terminate TLS instead.
export default defineConfig({
  plugins: process.env.TS_SERVE ? [recorder()] : [basicSsl(), recorder()],
  server: { host: true, allowedHosts: true },
  // A GitHub Pages project site serves under /<repo>/; the app and the dev server
  // both serve from a root, so this is only set for that one build.
  base: process.env.PAGES ? "/onboard-bayes/" : "/",
  // A bundled app serves from capacitor://localhost and has no dev server to POST to.
  define: { __RECORD_URL__: JSON.stringify(process.env.RECORD_URL ?? "") },
});
