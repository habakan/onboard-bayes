import type { CapacitorConfig } from '@capacitor/cli';

// CAP_SERVER_URL points the app at `npm run dev` instead of the copied `dist`, which is
// what keeps `/api/record` reachable: the recorder is vite middleware, not part of the app.
const dev = process.env.CAP_SERVER_URL;

const config: CapacitorConfig = {
  appId: 'com.habakan.onboardbayes',
  appName: 'onboard-bayes',
  webDir: 'dist',
  ...(dev ? { server: { url: dev, cleartext: true } } : {}),
};

export default config;
