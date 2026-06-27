import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  cacheDir: "../tmp/vite-cache",
  server: {
    host: "127.0.0.1",
  },
});
