import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/jobs": {
        target: "http://localhost:8000",
        ws: true,
      },
    },
  },
});
