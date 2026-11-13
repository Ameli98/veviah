// Builds the desktop companion page (app/companion.html) into dist-companion/, served by companion/server.mjs.
import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  root: 'app',
  plugins: [viteSingleFile()],
  build: { outDir: '../dist-companion', emptyOutDir: true, rollupOptions: { input: resolve(import.meta.dirname, 'app/companion.html') } },
});
