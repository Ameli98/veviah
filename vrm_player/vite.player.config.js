// Builds the player page (index.html + src/main.js) into dist-player/ (2026-11-13). The Windows app's hidden cloth
// window loads it from the companion (/player/) and calls window.hsin.clothSurface, as render_server.mjs does
// through Playwright in WSL.
import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

export default defineConfig({
  plugins: [viteSingleFile()],
  publicDir: false,          // never copy public/ (all the models, incl. private test ones) into the build
  build: { outDir: 'dist-player', emptyOutDir: true, rollupOptions: { input: resolve(import.meta.dirname, 'index.html') } },
});
