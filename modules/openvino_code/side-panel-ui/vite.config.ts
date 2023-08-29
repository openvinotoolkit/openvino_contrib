import { defineConfig } from 'vite';
import { fileURLToPath } from 'url';
import react from '@vitejs/plugin-react';
import copy from 'rollup-plugin-copy';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@shared': fileURLToPath(new URL('../shared', import.meta.url)),
    },
  },
  build: {
    outDir: 'dist',
    rollupOptions: {
      output: {
        entryFileNames: `assets/[name].js`,
        chunkFileNames: `assets/[name].js`,
        assetFileNames: `assets/[name].[ext]`,
      },
      plugins: [
        copy({
          targets: [
            {
              src: [
                '../node_modules/@vscode/codicons/dist/codicon.css',
                '../node_modules/@vscode/codicons/dist/codicon.ttf',
              ],
              dest: 'dist/assets',
            },
          ],
          hook: 'generateBundle',
        }),
      ],
    },
  },
});
