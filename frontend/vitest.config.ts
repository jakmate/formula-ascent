import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react-swc';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        'src/vite-env.d.ts',
        'src/main.tsx',
        'src/types/**',
        'src/data/**',
        '**/*.d.ts',
        'dist/**',
        '*.config.js',
        '*.config.ts',
      ],
      thresholds: { lines: 0, functions: 0, branches: 0, statements: 0 },
    },
  },
});
