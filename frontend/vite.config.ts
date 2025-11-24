import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  root: resolve(__dirname),
  server: {
    port: 3000,
    open: true
  },
  // Verhindert, dass Vite nach oben im Verzeichnisbaum sucht
  optimizeDeps: {
    entries: ['src/**/*.{ts,tsx}']
  }
})

