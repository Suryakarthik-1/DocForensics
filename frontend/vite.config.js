import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // In dev, forward API calls to the FastAPI backend so the app can use
    // same-origin '/api' both locally and in production.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
