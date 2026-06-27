import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, calls to /api are proxied to the FastAPI backend (port 8001 on this
// machine). In production set VITE_API_BASE_URL to the deployed backend URL.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://localhost:8001' },
  },
})
