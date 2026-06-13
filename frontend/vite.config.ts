import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Фронт — отдельное приложение (тонкий клиент). Dev-сервер проксирует API на
// FastAPI-мозг (порт 8000). Сборка кладётся в dist/ — её отдаёт сам FastAPI.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '',
  server: {
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
      '/chats': 'http://127.0.0.1:8000',
      '/memory': 'http://127.0.0.1:8000',
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
