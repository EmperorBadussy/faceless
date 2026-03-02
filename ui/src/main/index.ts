/**
 * PHANTOM-FACE — Electron main process.
 *
 * Frameless window with custom titlebar.
 * Spawns Python WebSocket server as child process.
 */

import { app, BrowserWindow, shell } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { registerIpcHandlers } from './ipc-handlers'
import { startPython, stopPython } from './python-bridge'

let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 960,
    minHeight: 600,
    frame: false,
    backgroundColor: '#0a0a0f',
    titleBarStyle: 'hidden',
    show: false,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
    },
  })

  // Show when ready (prevents white flash)
  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  // Dev: load from vite dev server. Prod: load built files.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

// ── App Lifecycle ──────────────────────────────────────────────────────────

app.whenReady().then(async () => {
  electronApp.setAppUserModelId('com.phantom.face')

  // Dev: open devtools with F12
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  registerIpcHandlers()

  // Start Python backend
  try {
    await startPython()
    console.log('[main] Python backend ready')
  } catch (err) {
    console.error('[main] Python failed to start:', err)
    // Continue anyway — UI will show connection error
  }

  createWindow()
})

app.on('window-all-closed', () => {
  stopPython()
  app.quit()
})

app.on('before-quit', () => {
  stopPython()
})
