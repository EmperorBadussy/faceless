/**
 * PHANTOM-FACE — IPC handlers for window controls and file dialogs.
 */

import { ipcMain, BrowserWindow, dialog, app } from 'electron'

export function registerIpcHandlers(): void {
  // ── Window Controls ────────────────────────────────────────────────────
  ipcMain.on('window:minimize', (event) => {
    BrowserWindow.fromWebContents(event.sender)?.minimize()
  })

  ipcMain.on('window:maximize', (event) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    if (win) {
      win.isMaximized() ? win.unmaximize() : win.maximize()
    }
  })

  ipcMain.on('window:close', (event) => {
    BrowserWindow.fromWebContents(event.sender)?.close()
  })

  ipcMain.handle('window:isMaximized', (event) => {
    return BrowserWindow.fromWebContents(event.sender)?.isMaximized() ?? false
  })

  // ── File Dialogs ───────────────────────────────────────────────────────
  ipcMain.handle('dialog:openImage', async (event) => {
    const win = BrowserWindow.fromWebContents(event.sender)
    if (!win) return null

    const result = await dialog.showOpenDialog(win, {
      title: 'Select Source Face',
      filters: [
        { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'bmp', 'webp'] }
      ],
      properties: ['openFile'],
    })

    return result.canceled ? null : result.filePaths[0]
  })

  // ── App Info ───────────────────────────────────────────────────────────
  ipcMain.handle('app:getVersion', () => {
    return app.getVersion() || '1.0.0'
  })
}
