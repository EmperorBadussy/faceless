/**
 * PHANTOM-FACE — Preload script.
 *
 * Exposes `window.phantomFace` API to the renderer via context bridge.
 */

import { contextBridge, ipcRenderer } from 'electron'
import type { PhantomFaceAPI } from './types'

const api: PhantomFaceAPI = {
  // Window controls
  windowMinimize: () => ipcRenderer.send('window:minimize'),
  windowMaximize: () => ipcRenderer.send('window:maximize'),
  windowClose: () => ipcRenderer.send('window:close'),
  windowIsMaximized: () => ipcRenderer.invoke('window:isMaximized'),

  // File dialogs
  openImageDialog: () => ipcRenderer.invoke('dialog:openImage'),

  // App info
  getVersion: () => ipcRenderer.invoke('app:getVersion'),
}

contextBridge.exposeInMainWorld('phantomFace', api)
