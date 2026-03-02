/**
 * FACELESS — Preload script.
 *
 * Exposes `window.faceless` API to the renderer via context bridge.
 */

import { contextBridge, ipcRenderer } from 'electron'
import type { FacelessAPI } from './types'

const api: FacelessAPI = {
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

contextBridge.exposeInMainWorld('faceless', api)
