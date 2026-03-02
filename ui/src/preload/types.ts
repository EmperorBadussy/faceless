/**
 * FACELESS — Preload API type definitions.
 */

export interface FacelessAPI {
  // Window controls
  windowMinimize: () => void
  windowMaximize: () => void
  windowClose: () => void
  windowIsMaximized: () => Promise<boolean>

  // File dialogs
  openImageDialog: () => Promise<string | null>

  // App info
  getVersion: () => Promise<string>
}

declare global {
  interface Window {
    faceless: FacelessAPI
  }
}
