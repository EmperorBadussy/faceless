/**
 * PHANTOM-FACE — Controls state (Zustand).
 *
 * All toggles, sliders, camera selection, source face state.
 */

import { create } from 'zustand'

interface Controls {
  many_faces: boolean
  poisson_blend: boolean
  mouth_mask: boolean
  color_correction: boolean
  face_enhancer: boolean
  face_enhancer_gpen256: boolean
  face_enhancer_gpen512: boolean
  live_mirror: boolean
  show_fps: boolean
  virtual_camera: boolean
  opacity: number
  sharpness: number
  quality_preset: string
  selectedCamera: number
}

interface Camera {
  index: number
  name: string
}

interface ControlsState {
  controls: Controls
  cameras: Camera[]
  sourceThumbnail: string | null
  sourceDetected: boolean

  setControl: (key: string, value: unknown) => void
  setCameras: (cameras: Camera[]) => void
  setSourceFace: (detected: boolean, thumbnail: string | null) => void
  applyServerState: (serverControls: Record<string, unknown>) => void
}

const defaultControls: Controls = {
  many_faces: false,
  poisson_blend: false,
  mouth_mask: false,
  color_correction: false,
  face_enhancer: false,
  face_enhancer_gpen256: false,
  face_enhancer_gpen512: false,
  live_mirror: false,
  show_fps: false,
  virtual_camera: false,
  opacity: 1.0,
  sharpness: 0.0,
  quality_preset: 'normal',
  selectedCamera: 0,
}

export const useControlsStore = create<ControlsState>((set) => ({
  controls: { ...defaultControls },
  cameras: [],
  sourceThumbnail: null,
  sourceDetected: false,

  setControl: (key, value) =>
    set((state) => ({
      controls: { ...state.controls, [key]: value },
    })),

  setCameras: (cameras) => set({ cameras }),

  setSourceFace: (detected, thumbnail) =>
    set({ sourceDetected: detected, sourceThumbnail: thumbnail }),

  applyServerState: (serverControls) =>
    set((state) => {
      const merged = { ...state.controls }
      for (const [key, value] of Object.entries(serverControls)) {
        if (key in merged) {
          ;(merged as Record<string, unknown>)[key] = value
        }
      }
      return { controls: merged as Controls }
    }),
}))
