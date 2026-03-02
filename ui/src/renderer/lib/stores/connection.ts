/**
 * FACELESS — Connection state (Zustand).
 *
 * Tracks WebSocket status, FPS stats, streaming state.
 */

import { create } from 'zustand'

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected'

interface ConnectionState {
  status: ConnectionStatus
  streaming: boolean
  processFps: number
  detectFps: number
  captureFps: number
  frameCount: number
  lastStatusMessage: string

  // WebSocket reference (not serializable — stored outside Zustand)
  // Managed by useWebSocket hook

  // Actions
  setStatus: (status: ConnectionStatus) => void
  setStreaming: (streaming: boolean) => void
  setStats: (stats: { process_fps: number; detect_fps: number; capture_fps: number; frame_count: number; is_running: boolean }) => void
  setStatusMessage: (message: string) => void

  // WebSocket send (set by hook)
  sendJson: (msg: Record<string, unknown>) => void
  setSendJson: (fn: (msg: Record<string, unknown>) => void) => void
}

const noop = () => {}

export const useConnectionStore = create<ConnectionState>((set) => ({
  status: 'disconnected',
  streaming: false,
  processFps: 0,
  detectFps: 0,
  captureFps: 0,
  frameCount: 0,
  lastStatusMessage: '',

  sendJson: noop,

  setStatus: (status) => set({ status }),
  setStreaming: (streaming) => set({ streaming }),
  setStats: (stats) => set({
    processFps: stats.process_fps,
    detectFps: stats.detect_fps,
    captureFps: stats.capture_fps,
    frameCount: stats.frame_count,
    streaming: stats.is_running,
  }),
  setStatusMessage: (message) => set({ lastStatusMessage: message }),

  setSendJson: (fn) => set({ sendJson: fn }),
}))
