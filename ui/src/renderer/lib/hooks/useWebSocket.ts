/**
 * PHANTOM-FACE — WebSocket lifecycle hook.
 *
 * Connects to Python backend, handles reconnect, routes messages.
 * Binary messages (JPEG frames) go to a callback ref.
 * JSON messages are dispatched to Zustand stores.
 */

import { useEffect, useRef, useCallback } from 'react'
import { useConnectionStore } from '@/lib/stores/connection'
import { useControlsStore } from '@/lib/stores/controls'

const WS_URL = 'ws://127.0.0.1:7865'
const RECONNECT_DELAY = 2000
const MAX_RECONNECT_DELAY = 10000

/** Set this ref from useFrameStream to receive binary frames */
export const onBinaryFrame: { current: ((data: Blob) => void) | null } = { current: null }

export function useWebSocket(): void {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectDelay = useRef(RECONNECT_DELAY)
  const mountedRef = useRef(true)

  const {
    setStatus,
    setStreaming,
    setStats,
    setStatusMessage,
    setSendJson,
  } = useConnectionStore.getState()

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')

    const ws = new WebSocket(WS_URL)
    ws.binaryType = 'blob'
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[ws] Connected')
      setStatus('connected')
      reconnectDelay.current = RECONNECT_DELAY

      // Register send function in store
      setSendJson((msg: Record<string, unknown>) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(msg))
        }
      })

      // Request initial state + cameras
      ws.send(JSON.stringify({ type: 'get_state' }))
      ws.send(JSON.stringify({ type: 'get_cameras' }))
    }

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        // Binary frame — route to frame display (bypasses React)
        onBinaryFrame.current?.(event.data)
        return
      }

      // JSON message
      try {
        const msg = JSON.parse(event.data as string)
        handleJsonMessage(msg)
      } catch (err) {
        console.error('[ws] Failed to parse message:', err)
      }
    }

    ws.onclose = () => {
      console.log('[ws] Disconnected')
      setStatus('disconnected')
      setStreaming(false)
      scheduleReconnect()
    }

    ws.onerror = () => {
      // onclose will fire after this
    }
  }, [])

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current) return
    if (reconnectTimer.current) return

    reconnectTimer.current = setTimeout(() => {
      reconnectTimer.current = null
      reconnectDelay.current = Math.min(reconnectDelay.current * 1.5, MAX_RECONNECT_DELAY)
      connect()
    }, reconnectDelay.current)
  }, [connect])

  useEffect(() => {
    mountedRef.current = true
    connect()

    return () => {
      mountedRef.current = false
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect])
}

function handleJsonMessage(msg: Record<string, unknown>): void {
  const { setStats, setStreaming, setStatusMessage } = useConnectionStore.getState()
  const { applyServerState, setCameras, setSourceFace } = useControlsStore.getState()

  switch (msg.type) {
    case 'state':
      applyServerState(msg.controls as Record<string, unknown>)
      break

    case 'cameras': {
      const cameraList = msg.list as { index: number; name: string }[]
      setCameras(cameraList)
      // Auto-select first available camera if current selection is invalid
      if (cameraList.length > 0) {
        const currentCam = useControlsStore.getState().controls.selectedCamera
        const validIndices = cameraList.map((c) => c.index)
        if (!validIndices.includes(currentCam)) {
          useControlsStore.getState().setControl('selectedCamera', cameraList[0].index)
        }
      }
      break
    }

    case 'stats':
      setStats(msg as { process_fps: number; detect_fps: number; capture_fps: number; frame_count: number; is_running: boolean })
      break

    case 'source_face':
      setSourceFace(
        msg.detected as boolean,
        (msg.thumbnail as string) || null,
      )
      break

    case 'status':
      setStatusMessage(msg.message as string)
      if ((msg.message as string).includes('started')) {
        setStreaming(true)
      } else if ((msg.message as string).includes('stopped')) {
        setStreaming(false)
      }
      break

    case 'error':
      setStatusMessage(`Error: ${msg.message}`)
      break
  }
}
