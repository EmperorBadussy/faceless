/**
 * PHANTOM-FACE — Frame stream hook.
 *
 * Receives binary WebSocket messages (JPEG blobs), creates Blob URLs,
 * and updates an <img> element's src directly via ref (no React re-render).
 */

import { useEffect, RefObject } from 'react'
import { onBinaryFrame } from './useWebSocket'

export function useFrameStream(imgRef: RefObject<HTMLImageElement | null>): void {
  useEffect(() => {
    let currentUrl: string | null = null

    onBinaryFrame.current = (blob: Blob) => {
      // Revoke previous Blob URL to prevent memory leak
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl)
      }

      currentUrl = URL.createObjectURL(blob)

      // Direct DOM update — bypasses React render cycle
      if (imgRef.current) {
        imgRef.current.src = currentUrl
      }
    }

    return () => {
      onBinaryFrame.current = null
      if (currentUrl) {
        URL.revokeObjectURL(currentUrl)
        currentUrl = null
      }
    }
  }, [imgRef])
}
