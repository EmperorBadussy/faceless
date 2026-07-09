/**
 * FACELESS — Frame stream hook.
 *
 * Receives binary WebSocket messages (JPEG blobs), creates Blob URLs,
 * and updates an <img> element's src directly via ref (no React re-render).
 */

import { useEffect, RefObject } from 'react'
import { onBinaryFrame } from './useWebSocket'

export function useFrameStream(imgRef: RefObject<HTMLImageElement | null>): void {
  useEffect(() => {
    // URL currently decoded and shown by the <img>. We must NOT revoke a Blob URL
    // while the browser is still decoding/painting it, or the frame tears (torn
    // stripes / partial render). This bug only shows at high frame rates, when a
    // new frame arrives before the previous one finished decoding.
    let displayedUrl: string | null = null

    onBinaryFrame.current = (blob: Blob) => {
      const img = imgRef.current
      const url = URL.createObjectURL(blob)
      if (!img) {
        URL.revokeObjectURL(url)
        return
      }

      const prevUrl = displayedUrl
      // Revoke the previously displayed frame only once the new one has decoded
      // (or failed), so the visible frame is never freed mid-decode.
      img.onload = img.onerror = () => {
        if (prevUrl) URL.revokeObjectURL(prevUrl)
      }
      displayedUrl = url

      // Direct DOM update — bypasses React render cycle
      img.src = url
    }

    return () => {
      onBinaryFrame.current = null
      if (displayedUrl) {
        URL.revokeObjectURL(displayedUrl)
        displayedUrl = null
      }
    }
  }, [imgRef])
}
