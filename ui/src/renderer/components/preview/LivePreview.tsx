import { useRef } from 'react'
import { useFrameStream } from '@/lib/hooks/useFrameStream'
import { useConnectionStore } from '@/lib/stores/connection'
import { FpsOverlay } from './FpsOverlay'
import { Video, Loader2 } from 'lucide-react'

export function LivePreview() {
  const imgRef = useRef<HTMLImageElement>(null)
  const streaming = useConnectionStore((s) => s.streaming)
  const status = useConnectionStore((s) => s.status)

  // This hook updates imgRef.current.src directly — bypasses React render
  useFrameStream(imgRef)

  if (status === 'disconnected') {
    return (
      <div className="flex flex-col items-center justify-center gap-4 text-text-tertiary">
        <Loader2 size={32} className="animate-spin text-accent-dim" />
        <span className="font-[Rajdhani] text-[15px]">Connecting to Python backend...</span>
        <span className="font-[JetBrains_Mono] text-[11px] text-text-tertiary">ws://127.0.0.1:7865</span>
      </div>
    )
  }

  if (!streaming) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 text-text-tertiary">
        <Video size={48} className="text-phantom-600" />
        <span className="font-[Rajdhani] text-[16px]">No Preview</span>
        <span className="font-[Rajdhani] text-[13px] text-text-tertiary">
          Select a source face and click Start Preview
        </span>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full flex items-center justify-center p-4">
      {/* Live frame */}
      <img
        ref={imgRef}
        alt="Live preview"
        className="max-w-full max-h-full object-contain rounded-lg border-2 border-accent-primary/30 animate-stream-pulse"
      />

      {/* FPS overlay */}
      <FpsOverlay />
    </div>
  )
}
