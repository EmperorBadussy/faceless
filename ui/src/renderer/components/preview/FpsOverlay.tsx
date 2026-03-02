import { useConnectionStore } from '@/lib/stores/connection'

export function FpsOverlay() {
  const processFps = useConnectionStore((s) => s.processFps)
  const detectFps = useConnectionStore((s) => s.detectFps)

  return (
    <div className="absolute bottom-6 right-6 glass rounded-md px-3 py-2 font-[JetBrains_Mono] text-[12px] leading-relaxed">
      <div className="flex items-center gap-2">
        <span className="text-text-tertiary">FPS:</span>
        <span className="text-accent-tertiary font-semibold">{processFps}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-text-tertiary">Det:</span>
        <span className="text-accent-tertiary font-semibold">{detectFps}</span>
      </div>
    </div>
  )
}
