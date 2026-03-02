import { useConnectionStore } from '@/lib/stores/connection'
import { Wifi, WifiOff } from 'lucide-react'

export function StatusBar() {
  const status = useConnectionStore((s) => s.status)
  const lastMessage = useConnectionStore((s) => s.lastStatusMessage)

  const isConnected = status === 'connected'

  return (
    <div className="h-[26px] flex items-center justify-between px-4 border-t border-phantom-800 bg-phantom-950/90 backdrop-blur-sm text-[11px] font-[Rajdhani]">
      {/* Connection status */}
      <div className="flex items-center gap-2">
        {isConnected ? (
          <>
            <Wifi size={12} className="text-success" />
            <span className="text-success">Connected</span>
          </>
        ) : (
          <>
            <WifiOff size={12} className="text-error" />
            <span className="text-error">Disconnected</span>
          </>
        )}
        <span className="text-text-tertiary">ws://127.0.0.1:7865</span>
      </div>

      {/* Last status message */}
      <div className="flex items-center gap-3">
        {lastMessage && (
          <span className="text-text-tertiary">{lastMessage}</span>
        )}
        <span className="text-text-tertiary font-[JetBrains_Mono] text-[10px]">
          PHANTOM-FACE v1.0.0
        </span>
      </div>
    </div>
  )
}
