import { useState, useCallback } from 'react'
import { Minus, Square, X, Copy } from 'lucide-react'

export function Titlebar() {
  const [maximized, setMaximized] = useState(false)

  const handleMaximize = useCallback(async () => {
    window.faceless.windowMaximize()
    const isMax = await window.faceless.windowIsMaximized()
    setMaximized(isMax)
  }, [])

  return (
    <div className="drag-region fixed top-0 left-0 right-0 z-50 h-[36px] flex items-center border-b border-phantom-800 bg-phantom-950/90 backdrop-blur-md px-4">
      {/* App title */}
      <span className="font-[Orbitron] text-[11px] font-bold tracking-[0.25em] text-accent-primary glow-text uppercase">
        FACELESS
      </span>

      {/* Spacer (drag region) */}
      <div className="flex-1" />

      {/* Window controls */}
      <div className="no-drag flex items-center">
        <button
          onClick={() => window.faceless.windowMinimize()}
          className="flex h-[36px] w-[46px] items-center justify-center text-text-secondary hover:bg-phantom-700 hover:text-text-primary transition-colors"
        >
          <Minus size={14} />
        </button>
        <button
          onClick={handleMaximize}
          className="flex h-[36px] w-[46px] items-center justify-center text-text-secondary hover:bg-phantom-700 hover:text-text-primary transition-colors"
        >
          {maximized ? <Copy size={12} /> : <Square size={12} />}
        </button>
        <button
          onClick={() => window.faceless.windowClose()}
          className="flex h-[36px] w-[46px] items-center justify-center text-text-secondary hover:bg-red-500/80 hover:text-white transition-colors"
        >
          <X size={14} />
        </button>
      </div>
    </div>
  )
}
