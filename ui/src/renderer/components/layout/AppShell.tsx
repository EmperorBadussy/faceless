import { SettingsPanel } from '@/components/controls/SettingsPanel'
import { LivePreview } from '@/components/preview/LivePreview'
import { StatusBar } from '@/components/preview/StatusBar'

export function AppShell() {
  return (
    <div className="absolute top-[36px] bottom-0 left-0 right-0 flex flex-col">
      <div className="flex flex-1 min-h-0">
        {/* Left panel — Controls */}
        <div className="w-[280px] min-w-[280px] border-r border-phantom-800 bg-phantom-900/60 backdrop-blur-sm overflow-y-auto">
          <SettingsPanel />
        </div>

        {/* Main area — Preview */}
        <div className="flex-1 flex items-center justify-center bg-phantom-950 relative">
          <LivePreview />
        </div>
      </div>

      {/* Bottom status bar */}
      <StatusBar />
    </div>
  )
}
