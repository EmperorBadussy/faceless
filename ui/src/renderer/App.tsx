import { Titlebar } from '@/components/layout/Titlebar'
import { AppShell } from '@/components/layout/AppShell'
import { useWebSocket } from '@/lib/hooks/useWebSocket'

export default function App() {
  // Initialize WebSocket connection
  useWebSocket()

  return (
    <div className="h-screen w-screen overflow-hidden bg-phantom-950">
      <Titlebar />
      <AppShell />
    </div>
  )
}
