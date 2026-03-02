import { ChevronDown } from 'lucide-react'

interface Camera {
  index: number
  name: string
}

interface CameraSelectProps {
  cameras: Camera[]
  selected: number
  onChange: (index: number) => void
  onRefresh: () => void
}

export function CameraSelect({ cameras, selected, onChange, onRefresh }: CameraSelectProps) {
  return (
    <div className="py-1.5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[14px] font-[Rajdhani] text-text-secondary">Camera</span>
        <button
          onClick={onRefresh}
          className="text-[11px] font-[Rajdhani] text-accent-tertiary hover:text-accent-primary transition-colors"
        >
          Refresh
        </button>
      </div>
      <div className="relative">
        <select
          value={selected}
          onChange={(e) => onChange(parseInt(e.target.value))}
          className="w-full appearance-none bg-phantom-800 border border-phantom-600 rounded-md px-3 py-2 text-[13px] font-[Rajdhani] text-text-primary cursor-pointer hover:border-accent-dim focus:border-accent-primary focus:outline-none transition-colors"
        >
          {cameras.length === 0 && (
            <option value={0}>No cameras found</option>
          )}
          {cameras.map((cam) => (
            <option key={cam.index} value={cam.index}>
              {cam.name}
            </option>
          ))}
        </select>
        <ChevronDown
          size={14}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-text-tertiary pointer-events-none"
        />
      </div>
    </div>
  )
}
