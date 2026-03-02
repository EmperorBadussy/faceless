interface SliderProps {
  label: string
  value: number
  min?: number
  max?: number
  step?: number
  onChange: (value: number) => void
}

export function Slider({ label, value, min = 0, max = 1, step = 0.05, onChange }: SliderProps) {
  const percent = ((value - min) / (max - min)) * 100

  return (
    <div className="py-1.5">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[14px] font-[Rajdhani] text-text-secondary">{label}</span>
        <span className="text-[12px] font-[JetBrains_Mono] text-accent-tertiary">
          {value.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-[4px] rounded-full appearance-none cursor-pointer"
        style={{
          background: `linear-gradient(to right, #a855f7 0%, #a855f7 ${percent}%, #2a2a3d ${percent}%, #2a2a3d 100%)`,
        }}
      />
      <style>{`
        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: #c084fc;
          border: 2px solid #a855f7;
          box-shadow: 0 0 8px rgba(168, 85, 247, 0.5);
          cursor: pointer;
        }
        input[type="range"]::-webkit-slider-thumb:hover {
          box-shadow: 0 0 14px rgba(168, 85, 247, 0.7);
        }
      `}</style>
    </div>
  )
}
