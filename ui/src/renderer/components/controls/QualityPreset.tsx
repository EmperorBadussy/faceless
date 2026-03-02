interface QualityPresetProps {
  value: string
  onChange: (value: string) => void
}

export function QualityPreset({ value, onChange }: QualityPresetProps) {
  return (
    <div className="py-1.5">
      <span className="text-[14px] font-[Rajdhani] text-text-secondary block mb-1">Quality</span>
      <div className="flex rounded-md overflow-hidden border border-phantom-600">
        <button
          onClick={() => onChange('normal')}
          className={`flex-1 py-1.5 text-[13px] font-[Rajdhani] font-semibold transition-all ${
            value === 'normal'
              ? 'bg-accent-primary text-white shadow-[0_0_10px_rgba(168,85,247,0.4)]'
              : 'bg-phantom-800 text-text-secondary hover:text-text-primary'
          }`}
        >
          Normal
        </button>
        <button
          onClick={() => onChange('high')}
          className={`flex-1 py-1.5 text-[13px] font-[Rajdhani] font-semibold transition-all ${
            value === 'high'
              ? 'bg-accent-primary text-white shadow-[0_0_10px_rgba(168,85,247,0.4)]'
              : 'bg-phantom-800 text-text-secondary hover:text-text-primary'
          }`}
        >
          High
        </button>
      </div>
    </div>
  )
}
