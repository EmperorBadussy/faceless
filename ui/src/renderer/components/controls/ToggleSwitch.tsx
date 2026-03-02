interface ToggleSwitchProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}

export function ToggleSwitch({ label, checked, onChange }: ToggleSwitchProps) {
  return (
    <label className="flex items-center justify-between py-1.5 cursor-pointer group">
      <span className="text-[14px] font-[Rajdhani] text-text-secondary group-hover:text-text-primary transition-colors">
        {label}
      </span>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`
          relative w-[38px] h-[20px] rounded-full transition-all duration-200
          ${checked
            ? 'bg-accent-primary shadow-[0_0_10px_rgba(168,85,247,0.4)]'
            : 'bg-phantom-600'}
        `}
      >
        <div
          className={`
            absolute top-[2px] w-[16px] h-[16px] rounded-full bg-white transition-transform duration-200
            ${checked ? 'left-[20px]' : 'left-[2px]'}
          `}
        />
      </button>
    </label>
  )
}
