import { User, Upload } from 'lucide-react'

interface SourceFaceProps {
  thumbnail: string | null
  detected: boolean
  onSelect: () => void
}

export function SourceFace({ thumbnail, detected, onSelect }: SourceFaceProps) {
  return (
    <div className="py-2">
      <span className="text-[14px] font-[Rajdhani] text-text-secondary block mb-2">Source Face</span>
      <div className="flex items-center gap-3">
        {/* Thumbnail */}
        <div
          className={`
            w-[64px] h-[64px] rounded-lg border overflow-hidden flex items-center justify-center
            ${detected
              ? 'border-accent-primary shadow-[0_0_10px_rgba(168,85,247,0.3)]'
              : thumbnail
                ? 'border-warning'
                : 'border-phantom-600'}
            bg-phantom-800
          `}
        >
          {thumbnail ? (
            <img
              src={`data:image/jpeg;base64,${thumbnail}`}
              alt="Source face"
              className="w-full h-full object-cover"
            />
          ) : (
            <User size={24} className="text-text-tertiary" />
          )}
        </div>

        {/* Select button */}
        <button
          onClick={onSelect}
          className="flex-1 flex items-center justify-center gap-2 py-2 rounded-md border border-phantom-600 bg-phantom-800 text-[13px] font-[Rajdhani] text-text-secondary hover:border-accent-dim hover:text-text-primary transition-all"
        >
          <Upload size={14} />
          Select Face
        </button>
      </div>

      {/* Status text */}
      {thumbnail && !detected && (
        <p className="mt-1.5 text-[11px] font-[Rajdhani] text-warning">
          No face detected in image
        </p>
      )}
      {detected && (
        <p className="mt-1.5 text-[11px] font-[Rajdhani] text-success">
          Face detected
        </p>
      )}
    </div>
  )
}
