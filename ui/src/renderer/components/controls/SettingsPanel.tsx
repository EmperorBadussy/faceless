import { Play, Square } from 'lucide-react'
import { useConnectionStore } from '@/lib/stores/connection'
import { useControlsStore } from '@/lib/stores/controls'
import { SourceFace } from './SourceFace'
import { QualityPreset } from './QualityPreset'
import { CameraSelect } from './CameraSelect'
import { ToggleSwitch } from './ToggleSwitch'
import { Slider } from './Slider'

export function SettingsPanel() {
  const sendJson = useConnectionStore((s) => s.sendJson)
  const streaming = useConnectionStore((s) => s.streaming)
  const controls = useControlsStore((s) => s.controls)
  const cameras = useControlsStore((s) => s.cameras)
  const sourceThumbnail = useControlsStore((s) => s.sourceThumbnail)
  const sourceDetected = useControlsStore((s) => s.sourceDetected)
  const setControl = useControlsStore((s) => s.setControl)

  const handleSelectFace = async () => {
    const path = await window.phantomFace.openImageDialog()
    if (path) {
      sendJson({ type: 'set_source', path })
    }
  }

  const handleControl = (key: string, value: boolean | number | string) => {
    setControl(key, value)
    sendJson({ type: 'control', key, value })
  }

  const handleVcamToggle = (enabled: boolean) => {
    setControl('virtual_camera', enabled)
    sendJson({ type: 'toggle_vcam', enabled })
  }

  const handleStart = () => {
    sendJson({ type: 'start_preview', camera_index: controls.selectedCamera })
  }

  const handleStop = () => {
    sendJson({ type: 'stop_preview' })
  }

  const handleRefreshCameras = () => {
    sendJson({ type: 'get_cameras' })
  }

  return (
    <div className="p-4 flex flex-col h-full stagger-children">
      {/* Source Face */}
      <SourceFace
        thumbnail={sourceThumbnail}
        detected={sourceDetected}
        onSelect={handleSelectFace}
      />

      <div className="h-px bg-phantom-700 my-2" />

      {/* Quality Preset */}
      <QualityPreset
        value={controls.quality_preset}
        onChange={(v) => handleControl('quality_preset', v)}
      />

      <div className="h-px bg-phantom-700 my-2" />

      {/* Camera */}
      <CameraSelect
        cameras={cameras}
        selected={controls.selectedCamera}
        onChange={(idx) => {
          useControlsStore.getState().setControl('selectedCamera', idx)
        }}
        onRefresh={handleRefreshCameras}
      />

      <div className="h-px bg-phantom-700 my-2" />

      {/* Processing toggles */}
      <div>
        <span className="text-[12px] font-[Orbitron] tracking-widest text-text-tertiary uppercase block mb-1">
          Processing
        </span>
        <ToggleSwitch
          label="Many Faces"
          checked={controls.many_faces}
          onChange={(v) => handleControl('many_faces', v)}
        />
        <ToggleSwitch
          label="Poisson Blend"
          checked={controls.poisson_blend}
          onChange={(v) => handleControl('poisson_blend', v)}
        />
        <ToggleSwitch
          label="Mouth Mask"
          checked={controls.mouth_mask}
          onChange={(v) => handleControl('mouth_mask', v)}
        />
        <ToggleSwitch
          label="Color Correction"
          checked={controls.color_correction}
          onChange={(v) => handleControl('color_correction', v)}
        />
        <ToggleSwitch
          label="Face Enhancer"
          checked={controls.face_enhancer}
          onChange={(v) => handleControl('face_enhancer', v)}
        />
        <ToggleSwitch
          label="GPEN 256"
          checked={controls.face_enhancer_gpen256}
          onChange={(v) => handleControl('face_enhancer_gpen256', v)}
        />
        <ToggleSwitch
          label="GPEN 512"
          checked={controls.face_enhancer_gpen512}
          onChange={(v) => handleControl('face_enhancer_gpen512', v)}
        />
        <ToggleSwitch
          label="Mirror"
          checked={controls.live_mirror}
          onChange={(v) => handleControl('live_mirror', v)}
        />
      </div>

      <div className="h-px bg-phantom-700 my-2" />

      {/* Virtual Camera */}
      <div>
        <span className="text-[12px] font-[Orbitron] tracking-widest text-text-tertiary uppercase block mb-1">
          Output
        </span>
        <ToggleSwitch
          label="Virtual Camera"
          checked={controls.virtual_camera}
          onChange={handleVcamToggle}
        />
      </div>

      <div className="h-px bg-phantom-700 my-2" />

      {/* Sliders */}
      <Slider
        label="Opacity"
        value={controls.opacity}
        onChange={(v) => handleControl('opacity', v)}
      />
      <Slider
        label="Sharpness"
        value={controls.sharpness}
        onChange={(v) => handleControl('sharpness', v)}
      />

      {/* Spacer */}
      <div className="flex-1" />

      {/* Start / Stop buttons */}
      <div className="flex flex-col gap-2 pt-3">
        {!streaming ? (
          <button
            onClick={handleStart}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-md bg-accent-primary hover:bg-accent-secondary text-white font-[Rajdhani] font-semibold text-[15px] transition-all shadow-[0_0_15px_rgba(168,85,247,0.3)] hover:shadow-[0_0_25px_rgba(168,85,247,0.5)]"
          >
            <Play size={16} />
            START PREVIEW
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-md bg-phantom-700 hover:bg-error/80 text-text-secondary hover:text-white font-[Rajdhani] font-semibold text-[15px] transition-all border border-phantom-600 hover:border-error"
          >
            <Square size={14} />
            STOP PREVIEW
          </button>
        )}
      </div>
    </div>
  )
}
