/**
 * PHANTOM-FACE — Python child process management.
 *
 * Spawns `python run.py --server --execution-provider cuda`
 * and monitors stdout for the "Server ready" line.
 */

import { spawn, ChildProcess } from 'child_process'
import { join } from 'path'
import { existsSync } from 'fs'

let pythonProcess: ChildProcess | null = null
let pythonReady = false

/** Find the project root (parent of ui/) */
function getProjectRoot(): string {
  // In dev: ui/ is CWD, project root is ../
  // In prod: app is packaged, project root is alongside
  const devRoot = join(__dirname, '..', '..', '..')
  if (existsSync(join(devRoot, 'run.py'))) return devRoot

  const prodRoot = join(process.resourcesPath, '..')
  if (existsSync(join(prodRoot, 'run.py'))) return prodRoot

  // Fallback: assume CWD
  return process.cwd()
}

export function startPython(executionProvider: string = 'cuda'): Promise<void> {
  return new Promise((resolve, reject) => {
    if (pythonProcess) {
      resolve()
      return
    }

    const root = getProjectRoot()
    const runPy = join(root, 'run.py')

    if (!existsSync(runPy)) {
      reject(new Error(`run.py not found at ${runPy}`))
      return
    }

    console.log(`[python-bridge] Starting: python "${runPy}" --server --execution-provider ${executionProvider}`)

    pythonProcess = spawn('python', [
      '-u',  // Unbuffered stdout — critical for "Server ready" detection
      runPy,
      '--server',
      '--execution-provider', executionProvider,
    ], {
      cwd: root,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    })

    const timeout = setTimeout(() => {
      if (!pythonReady) {
        console.log('[python-bridge] Python started (timeout — assuming ready)')
        pythonReady = true
        resolve()
      }
    }, 30000) // 30s timeout for model loading

    pythonProcess.stdout?.on('data', (data: Buffer) => {
      const text = data.toString()
      process.stdout.write(`[python] ${text}`)

      if (text.includes('Server ready') && !pythonReady) {
        pythonReady = true
        clearTimeout(timeout)
        resolve()
      }
    })

    pythonProcess.stderr?.on('data', (data: Buffer) => {
      const text = data.toString()
      // ONNX/CUDA warnings go to stderr — not errors
      process.stderr.write(`[python:err] ${text}`)
    })

    pythonProcess.on('error', (err) => {
      console.error('[python-bridge] Failed to start Python:', err)
      pythonProcess = null
      pythonReady = false
      clearTimeout(timeout)
      reject(err)
    })

    pythonProcess.on('exit', (code, signal) => {
      console.log(`[python-bridge] Python exited (code=${code}, signal=${signal})`)
      pythonProcess = null
      pythonReady = false
      clearTimeout(timeout)
    })
  })
}

export function stopPython(): void {
  if (!pythonProcess) return

  console.log('[python-bridge] Stopping Python...')

  // Send SIGTERM (graceful), then SIGKILL after 5s
  pythonProcess.kill('SIGTERM')

  const forceKill = setTimeout(() => {
    if (pythonProcess) {
      console.log('[python-bridge] Force killing Python')
      pythonProcess.kill('SIGKILL')
      pythonProcess = null
    }
  }, 5000)

  pythonProcess.on('exit', () => {
    clearTimeout(forceKill)
    pythonProcess = null
    pythonReady = false
  })
}

export function isPythonRunning(): boolean {
  return pythonProcess !== null && !pythonProcess.killed
}

export function isPythonReady(): boolean {
  return pythonReady
}
