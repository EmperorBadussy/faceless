#!/usr/bin/env python3

import os
import sys
import glob

# Add NVIDIA CUDA DLL paths (pip-installed) to PATH before any imports
# This is required for onnxruntime to find cublasLt64_12.dll, cudnn, etc.
try:
    _nvidia_base = os.path.join(
        os.path.dirname(os.path.dirname(__import__('nvidia.cublas').__file__))
    )
    for _pkg in glob.glob(os.path.join(_nvidia_base, 'nvidia', '*')):
        for _sub in ('bin', 'lib'):
            _p = os.path.join(_pkg, _sub)
            if os.path.isdir(_p) and _p not in os.environ.get('PATH', ''):
                os.environ['PATH'] = _p + os.pathsep + os.environ['PATH']
except Exception:
    pass  # nvidia packages not installed, skip

if __name__ == '__main__':
    if '--server' in sys.argv:
        # WebSocket server mode for Electron UI — skip tkinter entirely
        sys.argv.remove('--server')

        from modules import core
        core.parse_args()
        if not core.pre_check():
            sys.exit(1)
        core.limit_resources()

        print("[PHANTOM] Starting WebSocket server mode...")
        from modules.server import main as server_main
        server_main()
    else:
        # Original tkinter GUI mode
        import tkinter_fix
        from modules import core
        core.run()
