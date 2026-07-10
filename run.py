#!/usr/bin/env python3

import os
import sys

# Register CUDA / cuDNN / TensorRT DLL directories before onnxruntime is imported.
# Must run before any `from modules import ...` that pulls in onnxruntime.
import modules.gpu_dll_setup  # noqa: F401

if __name__ == '__main__':
    if '--server' in sys.argv:
        # WebSocket server mode for Electron UI — skip tkinter entirely
        sys.argv.remove('--server')

        from modules import core
        core.parse_args()
        if not core.pre_check():
            sys.exit(1)
        core.limit_resources()

        print("[FACELESS] Starting WebSocket server mode...")
        from modules.server import main as server_main
        server_main()
    else:
        # Original tkinter GUI mode
        import tkinter_fix
        from modules import core
        core.run()
