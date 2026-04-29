#!/usr/bin/env python3
"""Continuously run GEMM on GPU to keep it active and prevent sleep."""

import signal
import sys
import time

running = True

def handler(sig, frame):
    global running
    print(f"\nCaught signal {sig}, stopping...")
    running = False

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)

try:
    import torch
except ImportError:
    print("ERROR: torch not found")
    sys.exit(1)

# Try to get GPU
if not torch.cuda.is_available():
    # Iluvatar Corex may need special init
    try:
        # Force device creation
        device = torch.device("cuda:0")
        a = torch.randn(16, 16, device=device)
        print(f"GPU available via direct device creation")
    except Exception as e:
        print(f"Cannot access GPU: {e}")
        print("Trying CPU fallback won't help for keeping GPU alive.")
        sys.exit(1)
else:
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

N = 2048
print(f"Matrix size: {N}x{N} FP32")
print(f"Running continuous GEMM on {device} to keep GPU alive...")
print("Kill this process (kill PID or Ctrl+C) to stop.\n")

a = torch.randn(N, N, device=device)
b = torch.randn(N, N, device=device)

iteration = 0
while running:
    for _ in range(10):
        if not running:
            break
        c = torch.mm(a, b)
    iteration += 10

    if iteration % 100 == 0:
        print(f"Iterations: {iteration}", flush=True)

    time.sleep(0.1)

print(f"Total iterations: {iteration}")
print("Exiting.")
