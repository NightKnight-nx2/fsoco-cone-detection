"""Print the GPU / CUDA situation and suggest a batch size. Run this first: python check_gpu.py"""
import shutil
import subprocess

import torch

print(f"torch {torch.__version__} | CUDA build: {torch.version.cuda} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gb = p.total_memory / 1024 ** 3
        batch = 8 if gb < 5 else 16 if gb < 10 else 32 if gb < 16 else 64
        print(f"  GPU {i}: {p.name} | {gb:.1f} GB -> suggested --batch {batch} (or --batch -1 for auto)")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("  Apple MPS available (works, but slower than CUDA).")
else:
    print("  No GPU visible to PyTorch. Use notebooks/colab_train.ipynb (free T4), or on Windows install the CUDA build:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip())
