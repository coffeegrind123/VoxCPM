#!/usr/bin/env python3
"""Pre-download all models using hf_transfer for speed and resilience."""
import os
import sys
import threading
import torch
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download, hf_hub_download
from modelscope import snapshot_download as ms_snapshot_download

print("=== VoxCPM Model Preloader ===", flush=True)
print(f"CUDA: {torch.cuda.is_available()}, hf_transfer: ", end="", flush=True)
try:
    import hf_transfer
    print("loaded", flush=True)
except ImportError:
    print("not available", flush=True)

_cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
_modelscope_dir = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
_running = True

def _cache_size_mb(path):
    total = 0
    if os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
    return total / (1024 * 1024)

def _progress_reporter():
    while _running:
        hf_mb = _cache_size_mb(_cache_dir)
        ms_mb = _cache_size_mb(_modelscope_dir)
        print(f"[progress] hf: {hf_mb:.0f}MB  modelscope: {ms_mb:.0f}MB", flush=True)
        time.sleep(30)

_reporter = threading.Thread(target=_progress_reporter, daemon=True)
_reporter.start()

def download_with_retry(fn, name, **kwargs):
    for attempt in range(1, 4):
        try:
            print(f"[{attempt}/3] Downloading {name}...", flush=True)
            result = fn(**kwargs)
            print(f"{name} done.", flush=True)
            return result
        except Exception as e:
            print(f"Attempt {attempt} failed: {e}", flush=True)
            if attempt < 3:
                time.sleep(5)
            else:
                raise

download_with_retry(snapshot_download, "VoxCPM2", repo_id="openbmb/VoxCPM2")
download_with_retry(ms_snapshot_download, "ZipEnhancer",
                    repo_id="iic/speech_zipenhancer_ans_multiloss_16k_base")
download_with_retry(ms_snapshot_download, "SenseVoiceSmall",
                    repo_id="iic/SenseVoiceSmall")

_running = False
print("\n=== All models downloaded ===", flush=True)
print("Starting app...", flush=True)
os.execlp("python", "python", "-u", "app.py", "--port", "8808")
