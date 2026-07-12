"""
Downloads DistilGPT2 ONCE and splits into chunks for repo storage.
Run this manually one time via GitHub Actions.
After it completes, the model lives in your repo forever.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import shutil
import time

CHUNK_SIZE_MB = 45
CHUNK_SIZE = CHUNK_SIZE_MB * 1024 * 1024

def download_and_chunk():
    print("=" * 50)
    print("📥 PULLBOT MODEL CHUNKER")
    print("=" * 50)
    
    # Step 1: Download
    print("\n📥 Downloading DistilGPT2 (this takes ~3 min)...")
    start = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   ✅ Downloaded in {time.time() - start:.0f}s")
    
    # Step 2: Save to temp location
    temp_dir = os.path.join(REPO_ROOT, "models", "temp_full")
    os.makedirs(temp_dir, exist_ok=True)
    
    print("\n💾 Saving full model...")
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    
    # Step 3: Find the model weights file
    safetensors_path = os.path.join(temp_dir, "model.safetensors")
    pytorch_path = os.path.join(temp_dir, "pytorch_model.bin")
    
    if os.path.exists(safetensors_path):
        weights_path = safetensors_path
    elif os.path.exists(pytorch_path):
        weights_path = pytorch_path
    else:
        print("❌ Could not find model weights file!")
        return
    
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    print(f"   Model weights: {file_size_mb:.1f}MB")
    
    # Step 4: Split into chunks
    chunks_dir = os.path.join(REPO_ROOT, "models", "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    print(f"\n🔪 Splitting into {CHUNK_SIZE_MB}MB chunks...")
    
    with open(weights_path, 'rb') as f:
        data = f.read()
    
    total_size = len(data)
    chunks = []
    
    for i in range(0, total_size, CHUNK_SIZE):
        chunk_data = data[i:i + CHUNK_SIZE]
        chunk_name = f"model_chunk_{i // CHUNK_SIZE:03d}.bin"
        chunk_path = os.path.join(chunks_dir, chunk_name)
        
        with open(chunk_path, 'wb') as f:
            f.write(chunk_data)
        
        chunk_size_mb = len(chunk_data) / (1024 * 1024)
        chunks.append(f"models/chunks/{chunk_name}")
        print(f"   ✅ {chunk_name} ({chunk_size_mb:.1f}MB)")
    
    # Step 5: Save manifest
    manifest = {
        "model_name": "distilgpt2",
        "total_size": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 1),
        "num_chunks": len(chunks),
        "chunks": chunks,
        "weights_filename": os.path.basename(weights_path),
        "created": time.time()
    }
    
    manifest_path = os.path.join(chunks_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Step 6: Copy config files (tiny, no need to chunk)
    config_files = ['config.json', 'tokenizer_config.json', 'vocab.json', 
                    'merges.txt', 'special_tokens_map.json', 'tokenizer.json']
    
    for fname in config_files:
        src = os.path.join(temp_dir, fname)
        if os.path.exists(src):
            dst = os.path.join(chunks_dir, fname)
            shutil.copy(src, dst)
            print(f"   📋 Copied {fname}")
    
    # Step 7: Cleanup
    shutil.rmtree(temp_dir)
    
    print(f"\n✅ DONE! {len(chunks)} chunks saved to models/chunks/")
    print(f"   Total: {total_size / (1024*1024):.1f}MB")
    print(f"   Now commit and push models/chunks/ to your repo.")
    print(f"   The model will load from chunks from now on — no more downloading!")

if __name__ == '__main__':
    download_and_chunk()
