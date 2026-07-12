"""
Run this ONCE to download DistilGPT2 and split into chunks.
After running, push the models/ directory to your repo.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import shutil

CHUNK_SIZE = 45 * 1024 * 1024  # 45MB chunks

def download_and_chunk():
    print("📥 Downloading DistilGPT2...")
    
    # Download
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Save full model temporarily
    os.makedirs("models/full_temp", exist_ok=True)
    model.save_pretrained("models/full_temp")
    tokenizer.save_pretrained("models/full_temp")
    
    # Split the big safetensors file into chunks
    safetensors_file = "models/full_temp/model.safetensors"
    
    if not os.path.exists(safetensors_file):
        # Older format - look for pytorch_model.bin
        safetensors_file = "models/full_temp/pytorch_model.bin"
    
    print(f"📦 Splitting {safetensors_file} into {CHUNK_SIZE//1024//1024}MB chunks...")
    
    os.makedirs("models/chunks", exist_ok=True)
    
    with open(safetensors_file, 'rb') as f:
        data = f.read()
    
    total_size = len(data)
    chunks = []
    
    for i in range(0, total_size, CHUNK_SIZE):
        chunk = data[i:i+CHUNK_SIZE]
        chunk_name = f"models/chunks/model_chunk_{i//CHUNK_SIZE:03d}.bin"
        with open(chunk_name, 'wb') as f:
            f.write(chunk)
        chunks.append(chunk_name)
    
    # Save manifest
    manifest = {
        "total_size": total_size,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "original_file": os.path.basename(safetensors_file)
    }
    
    with open("models/chunks/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Copy config files (small, no need to chunk)
    for fname in ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json']:
        src = f"models/full_temp/{fname}"
        if os.path.exists(src):
            shutil.copy(src, f"models/chunks/{fname}")
    
    # Cleanup
    shutil.rmtree("models/full_temp")
    
    print(f"✅ Done! {len(chunks)} chunks saved to models/chunks/")
    print(f"   Total size: {total_size//1024//1024}MB")
    print(f"   Now push models/chunks/ to your repo")

if __name__ == '__main__':
    download_and_chunk()
