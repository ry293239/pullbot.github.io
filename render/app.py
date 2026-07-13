"""
Pullbot API - Render Backend
Downloads model chunks from GitHub, reassembles, serves responses.
"""

import os
import sys
import json
import time
import glob
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIG
# ============================================

GITHUB_REPO = "https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main"
MODEL_DIR = "/tmp/pullbot_model"
CHUNKS_DIR = os.path.join(MODEL_DIR, "chunks")

model = None
tokenizer = None
knowledge_chunks = []

# ============================================
# DOWNLOAD & LOAD MODEL
# ============================================

def download_file(url, dest):
    """Download a file from GitHub raw"""
    print(f"   Downloading: {url.split('/')[-1]}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        f.write(r.content)

def setup_model():
    """Download model chunks and config from GitHub, reassemble, load"""
    global model, tokenizer, knowledge_chunks
    
    print("=" * 50)
    print("📥 DOWNLOADING PULLBOT MODEL")
    print("=" * 50)
    
    # 1. Download manifest to know what chunks exist
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    manifest_url = f"{GITHUB_REPO}/models/chunks/manifest.json"
    manifest_path = os.path.join(CHUNKS_DIR, "manifest.json")
    download_file(manifest_url, manifest_path)
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"   Model: {manifest.get('total_size_mb', '?')}MB, {manifest.get('num_chunks', 0)} chunks")
    
    # 2. Download config files
    config_files = ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']
    for fname in config_files:
        url = f"{GITHUB_REPO}/models/chunks/{fname}"
        dest = os.path.join(CHUNKS_DIR, fname)
        try:
            download_file(url, dest)
        except:
            pass  # Some files might not exist
    
    # 3. Download model chunks
    chunks = manifest.get('chunks', [])
    for chunk_path in chunks:
        chunk_name = os.path.basename(chunk_path)
        url = f"{GITHUB_REPO}/{chunk_path}"
        dest = os.path.join(CHUNKS_DIR, chunk_name)
        download_file(url, dest)
    
    # 4. Reassemble model
    weights_file = manifest.get('weights_filename', 'model.safetensors')
    reassembled = os.path.join(CHUNKS_DIR, weights_file)
    
    if not os.path.exists(reassembled):
        print(f"   🧩 Reassembling model from {len(chunks)} chunks...")
        with open(reassembled, 'wb') as outfile:
            for chunk_path in chunks:
                chunk_name = os.path.basename(chunk_path)
                chunk_file = os.path.join(CHUNKS_DIR, chunk_name)
                with open(chunk_file, 'rb') as infile:
                    outfile.write(infile.read())
        print(f"   ✅ Reassembled")
    
    # 5. Load model
    print("📂 Loading model into memory...")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(CHUNKS_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        CHUNKS_DIR,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    elapsed = time.time() - start
    params = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded in {elapsed:.1f}s ({params:,} parameters)")
    
    # 6. Download knowledge store
    try:
        knowledge_url = f"{GITHUB_REPO}/knowledge/store.json"
        r = requests.get(knowledge_url, timeout=30)
        if r.status_code == 200:
            knowledge_chunks = r.json()
            print(f"📚 Loaded {len(knowledge_chunks)} knowledge chunks")
    except:
        print("⚠️ No knowledge store found")
        knowledge_chunks = []

# ============================================
# KNOWLEDGE SEARCH
# ============================================

def search_knowledge(query, top_k=3):
    if not knowledge_chunks:
        return []
    
    q = query.lower()
    words = [w for w in q.split() if len(w) > 2]
    scored = []
    
    for chunk in knowledge_chunks:
        text = chunk.get('text', '').lower()
        score = sum(1 for w in words if w in text)
        if q in text:
            score += 5
        if score > 0:
            scored.append((score, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]

# ============================================
# GENERATION
# ============================================

def generate_response(question, max_tokens=200, temperature=0.8):
    results = search_knowledge(question)
    context = "\n\n".join([r['text'][:300] for r in results[:2]]) if results else ""
    
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"User: {question}\n\nPullbot:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return {
        'question': question,
        'response': response,
        'context_used': len(results) > 0,
        'sources': [r.get('source', 'unknown') for r in results[:2]]
    }

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'model_loaded': model is not None,
        'parameters': sum(p.numel() for p in model.parameters()) if model else 0,
        'knowledge_chunks': len(knowledge_chunks)
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok' if model else 'loading'})

@app.route('/ask', methods=['GET', 'POST'])
def ask():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        question = data.get('q', '')
    else:
        question = request.args.get('q', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    if not model:
        return jsonify({'error': 'Model still loading'}), 503
    
    result = generate_response(question)
    return jsonify(result)

# ============================================
# STARTUP
# ============================================

print("\n🚀 Starting Pullbot API...")
setup_model()
print("\n✅ Pullbot API ready!")
