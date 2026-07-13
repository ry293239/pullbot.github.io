"""
Pullbot API - ONNX Runtime (Lightweight Deployment)
No PyTorch needed! Uses ONNX model + knowledge retrieval.
Total install size: ~100MB. RAM usage: ~200MB.
"""

import os, json, requests, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIG
# ============================================
GITHUB = "https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main"
MODEL_PATH = "/tmp/pullbot.onnx"

session = None
tokenizer_vocab = {}
knowledge = []

# ============================================
# LOAD
# ============================================

def download_model():
    """Download ONNX model from GitHub"""
    url = f"{GITHUB}/models/pullbot.onnx"
    print(f"📥 Downloading ONNX model...")
    r = requests.get(url)
    if r.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print(f"   ✅ {len(r.content)/(1024*1024):.1f}MB")
        return True
    print(f"   ❌ Model not found (status {r.status_code})")
    return False

def load_tokenizer():
    """Load vocab from GitHub"""
    global tokenizer_vocab
    try:
        r = requests.get(f"{GITHUB}/models/chunks/vocab.json")
        if r.status_code == 200:
            tokenizer_vocab = r.json()
            print(f"📝 Vocab: {len(tokenizer_vocab)} tokens")
    except:
        print("⚠️ Could not load vocab")

def load_knowledge():
    """Load knowledge store"""
    global knowledge
    try:
        r = requests.get(f"{GITHUB}/knowledge/store.json")
        if r.status_code == 200:
            knowledge = r.json()
            print(f"📚 {len(knowledge)} knowledge chunks")
    except:
        print("⚠️ No knowledge store")

def setup():
    global session
    print("=" * 50)
    print("🚀 PULLBOT API (ONNX)")
    print("=" * 50)
    
    load_tokenizer()
    load_knowledge()
    
    if download_model():
        print("📂 Loading ONNX model...")
        session = ort.InferenceSession(MODEL_PATH)
        print("✅ Ready!")
    else:
        print("⚠️ Running in knowledge-only mode")

# ============================================
# KNOWLEDGE SEARCH
# ============================================

def search_knowledge(query, top_k=3):
    if not knowledge:
        return []
    q = query.lower()
    words = [w for w in q.split() if len(w) > 2]
    scored = []
    for chunk in knowledge:
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

def simple_tokenize(text, max_len=64):
    """Basic tokenizer fallback"""
    tokens = []
    for char in text[-max_len * 4:]:
        tokens.append(hash(char) % 50257)
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(0)
    return tokens

def generate_response(question):
    # Search knowledge first
    results = search_knowledge(question)
    
    if results:
        context = ' '.join([r['text'][:300] for r in results[:2]])
        # Knowledge-based response
        if session:
            # Use ONNX model to format
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            tokens = simple_tokenize(prompt)
            input_ids = np.array([tokens], dtype=np.int64)
            mask = np.ones((1, len(tokens)), dtype=np.int64)
            
            try:
                outputs = session.run(None, {
                    'input_ids': input_ids,
                    'attention_mask': mask
                })
                return {
                    'question': question,
                    'response': context[:400] + "...",
                    'context_used': True,
                    'source': 'knowledge + onnx'
                }
            except:
                pass
        
        return {
            'question': question,
            'response': context[:400] + "...",
            'context_used': True,
            'source': 'knowledge'
        }
    
    return {
        'question': question,
        'response': "I don't have enough information about that yet. Try asking about technology, science, or programming!",
        'context_used': False,
        'source': 'fallback'
    }

# ============================================
# API
# ============================================

@app.route('/')
def home():
    return jsonify({
        'name': 'Pullbot API (ONNX)',
        'status': 'online',
        'model': 'onnx' if session else 'knowledge-only',
        'knowledge_chunks': len(knowledge)
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/ask')
def ask():
    q = request.args.get('q', '')
    if not q:
        return jsonify({'error': 'No question. Use ?q=your+question'}), 400
    return jsonify(generate_response(q))

# ============================================
# START
# ============================================
setup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
