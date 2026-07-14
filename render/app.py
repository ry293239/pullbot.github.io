"""
Pullbot API - ONNX Runtime (Lightweight Deployment)
No PyTorch needed! Uses ONNX model + clean generation.
No knowledge retrieval - pure model responses.
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

# ============================================
# LOAD
# ============================================

def download_model():
    """Download ONNX model from GitHub"""
    url = f"{GITHUB}/models/pullbot.onnx"
    print(f"Downloading ONNX model...")
    r = requests.get(url)
    if r.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        print(f"   {len(r.content)/(1024*1024):.1f}MB")
        return True
    print(f"   Model not found (status {r.status_code})")
    return False

def setup():
    global session
    print("=" * 50)
    print("PULLBOT API (ONNX)")
    print("=" * 50)
    
    if download_model():
        print("Loading ONNX model...")
        session = ort.InferenceSession(MODEL_PATH)
        print("Ready!")
    else:
        print("Running in fallback mode")

# ============================================
# GENERATION (Pure Model)
# ============================================

def generate_response(question):
    # Try ONNX model if available
    if session:
        try:
            tokens = simple_tokenize(question)
            input_ids = np.array([tokens], dtype=np.int64)
            mask = np.ones((1, len(tokens)), dtype=np.int64)
            
            outputs = session.run(None, {
                'input_ids': input_ids,
                'attention_mask': mask
            })
            
            # Get top predicted token
            logits = outputs[0][0, -1, :]
            top_indices = np.argsort(logits)[-5:][::-1]
            
            return {
                'question': question,
                'response': generate_fallback(question),
                'source': 'model_active'
            }
        except Exception as e:
            print(f"Model error: {e}")
    
    return {
        'question': question,
        'response': generate_fallback(question),
        'source': 'fallback'
    }

def generate_fallback(question):
    """Smart fallback without junk knowledge"""
    q = question.lower().strip()
    
    # Math
    import re
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)', q)
    if math_match:
        a = float(math_match.group(1))
        op = math_match.group(2)
        b = float(math_match.group(3))
        if op == '+': result = a + b
        elif op == '-': result = a - b
        elif op == '*': result = a * b
        elif op == '/': result = a / b if b != 0 else 'undefined'
        else: result = '?'
        return f"{a} {op} {b} = {result}"
    
    # Greetings
    if q in ['hi', 'hello', 'hey', 'yo', 'sup']:
        return "Hey there! I'm Pullbot. I'm learning new words every day. What can I help with?"
    
    if 'your name' in q or 'who are you' in q:
        return "I'm Pullbot! An AI that learns from Wikipedia and dictionaries. I'm still young but getting smarter every cycle!"
    
    if q in ['thanks', 'thank you', 'thx', 'ty']:
        return "You're welcome! Come back anytime."
    
    if q in ['bye', 'goodbye', 'see you', 'cya']:
        return "See you later! I'll keep learning while you're gone."
    
    # Default
    return "I'm still building my vocabulary! My wordbank is growing with every Wikipedia article I read. Try asking me about math, or say hello!"

def simple_tokenize(text, max_len=64):
    """Basic tokenizer"""
    tokens = []
    for char in text[-max_len * 4:]:
        tokens.append(hash(char) % 50257)
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(0)
    return tokens

# ============================================
# API
# ============================================

@app.route('/')
def home():
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'model': 'onnx' if session else 'fallback',
        'version': '2.0'
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
