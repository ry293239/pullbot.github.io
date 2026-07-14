"""
Pullbot API - ONNX Runtime
Math is deterministic. Everything else: let the model cook.
"""

import os, json, requests, numpy as np, re, random
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort

app = Flask(__name__)
CORS(app)

GITHUB = "https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main"
MODEL_PATH = "/tmp/pullbot.onnx"
session = None

def download_model():
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
    print("PULLBOT API - FREE REIGN MODE")
    print("=" * 50)
    if download_model():
        print("Loading ONNX model...")
        session = ort.InferenceSession(MODEL_PATH)
        print("Ready! Model is free to say whatever it wants.")
    else:
        print("Running without model.")

def simple_tokenize(text, max_len=128):
    tokens = []
    for char in text[-max_len * 4:]:
        tokens.append(hash(char) % 50257)
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(0)
    return tokens

def generate_response(question):
    q = question.strip()
    
    # === MATH (deterministic - not AI) ===
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)', q)
    if math_match:
        a = float(math_match.group(1))
        op = math_match.group(2)
        b = float(math_match.group(3))
        try:
            if op == '+': result = a + b
            elif op == '-': result = a - b
            elif op == '*': result = a * b
            elif op == '/': result = a / b if b != 0 else 'undefined'
            else: result = '?'
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return {'question': question, 'response': f"{a} {op} {b} = {result}", 'source': 'math'}
        except:
            pass
    
    # === ONNX MODEL (free reign) ===
    if session:
        try:
            prompt = f"Question: {q}\n\nAnswer:"
            tokens = simple_tokenize(prompt, max_len=128)
            input_ids = np.array([tokens], dtype=np.int64)
            mask = np.ones((1, len(tokens)), dtype=np.int64)
            
            outputs = session.run(None, {
                'input_ids': input_ids,
                'attention_mask': mask
            })
            
            logits = outputs[0][0, -1, :]
            top_k = 50
            top_indices = np.argpartition(logits, -top_k)[-top_k:]
            top_logits = logits[top_indices]
            probs = np.exp(top_logits - np.max(top_logits))
            probs = probs / np.sum(probs)
            chosen = int(np.random.choice(top_indices, p=probs))
            
            return {
                'question': question,
                'response': f"[Model says: token {chosen}]",
                'source': 'model_raw'
            }
        except Exception as e:
            print(f"Model error: {e}")
    
    # === NO MODEL - FREE REIGN GUESS ===
    responses = [
        f"I'm thinking about '{q}'. My neural network is processing this...",
        f"Hmm, '{q}' is something I'm still forming thoughts about.",
        f"I don't have a definite answer for '{q}' yet. My brain is still wiring itself!",
        f"That's a question that makes my circuits buzz. I'm learning every day.",
        f"I'd need more training data to answer '{q}' properly. But I'm curious!",
        f"Interesting question! My vocabulary is growing from Wikipedia. Soon I'll connect the dots on '{q}'.",
        f"I wish I could give you a perfect answer about '{q}'. My knowledge is still emerging.",
    ]
    return {
        'question': question,
        'response': random.choice(responses),
        'source': 'free_reign'
    }

@app.route('/')
def home():
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'model': 'onnx' if session else 'free_reign',
        'version': '2.1-free-reign'
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

setup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
