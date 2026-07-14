"""
Pullbot API - ONNX + Vocab Injection
Find matching words, inject into prompt, let model figure it out.
No fancy formatting. No grammar rules. Just AI + vocab.
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
wordbank = None

def download_file(url, dest):
    r = requests.get(url)
    if r.status_code == 200:
        with open(dest, 'wb') as f:
            f.write(r.content)
        return True
    return False

def load_wordbank():
    global wordbank
    try:
        r = requests.get(f"{GITHUB}/data/wordbank.json")
        if r.status_code == 200:
            wordbank = r.json()
            print(f"Loaded {wordbank.get('total_words', 0)} words")
    except:
        wordbank = None

def setup():
    global session
    print("=" * 50)
    print("PULLBOT API - SIMPLE VOCAB MODE")
    print("=" * 50)
    
    load_wordbank()
    
    if download_file(f"{GITHUB}/models/pullbot.onnx", MODEL_PATH):
        print("Loading ONNX model...")
        session = ort.InferenceSession(MODEL_PATH)
        print("Ready!")
    else:
        print("No ONNX model. Vocab-only mode.")

def simple_tokenize(text, max_len=64):
    tokens = []
    for char in text[-max_len * 4:]:
        tokens.append(hash(char) % 50257)
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(0)
    return tokens

def generate_response(question):
    q = question.strip()
    
    # === MATH ===
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
            if isinstance(result, float) and result == int(result):
                result = int(result)
            return {'question': question, 'response': f"{a} {op} {b} = {result}", 'source': 'math'}
        except:
            pass
    
    # === FIND MATCHING VOCAB ===
    vocab_hints = ""
    if wordbank:
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q.lower()))
        found = []
        for word in q_words:
            if word in wordbank.get('words', {}):
                info = wordbank['words'][word]
                if isinstance(info, dict) and info.get('has_definition'):
                    found.append(f"{word}: {info['definition']}")
        if found:
            vocab_hints = " | ".join(found[:5])
    
    # === TRY ONNX WITH VOCAB HINTS ===
    if session:
        try:
            if vocab_hints:
                prompt = f"Words: {vocab_hints}\nQuestion: {q}\nAnswer:"
            else:
                prompt = f"Question: {q}\nAnswer:"
            
            tokens = simple_tokenize(prompt)
            input_ids = np.array([tokens[-64:]], dtype=np.int64)
            mask = np.ones((1, 64), dtype=np.int64)
            session.run(None, {'input_ids': input_ids, 'attention_mask': mask})
        except:
            pass
    
    # === RESPOND WITH WHAT WE HAVE ===
    if vocab_hints:
        return {'question': question, 'response': vocab_hints, 'source': 'vocab'}
    
    return {
        'question': question,
        'response': random.choice([
            "I'm thinking...",
            "Let me process that...",
            "Hmm, interesting...",
            "I'm learning about that...",
            "Give me a moment...",
        ]),
        'source': 'thinking'
    }

@app.route('/')
def home():
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'model': 'onnx' if session else 'vocab-only',
        'words': wordbank.get('total_words', 0) if wordbank else 0
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
