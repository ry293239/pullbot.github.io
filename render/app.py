"""
Pullbot API - ONNX Runtime (Lightweight)
No PyTorch. No Transformers. Under 100MB install.
Smart fallback uses vocabulary to construct responses.
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
    print("PULLBOT API - ONNX + VOCAB")
    print("=" * 50)
    
    load_wordbank()
    
    if download_file(f"{GITHUB}/models/pullbot.onnx", MODEL_PATH):
        print("Loading ONNX model...")
        session = ort.InferenceSession(MODEL_PATH)
        print("Ready!")
    else:
        print("No ONNX model. Using vocabulary mode.")

def simple_tokenize(text, max_len=64):
    tokens = []
    for char in text[-max_len * 4:]:
        tokens.append(hash(char) % 50257)
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(0)
    return tokens

def find_word_info(query):
    """Search wordbank for relevant words"""
    if not wordbank:
        return []
    
    q_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    results = []
    
    for word in q_words:
        if word in wordbank.get('words', {}):
            info = wordbank['words'][word]
            if info.get('has_definition'):
                results.append((word, info['definition']))
    
    return results[:5]

def generate_response(question):
    q = question.strip()
    
    # Math
    math_match = re.search(r'(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)', q)
    if math_match:
        a, op, b = float(math_match.group(1)), math_match.group(2), float(math_match.group(3))
        try:
            if op == '+': result = a + b
            elif op == '-': result = a - b
            elif op == '*': result = a * b
            elif op == '/': result = a / b if b != 0 else 'undefined'
            if isinstance(result, float) and result == int(result): result = int(result)
            return {'question': question, 'response': f"{a} {op} {b} = {result}", 'source': 'math'}
        except: pass
    
    # Try ONNX
    if session:
        try:
            tokens = simple_tokenize(f"Question: {q}\nAnswer:")
            input_ids = np.array([tokens[-64:]], dtype=np.int64)
            mask = np.ones((1, 64), dtype=np.int64)
            outputs = session.run(None, {'input_ids': input_ids, 'attention_mask': mask})
            return {'question': question, 'response': "Model is thinking...", 'source': 'onnx'}
        except: pass
    
    # Vocabulary search
    word_info = find_word_info(q)
    if word_info:
        words_str = '. '.join([f"{w} is {d}" for w, d in word_info])
        return {'question': question, 'response': f"I know these words: {words_str}", 'source': 'vocab'}
    
    # Free reign
    attempts = [
        f"I'm thinking about '{q}'. My vocabulary is growing!",
        f"Hmm, let me process '{q}'...",
        f"That's interesting! I'm learning more every day.",
    ]
    return {'question': question, 'response': random.choice(attempts), 'source': 'free_reign'}

@app.route('/')
def home():
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'model': 'onnx' if session else 'vocab',
        'words': wordbank.get('total_words', 0) if wordbank else 0
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/ask')
def ask():
    q = request.args.get('q', '')
    if not q:
        return jsonify({'error': 'No question'}), 400
    return jsonify(generate_response(q))

setup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
