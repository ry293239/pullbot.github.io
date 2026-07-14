"""
Pullbot API - ONNX + Vocab Injection
Finds matching words, injects into prompt, generates real text token by token.
No fancy formatting. Just AI + vocab + actual generation.
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
    print("PULLBOT API - REAL GENERATION MODE")
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
    
    # === ACTUAL MODEL GENERATION ===
    if session and vocab_hints:
        try:
            prompt = f"Words: {vocab_hints}\nQuestion: {q}\nAnswer:"
            tokens = simple_tokenize(prompt, max_len=64)
            generated = list(tokens)
            
            # Generate token by token
            for _ in range(30):
                ids = generated[-64:]
                while len(ids) < 64:
                    ids = [0] + ids
                
                input_arr = np.array([ids[-64:]], dtype=np.int64)
                mask = np.ones((1, 64), dtype=np.int64)
                
                outputs = session.run(None, {
                    'input_ids': input_arr,
                    'attention_mask': mask
                })
                
                logits = outputs[0][0, -1, :]
                
                top_k = 40
                top_indices = np.argpartition(logits, -top_k)[-top_k:]
                top_logits = logits[top_indices]
                probs = np.exp(top_logits - np.max(top_logits))
                probs = probs / np.sum(probs)
                
                next_token = int(np.random.choice(top_indices, p=probs))
                generated.append(next_token)
            
            # Decode new tokens
            new_tokens = generated[len(tokens):]
            chars = []
            for t in new_tokens:
                c = t % 256
                if 32 <= c < 127:
                    chars.append(chr(c))
                elif c == 10:
                    chars.append('\n')
            
            response = ''.join(chars).strip()
            
            if len(response) > 5:
                return {'question': question, 'response': response, 'source': 'model'}
        except Exception as e:
            print(f"Model error: {e}")
    
    # === VOCAB FALLBACK ===
    if vocab_hints:
        return {'question': question, 'response': vocab_hints, 'source': 'vocab'}
    
    # === HONEST THINKING ===
    return {
        'question': question,
        'response': random.choice([
            "Hmm, let me think... I don't have enough words yet to answer properly.",
            "I'm searching my vocabulary... I know some words but can't connect them yet.",
            "That's a good question. My wordbank is growing every few minutes!",
            "I need more words to answer that. Try asking about something simpler.",
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
