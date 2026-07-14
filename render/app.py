"""
Pullbot API - WHOLE VOCAB MODE
Dumps the entire wordbank into the prompt. No filtering. No sorting.
Just AI + everything we know + hope.
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
all_vocab_text = ""

def download_file(url, dest):
    r = requests.get(url)
    if r.status_code == 200:
        with open(dest, 'wb') as f:
            f.write(r.content)
        return True
    return False

def load_wordbank():
    global all_vocab_text
    try:
        r = requests.get(f"{GITHUB}/data/wordbank.json", timeout=30)
        if r.status_code == 200:
            wordbank = r.json()
            # DUMP EVERYTHING into one big string
            parts = []
            for word, info in wordbank.get('words', {}).items():
                if isinstance(info, dict) and info.get('has_definition'):
                    parts.append(f"{word}={info['definition']}")
            all_vocab_text = " | ".join(parts)
            print(f"Loaded {len(parts)} definitions ({len(all_vocab_text)} chars)")
    except Exception as e:
        print(f"Wordbank failed: {e}")
        all_vocab_text = ""

def setup():
    global session
    print("=" * 50)
    print("PULLBOT API - WHOLE VOCAB MODE")
    print("=" * 50)
    
    load_wordbank()
    
    if download_file(f"{GITHUB}/models/pullbot.onnx", MODEL_PATH):
        print("Loading ONNX model...")
        session = ort.InferenceSession(MODEL_PATH)
        print("Ready!")
    else:
        print("No ONNX model.")

def simple_tokenize(text, max_len=512):
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
        a, op, b = float(math_match.group(1)), math_match.group(2), float(math_match.group(3))
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
    
    # === DUMP WHOLE VOCAB INTO PROMPT ===
    if session and all_vocab_text:
        try:
            prompt = f"Vocab: {all_vocab_text}\n\nQuestion: {q}\n\nAnswer:"
            tokens = simple_tokenize(prompt, max_len=512)
            generated = list(tokens)
            
            for _ in range(40):
                ids = generated[-512:]
                while len(ids) < 512:
                    ids = [0] + ids
                
                input_arr = np.array([ids[-512:]], dtype=np.int64)
                mask = np.ones((1, 512), dtype=np.int64)
                
                outputs = session.run(None, {
                    'input_ids': input_arr,
                    'attention_mask': mask
                })
                
                logits = outputs[0][0, -1, :]
                top_k = 50
                top_indices = np.argpartition(logits, -top_k)[-top_k:]
                top_logits = logits[top_indices]
                probs = np.exp(top_logits - np.max(top_logits))
                probs = probs / np.sum(probs)
                
                next_token = int(np.random.choice(top_indices, p=probs))
                generated.append(next_token)
            
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
    
    # === NOTHING WORKED ===
    return {
        'question': question,
        'response': random.choice([
            "I'm thinking...",
            "Let me process...",
            "Hmm...",
        ]),
        'source': 'thinking'
    }

@app.route('/')
def home():
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'model': 'onnx' if session else 'none',
        'vocab_size': len(all_vocab_text)
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
