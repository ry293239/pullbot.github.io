"""
Pullbot API - GGUF Mode
Uses llama.cpp to run model with low RAM.
"""

import os, json, requests, re, random
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

GITHUB = "https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main"
MODEL_PATH = "/tmp/pullbot.gguf"
llm = None
wordbank = None

def download_model():
    url = f"{GITHUB}/models/pullbot.gguf"
    r = requests.get(url)
    if r.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            f.write(r.content)
        return True
    return False

def load_wordbank():
    global wordbank
    try:
        r = requests.get(f"{GITHUB}/data/wordbank.json", timeout=30)
        if r.status_code == 200:
            wordbank = r.json()
    except:
        wordbank = None

def setup():
    global llm
    print("PULLBOT API - GGUF MODE")
    load_wordbank()
    
    if download_model():
        print(f"Loading GGUF model ({os.path.getsize(MODEL_PATH)/(1024*1024):.0f}MB)...")
        llm = Llama(model_path=MODEL_PATH, n_ctx=256, n_threads=2)
        print("Ready!")
    else:
        print("No GGUF model. Vocab-only mode.")

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
    
    # Try GGUF model
    if llm:
        try:
            output = llm(f"Question: {q}\n\nAnswer:", max_tokens=60, temperature=0.8)
            response = output['choices'][0]['text'].strip()
            if response and len(response) > 3:
                return {'question': question, 'response': response, 'source': 'model'}
        except Exception as e:
            print(f"Model error: {e}")
    
    # Vocab fallback
    if wordbank:
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q.lower()))
        found = []
        for word in q_words:
            if word in wordbank.get('words', {}):
                info = wordbank['words'][word]
                if isinstance(info, dict) and info.get('has_definition'):
                    found.append(f"{word}: {info['definition']}")
        if found:
            return {'question': question, 'response': ' | '.join(found[:5]), 'source': 'vocab'}
    
    return {'question': question, 'response': "I'm learning...", 'source': 'fallback'}

@app.route('/')
def home():
    return jsonify({'name': 'Pullbot API', 'status': 'online', 'model': 'gguf' if llm else 'vocab'})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/ask')
def ask():
    q = request.args.get('q', '')
    if not q: return jsonify({'error': 'No question'}), 400
    return jsonify(generate_response(q))

setup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
