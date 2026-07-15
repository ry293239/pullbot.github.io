"""
Pullbot API - GGUF Mode
Loads model from local file, wordbank from local or GitHub.
"""

import os, json, requests, re, random, hashlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

GITHUB = "https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pullbot.gguf")
WORD_BANK_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wordbank.json")

llm = None
wordbank = None
_memory_users = {"users": {}}

def load_wordbank():
    global wordbank
    if os.path.exists(WORD_BANK_PATH):
        try:
            with open(WORD_BANK_PATH) as f:
                wordbank = json.load(f)
            defined = sum(1 for w in wordbank.get('words', {}).values() if isinstance(w, dict) and w.get('has_definition'))
            print(f"Loaded {wordbank.get('total_words', 0)} words, {defined} defined (local)")
            return
        except:
            pass
    
    try:
        r = requests.get(f"{GITHUB}/data/wordbank.json", timeout=30)
        if r.status_code == 200:
            wordbank = r.json()
            print(f"Loaded wordbank from GitHub")
    except:
        wordbank = None

def setup():
    global llm
    print("=" * 50)
    print("PULLBOT API - GGUF MODE")
    print("=" * 50)
    
    load_wordbank()
    
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024*1024)
        print(f"Loading GGUF model ({size_mb:.0f}MB)...")
        try:
            llm = Llama(model_path=MODEL_PATH, n_ctx=256, n_threads=2, verbose=False)
            print("Ready!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            llm = None
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Vocab-only mode.")

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
            output = llm(f"Question: {q}\n\nAnswer:", max_tokens=60, temperature=0.8, top_p=0.9)
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
    
    return {'question': question, 'response': "I'm still learning...", 'source': 'fallback'}

# ========== AUTH ==========

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json() or {}
    username = data.get('username', '').strip().lower()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'}), 400
    if len(username) < 3:
        return jsonify({'success': False, 'error': 'Username too short'}), 400
    if username in _memory_users.get('users', {}):
        return jsonify({'success': False, 'error': 'Username taken'}), 409
    
    _memory_users['users'][username] = {
        'password': hash_password(password),
        'created': __import__('time').strftime('%Y-%m-%d'),
        'chats': []
    }
    
    return jsonify({
        'success': True,
        'username': username,
        'token': hash_password(username + password)[:20],
        'message': f'Welcome, {username}!'
    })

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip().lower()
    password = data.get('password', '').strip()
    
    if username not in _memory_users.get('users', {}):
        return jsonify({'success': False, 'error': 'User not found'}), 404
    if _memory_users['users'][username]['password'] != hash_password(password):
        return jsonify({'success': False, 'error': 'Wrong password'}), 401
    
    return jsonify({
        'success': True,
        'username': username,
        'token': hash_password(username + password)[:20],
        'message': f'Welcome back, {username}!'
    })

# ========== API ==========

@app.route('/')
def home():
    defined = 0
    if wordbank:
        defined = sum(1 for w in wordbank.get('words', {}).values() if isinstance(w, dict) and w.get('has_definition'))
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'model': 'gguf' if llm else 'vocab',
        'words': wordbank.get('total_words', 0) if wordbank else 0,
        'defined': defined,
        'users': len(_memory_users.get('users', {}))
    })

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
