"""
Pullbot API - Vocab Mode
No ONNX. No model. Just vocabulary responses.
Finds matching words and returns what we know.
"""

import os, json, requests, re, random
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GITHUB = "https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main"
wordbank = None

def load_wordbank():
    global wordbank
    try:
        r = requests.get(f"{GITHUB}/data/wordbank.json", timeout=30)
        if r.status_code == 200:
            wordbank = r.json()
            defined = sum(1 for w in wordbank.get('words', {}).values() if isinstance(w, dict) and w.get('has_definition'))
            print(f"Loaded {wordbank.get('total_words', 0)} words, {defined} defined")
    except Exception as e:
        print(f"Wordbank failed: {e}")
        wordbank = None

def setup():
    print("=" * 50)
    print("PULLBOT API - VOCAB MODE")
    print("=" * 50)
    load_wordbank()
    print("Ready!")

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
    
    # Find matching vocab
    if wordbank:
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q.lower()))
        found = []
        for word in q_words:
            if word in wordbank.get('words', {}):
                info = wordbank['words'][word]
                if isinstance(info, dict) and info.get('has_definition'):
                    found.append(f"{word}: {info['definition']}")
        
        if found:
            response = " | ".join(found[:5])
            return {'question': question, 'response': response, 'source': 'vocab'}
    
    return {
        'question': question,
        'response': random.choice([
            "I don't know that word yet. My vocabulary is growing every few minutes!",
            "Hmm, I haven't learned about that. Try another question!",
            "That's not in my wordbank yet. I'm scraping Wikipedia right now.",
        ]),
        'source': 'thinking'
    }

@app.route('/')
def home():
    defined = 0
    if wordbank:
        defined = sum(1 for w in wordbank.get('words', {}).values() if isinstance(w, dict) and w.get('has_definition'))
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'words': wordbank.get('total_words', 0) if wordbank else 0,
        'defined': defined
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
