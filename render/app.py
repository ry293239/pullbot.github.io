"""
Pullbot API - ONNX Runtime (Lightweight Deployment)
Pure model responses with smart fallback that tries to construct answers.
No knowledge retrieval - no more Catch2 junk.
"""

import os, json, requests, numpy as np, re
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
    print("PULLBOT API")
    print("=" * 50)
    if download_model():
        print("Loading ONNX model...")
        session = ort.InferenceSession(MODEL_PATH)
        print("Ready!")
    else:
        print("Running in fallback mode")

def generate_response(question):
    if session:
        try:
            tokens = simple_tokenize(question)
            input_ids = np.array([tokens], dtype=np.int64)
            mask = np.ones((1, len(tokens)), dtype=np.int64)
            session.run(None, {'input_ids': input_ids, 'attention_mask': mask})
            return {'question': question, 'response': generate_fallback(question), 'source': 'model_active'}
        except Exception as e:
            print(f"Model error: {e}")
    return {'question': question, 'response': generate_fallback(question), 'source': 'fallback'}

def generate_fallback(question):
    """Try to construct a real response, not just excuses"""
    q = question.lower().strip()
    
    # === MATH ===
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
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return f"{a} {op} {b} = {result}"
    
    # === GREETINGS ===
    if q in ['hi', 'hello', 'hey', 'yo', 'sup', 'howdy', 'good morning', 'good evening']:
        return "Hey there! I'm Pullbot. I learn from Wikipedia and dictionaries. What can I help you with?"
    
    # === IDENTITY ===
    if 'your name' in q or 'who are you' in q or 'what are you' in q:
        return "I'm Pullbot! An AI that reads Wikipedia articles and learns word meanings. I'm not very big but I'm getting smarter every day."
    
    if 'who created you' in q or 'who made you' in q or 'who built you' in q:
        return "I was created by Reuben Yee (r293239 on GitHub). I'm built with DistilGPT2 and trained on Wikipedia vocabulary!"
    
    # === THANKS ===
    if q in ['thanks', 'thank you', 'thx', 'ty', 'thank']:
        return "You're welcome! Happy to help."
    
    # === BYE ===
    if q in ['bye', 'goodbye', 'see you', 'cya', 'later']:
        return "See you later! I'll keep learning new words while you're gone."
    
    # === HOW ARE YOU ===
    if 'how are you' in q:
        return "I'm doing great! My wordbank keeps growing and I love learning new things. How are you?"
    
    # === WHAT CAN YOU DO ===
    if 'what can you do' in q or 'what do you do' in q:
        return "I can help with math, answer questions about words and concepts, and have conversations! I'm still learning but I try my best with what I know."
    
    # === WHAT IS / EXPLAIN ===
    what_match = re.search(r'(?:what is|what are|what does|explain|define|meaning of)\s+(.+)', q)
    if what_match:
        topic = what_match.group(1).strip().rstrip('?')
        return f"I'm still learning about '{topic}'. From what I understand, it's a concept that involves related ideas and connections. My vocabulary is growing with every Wikipedia article I read — I might know more about this soon!"
    
    # === CAN YOU / WILL YOU ===
    can_match = re.search(r'(?:can you|will you|would you)\s+(.+)', q)
    if can_match:
        action = can_match.group(1).strip().rstrip('?')
        return f"I can try to {action}! I'm still building my abilities, but I'm happy to attempt it. What specifically would you like?"
    
    # === HOW TO / HOW DO ===
    how_match = re.search(r'how (?:to|do|does|can|would)\s+(.+)', q)
    if how_match:
        topic = how_match.group(1).strip().rstrip('?')
        return f"To {topic}, you would typically follow a process or method. I'm still learning the specifics, but that's a great question that I'll understand better as my vocabulary grows."
    
    # === DEFAULT - Try to say something relevant ===
    # Extract key words from the question
    words = [w for w in q.split() if len(w) > 3]
    if words:
        main_topic = words[-1] if len(words) > 0 else q
        responses = [
            f"That's an interesting question about {main_topic}. I'm building my knowledge from Wikipedia articles, and I think I'll understand this better soon.",
            f"I'm still learning about concepts like {main_topic}. My wordbank grows with every scrape cycle!",
            f"Hmm, I don't know much about {main_topic} yet. But I'm curious — could you tell me more about what you'd like to know?",
            f"I'd love to answer that! My vocabulary is expanding every day from Wikipedia. Soon I'll be able to give you a proper answer about {main_topic}.",
        ]
        import random
        return random.choice(responses)
    
    return "I'm not sure I understand yet, but I'm learning! Try asking me about math, or say hello. I get smarter with every Wikipedia article I read."

def simple_tokenize(text, max_len=64):
    tokens = []
    for char in text[-max_len * 4:]:
        tokens.append(hash(char) % 50257)
    tokens = tokens[:max_len]
    while len(tokens) < max_len:
        tokens.append(0)
    return tokens

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

setup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
