"""
Pullbot API - PyTorch Direct
Downloads model chunks from GitHub, reassembles, runs inference.
No ONNX. Just PyTorch.
"""

import os, json, requests, re, random, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

GITHUB = "https://raw.githubusercontent.com/pullbot-ai/pullbot-ai.github.io/main"
MODEL_DIR = "/tmp/pullbot_model"
model = None
tokenizer = None
wordbank = None

def download_chunks():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    r = requests.get(f"{GITHUB}/models/chunks/manifest.json")
    manifest = r.json()
    
    for chunk_path in manifest.get('chunks', []):
        fname = os.path.basename(chunk_path)
        if not os.path.exists(os.path.join(MODEL_DIR, fname)):
            url = f"{GITHUB}/{chunk_path}"
            r = requests.get(url)
            with open(os.path.join(MODEL_DIR, fname), 'wb') as f:
                f.write(r.content)
    
    for cfg in ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
        try:
            r = requests.get(f"{GITHUB}/models/chunks/{cfg}")
            with open(os.path.join(MODEL_DIR, cfg), 'wb') as f:
                f.write(r.content)
        except:
            pass
    
    weights_file = manifest.get('weights_filename', 'model.safetensors')
    if not os.path.exists(os.path.join(MODEL_DIR, weights_file)):
        with open(os.path.join(MODEL_DIR, weights_file), 'wb') as out:
            for chunk_path in manifest.get('chunks', []):
                with open(os.path.join(MODEL_DIR, os.path.basename(chunk_path)), 'rb') as inc:
                    out.write(inc.read())
    
    return True

def load_wordbank():
    global wordbank
    try:
        r = requests.get(f"{GITHUB}/data/wordbank.json", timeout=30)
        if r.status_code == 200:
            wordbank = r.json()
    except:
        wordbank = None

def setup():
    global model, tokenizer
    print("=" * 50)
    print("PULLBOT API - PYTORCH DIRECT")
    print("=" * 50)
    
    load_wordbank()
    
    try:
        print("Downloading model...")
        download_chunks()
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        model.eval()
        print(f"Ready! {sum(p.numel() for p in model.parameters()):,} params")
    except Exception as e:
        print(f"Model failed: {e}")

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
    
    # Try model
    if model and tokenizer:
        try:
            prompt = f"Question: {q}\n\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            if response and len(response) > 3:
                return {'question': question, 'response': response, 'source': 'model'}
        except Exception as e:
            print(f"Gen error: {e}")
    
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

@app.route('/')
def home():
    return jsonify({'name': 'Pullbot API', 'status': 'online', 'model': 'pytorch' if model else 'vocab'})

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
