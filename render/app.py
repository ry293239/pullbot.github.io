"""
Pullbot API - PyTorch Direct
Loads model chunks, generates real text. No more token IDs.
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

def download_chunks():
    """Download model chunks from GitHub"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Download manifest
    r = requests.get(f"{GITHUB}/models/chunks/manifest.json")
    manifest = r.json()
    
    # Download chunks
    for chunk_path in manifest.get('chunks', []):
        fname = os.path.basename(chunk_path)
        url = f"{GITHUB}/{chunk_path}"
        r = requests.get(url)
        with open(os.path.join(MODEL_DIR, fname), 'wb') as f:
            f.write(r.content)
    
    # Download config files
    for cfg in ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']:
        try:
            r = requests.get(f"{GITHUB}/models/chunks/{cfg}")
            with open(os.path.join(MODEL_DIR, cfg), 'wb') as f:
                f.write(r.content)
        except:
            pass
    
    # Reassemble
    weights_file = manifest.get('weights_filename', 'model.safetensors')
    with open(os.path.join(MODEL_DIR, weights_file), 'wb') as out:
        for chunk_path in manifest.get('chunks', []):
            with open(os.path.join(MODEL_DIR, os.path.basename(chunk_path)), 'rb') as inc:
                out.write(inc.read())
    
    return True

def setup():
    global model, tokenizer
    print("=" * 50)
    print("PULLBOT API - FREE REIGN MODE")
    print("=" * 50)
    
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
        print(f"Could not load model: {e}")
        print("Running in math-only mode")

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
    
    # === MODEL GENERATION ===
    if model and tokenizer:
        try:
            prompt = f"Question: {q}\n\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=60,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            if response and len(response) > 2:
                return {'question': question, 'response': response, 'source': 'model'}
        except Exception as e:
            print(f"Generation error: {e}")
    
    # === MODEL FAILED - TRY ANYWAY ===
    words = [w for w in q.split() if len(w) > 3]
    topic = words[-1] if words else q
    attempts = [
        f"I think {topic} relates to things I've been learning about.",
        f"Hmm, {topic}... My circuits are trying to connect the dots on this one.",
        f"I'm forming thoughts about {topic}. Give me a moment to process.",
        f"That thing about {topic} — I might have read something related in my training.",
        f"{topic.title()}... I feel like I almost know something about this.",
    ]
    return {'question': question, 'response': random.choice(attempts), 'source': 'trying'}

@app.route('/')
def home():
    return jsonify({
        'name': 'Pullbot API',
        'status': 'online',
        'model': 'pytorch' if model else 'math-only',
        'version': '3.0-free-reign'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/ask')
def ask():
    q = request.args.get('q', '')
    if not q:
        return jsonify({'error': 'No question'}), 400
    return jsonify(generate_response(q))

setup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 7860)))
