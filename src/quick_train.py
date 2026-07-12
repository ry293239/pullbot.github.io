"""Quick test train - tiny corpus, 1 epoch, ~3 minutes"""

import os, sys, json, time, glob, shutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

# Reassemble model if needed
with open('models/chunks/manifest.json') as f:
    manifest = json.load(f)

weights_file = manifest['weights_filename']
reassembled = f'models/chunks/{weights_file}'

if not os.path.exists(reassembled) or os.path.getsize(reassembled) != manifest['total_size']:
    print('Reassembling model from chunks...')
    with open(reassembled, 'wb') as out:
        for chunk_path in manifest['chunks']:
            with open(chunk_path, 'rb') as inc:
                out.write(inc.read())
    print('Done reassembling')

# Load model
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained('models/chunks')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    'models/chunks',
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# Tiny corpus - 10 examples
corpus = (
    "Hello world. Python is a programming language. "
    "AI is artificial intelligence. GitHub hosts code. "
    "Wikipedia has facts. Chess is a strategy game. "
    "Coding is creative. Learning is important. "
    "Data is valuable. Knowledge is power. "
    "The sun is a star. Water is essential for life. "
    "Gravity keeps us on Earth. The internet connects people. "
    "Music is a universal language. Books contain wisdom."
)

chunks = [corpus[i:i+256] for i in range(0, len(corpus), 128)]
chunks = [c for c in chunks if len(c) > 20][:10]

print(f'Training on {len(chunks)} chunks')

def tokenize_fn(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=256
    )

dataset = Dataset.from_dict({'text': chunks})
dataset = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])

# Train
args = TrainingArguments(
    output_dir='models/checkpoints',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_steps=1,
    learning_rate=5e-5,
    save_total_limit=1,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print('Training...')
trainer.train()
print('Training complete')

# Save and chunk
temp_dir = 'models/temp_test'
os.makedirs(temp_dir, exist_ok=True)
model.save_pretrained(temp_dir)
tokenizer.save_pretrained(temp_dir)

# Find weights
weights_path = None
for fname in ['model.safetensors', 'pytorch_model.bin']:
    path = os.path.join(temp_dir, fname)
    if os.path.exists(path):
        weights_path = path
        break

if weights_path:
    with open(weights_path, 'rb') as f:
        data = f.read()
    
    # Clear old chunks
    for old in glob.glob('models/chunks/model_chunk_*.bin'):
        os.remove(old)
    
    CHUNK_SIZE = 45 * 1024 * 1024
    chunks_list = []
    for i in range(0, len(data), CHUNK_SIZE):
        name = f'model_chunk_{i//CHUNK_SIZE:03d}.bin'
        with open(f'models/chunks/{name}', 'wb') as out:
            out.write(data[i:i+CHUNK_SIZE])
        chunks_list.append(f'models/chunks/{name}')
    
    # Copy config files
    for cfg in ['config.json', 'tokenizer_config.json', 'vocab.json',
                'merges.txt', 'special_tokens_map.json', 'tokenizer.json']:
        src = os.path.join(temp_dir, cfg)
        if os.path.exists(src):
            shutil.copy(src, f'models/chunks/{cfg}')
    
    # Update manifest
    manifest = {
        'model_name': 'distilgpt2',
        'total_size': len(data),
        'total_size_mb': round(len(data)/(1024*1024), 1),
        'num_chunks': len(chunks_list),
        'chunks': chunks_list,
        'weights_filename': os.path.basename(weights_path),
        'fully_trained': True,
        'quick_test': True,
        'training_date': time.time()
    }
    with open('models/chunks/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

shutil.rmtree(temp_dir)
print(f'Done! {len(chunks_list)} chunks saved')
