"""
Pullbot Model - FULL FINE-TUNING
Trains the entire DistilGPT2 model directly on scraped data.
After training, re-chunks the model and saves to models/chunks/.
The website downloads YOUR trained model - not generic DistilGPT2.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import yaml
import json
import time
import glob
import shutil

config_path = os.path.join(REPO_ROOT, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

CHUNK_SIZE_MB = 45
CHUNK_SIZE = CHUNK_SIZE_MB * 1024 * 1024

class PullbotModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or config['model']['base']
        self.device = "cpu"
        self.chunks_dir = os.path.join(REPO_ROOT, "models", "chunks")
        
        print("=" * 50)
        print("🤖 PULLBOT FULL MODEL LOADER")
        print("=" * 50)
        
        # Load tokenizer from chunks
        print("\n📂 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the full model (will use cached version from HuggingFace
        # since we need the full weights for training)
        print(f"📂 Loading {self.model_name} for training...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Check if we have a previously trained model in chunks
        manifest_path = os.path.join(self.chunks_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            if manifest.get('fully_trained'):
                print("📂 Found previously trained model - loading state...")
                self._load_trained_state()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Model ready! Params: {total_params:,} total, {trainable_params:,} trainable")
    
    def _load_trained_state(self):
        """Load previously trained weights from chunks"""
        reassembled = os.path.join(self.chunks_dir, "model.safetensors")
        if os.path.exists(reassembled):
            print("  Loading trained weights...")
            state_dict = torch.load(reassembled, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
            print("  ✅ Trained weights loaded")
    
    def prepare_training_data(self):
        """Convert corpus to training dataset"""
        corpus_path = os.path.join(REPO_ROOT, "data", "processed", "corpus.txt")
        if not os.path.exists(corpus_path):
            print("❌ No corpus found! Run scrape first.")
            return None
        
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        if len(text) < 500:
            print(f"❌ Corpus too small ({len(text)} chars). Need more data.")
            return None
        
        max_length = config['model']['max_length']
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length//2)]
        chunks = chunks[:config['training']['max_examples']]
        
        print(f"\n📝 Training data: {len(chunks)} chunks from {len(text):,} chars")
        
        def tokenize_fn(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length
            )
        
        dataset = Dataset.from_dict({'text': chunks})
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
        return tokenized
    
    def train(self):
        """Full model training"""
        dataset = self.prepare_training_data()
        if dataset is None or len(dataset) == 0:
            print("❌ Cannot train - no data")
            return False
        
        print("\n" + "=" * 50)
        print("🚀 FULL MODEL TRAINING")
        print("=" * 50)
        print(f"   Examples: {len(dataset)}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")
        print(f"   This modifies ALL {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print("=" * 50 + "\n")
        
        training_args = TrainingArguments(
            output_dir=os.path.join(REPO_ROOT, "models", "checkpoints"),
            num_train_epochs=config['training']['epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=4,
            save_steps=500,
            logging_steps=50,
            learning_rate=config['training']['learning_rate'],
            warmup_steps=50,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, 
                mlm=False
            ),
        )
        
        start = time.time()
        trainer.train()
        elapsed = (time.time() - start) / 60
        
        print(f"\n✅ Training complete in {elapsed:.1f} minutes")
        
        # Save metrics
        metrics_path = os.path.join(REPO_ROOT, "models", "last_train.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'dataset_size': len(dataset),
                'minutes': elapsed,
                'type': 'full_fine_tune',
                'parameters_trained': sum(p.numel() for p in self.model.parameters())
            }, f)
        
        return True
    
    def save_and_chunk(self):
        """Save trained model and split into chunks for repo"""
        print("\n💾 Saving trained model...")
        
        # Save full model to temp location
        temp_dir = os.path.join(REPO_ROOT, "models", "temp_trained")
        os.makedirs(temp_dir, exist_ok=True)
        
        self.model.save_pretrained(temp_dir)
        self.tokenizer.save_pretrained(temp_dir)
        
        # Find the saved weights file
        safetensors_path = os.path.join(temp_dir, "model.safetensors")
        pytorch_path = os.path.join(temp_dir, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            weights_path = safetensors_path
        elif os.path.exists(pytorch_path):
            weights_path = pytorch_path
        else:
            print("❌ Could not find saved weights!")
            return False
        
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"   Trained model size: {file_size_mb:.1f}MB")
        
        # Clear old chunks
        chunks_dir = self.chunks_dir
        for old_file in glob.glob(os.path.join(chunks_dir, "model_chunk_*.bin")):
            os.remove(old_file)
        
        # Split into new chunks
        print(f"\n🔪 Splitting into {CHUNK_SIZE_MB}MB chunks...")
        
        with open(weights_path, 'rb') as f:
            data = f.read()
        
        total_size = len(data)
        chunks = []
        
        for i in range(0, total_size, CHUNK_SIZE):
            chunk_data = data[i:i + CHUNK_SIZE]
            chunk_name = f"model_chunk_{i // CHUNK_SIZE:03d}.bin"
            chunk_path = os.path.join(chunks_dir, chunk_name)
            
            with open(chunk_path, 'wb') as f:
                f.write(chunk_data)
            
            chunk_size_mb = len(chunk_data) / (1024 * 1024)
            chunks.append(f"models/chunks/{chunk_name}")
            print(f"   ✅ {chunk_name} ({chunk_size_mb:.1f}MB)")
        
        # Also save the full file for local loading
        weights_filename = os.path.basename(weights_path)
        shutil.copy(weights_path, os.path.join(chunks_dir, weights_filename))
        
        # Save updated manifest
        manifest = {
            "model_name": self.model_name,
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 1),
            "num_chunks": len(chunks),
            "chunks": chunks,
            "weights_filename": weights_filename,
            "fully_trained": True,
            "training_date": time.time(),
            "corpus_chars": len(open(os.path.join(REPO_ROOT, "data", "processed", "corpus.txt")).read()) if os.path.exists(os.path.join(REPO_ROOT, "data", "processed", "corpus.txt")) else 0
        }
        
        manifest_path = os.path.join(chunks_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Copy config files
        config_files = ['config.json', 'tokenizer_config.json', 'vocab.json',
                        'merges.txt', 'special_tokens_map.json', 'tokenizer.json']
        for fname in config_files:
            src = os.path.join(temp_dir, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(chunks_dir, fname))
        
        # Cleanup temp
        shutil.rmtree(temp_dir)
        
        print(f"\n✅ Model chunked: {len(chunks)} files, {round(total_size/(1024*1024),1)}MB total")
        print(f"   This is now YOUR trained model, not generic DistilGPT2!")
        return True
    
    def generate(self, prompt, context=None, max_length=None):
        """Generate text from the trained model"""
        if max_length is None:
            max_length = config['model']['max_length']
        
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"User: {prompt}\n\nPullbot:"
        
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_length,
                temperature=config['model']['temperature'],
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response.strip()

if __name__ == '__main__':
    import sys
    
    bot = PullbotModel()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            success = bot.train()
            if success:
                bot.save_and_chunk()
                print("\n🎉 PULLBOT HAS BEEN TRAINED!")
                print("   The model in models/chunks/ is now YOUR AI.")
                print("   Push to repo and the website will download YOUR brain.")
        
        elif command == 'generate':
            prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
            response = bot.generate(prompt)
            print(f"\n🤖 Pullbot: {response}")
        
        elif command == 'chunk':
            bot.save_and_chunk()
        
        else:
            print("Commands: train, generate 'prompt', chunk")
    else:
        print("Commands: train, generate 'prompt', chunk")
