"""
Pullbot Model
Loads DistilGPT2 from repo chunks — no internet needed.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import yaml
import json
import time
import glob

config_path = os.path.join(REPO_ROOT, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class PullbotModel:
    def __init__(self):
        self.device = "cpu"
        self.chunks_dir = os.path.join(REPO_ROOT, "models", "chunks")
        
        print("🤖 Loading Pullbot from repo chunks...")
        
        # Reassemble model if needed
        self._reassemble_model()
        
        # Load tokenizer from chunks dir
        print("📂 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.chunks_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model from chunks dir (config files are there)
        print("📂 Loading model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.chunks_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Apply LoRA for training
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"]
        )
        
        self.model = get_peft_model(self.base_model, self.lora_config)
        
        # Load trained adapter if exists
        adapter_path = os.path.join(REPO_ROOT, "models", "adapter")
        if os.path.exists(os.path.join(adapter_path, "adapter_model.bin")):
            print("📂 Loading trained adapter...")
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model ready! Trainable: {trainable:,} / Total: {total:,}")
    
    def _reassemble_model(self):
        """Reassemble chunks into a single safetensors file"""
        manifest_path = os.path.join(self.chunks_dir, "manifest.json")
        
        if not os.path.exists(manifest_path):
            print("❌ No manifest found in models/chunks/")
            return
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Check if already reassembled
        weights_file = manifest['weights_filename']
        reassembled_path = os.path.join(self.chunks_dir, weights_file)
        
        if os.path.exists(reassembled_path):
            expected_size = manifest['total_size']
            actual_size = os.path.getsize(reassembled_path)
            if actual_size == expected_size:
                print(f"  ✅ Model already reassembled ({actual_size//1024//1024}MB)")
                return
        
        print(f"  🧩 Reassembling model from {manifest['num_chunks']} chunks...")
        
        with open(reassembled_path, 'wb') as outfile:
            for chunk_rel_path in manifest['chunks']:
                chunk_path = os.path.join(REPO_ROOT, chunk_rel_path)
                with open(chunk_path, 'rb') as infile:
                    outfile.write(infile.read())
        
        print(f"  ✅ Reassembled {manifest['total_size_mb']}MB model")
    
    def save_adapter(self):
        adapter_path = os.path.join(REPO_ROOT, "models", "adapter")
        os.makedirs(adapter_path, exist_ok=True)
        self.model.save_pretrained(adapter_path)
        print(f"💾 Adapter saved")
    
    def prepare_training_data(self):
        corpus_path = os.path.join(REPO_ROOT, "data", "processed", "corpus.txt")
        if not os.path.exists(corpus_path):
            print("❌ No corpus found! Run scrape first.")
            return None
        
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        max_length = config['model']['max_length']
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length//2)]
        chunks = chunks[:config['training']['max_examples']]
        
        print(f"📝 Preparing {len(chunks)} training chunks")
        
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
        dataset = self.prepare_training_data()
        if dataset is None or len(dataset) == 0:
            print("❌ No training data")
            return
        
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
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        print(f"🚀 Training on {len(dataset)} chunks...")
        start = time.time()
        trainer.train()
        self.save_adapter()
        
        elapsed = (time.time() - start) / 60
        print(f"✅ Training complete in {elapsed:.1f} min")
        
        metrics_path = os.path.join(REPO_ROOT, "models", "last_train.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'dataset_size': len(dataset),
                'minutes': elapsed
            }, f)
    
    def generate(self, prompt, context=None):
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"User: {prompt}\n\nPullbot:"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=config['model']['max_length'],
                temperature=config['model']['temperature'],
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
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
        if sys.argv[1] == 'train':
            bot.train()
        elif sys.argv[1] == 'generate':
            prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
            print(bot.generate(prompt))
    else:
        print("Usage: python model.py [train|generate 'prompt']")
