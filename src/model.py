"""
Pullbot Model
Loads DistilGPT2, trains with LoRA, generates responses.
All in one file for simplicity.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset
import yaml
import os
import json
import time
import glob

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class PullbotModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or config['model']['base']
        self.device = "cpu"  # GitHub Actions is CPU only
        
        print(f"🤖 Loading base model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # CPU needs float32
            low_cpu_mem_usage=True
        )
        
        # Apply LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['model']['lora_rank'],
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"]  # DistilGPT2 attention layers
        )
        
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()
        
        # Load existing adapter if available
        self.load_adapter()
    
    def load_adapter(self, path="models/adapter"):
        """Load trained LoRA adapter if exists"""
        if os.path.exists(f"{path}/adapter_model.bin"):
            print("📂 Loading existing adapter...")
            self.model = PeftModel.from_pretrained(
                self.base_model, 
                path
            )
            print("  ✅ Adapter loaded")
        else:
            print("  🆕 Starting fresh (no adapter found)")
    
    def save_adapter(self, path="models/adapter"):
        """Save LoRA adapter"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        print(f"💾 Adapter saved to {path}")
    
    def prepare_training_data(self, corpus_path="data/processed/corpus.txt"):
        """Convert text corpus to training dataset"""
        if not os.path.exists(corpus_path):
            print("❌ No corpus found! Run scrape first.")
            return None
        
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        # Split into chunks
        max_length = config['model']['max_length']
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length//2)]
        chunks = chunks[:config['training']['max_examples']]
        
        print(f"📝 Preparing {len(chunks)} training chunks")
        
        # Tokenize
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
    
    def train(self, dataset=None):
        """Train the model"""
        if dataset is None:
            dataset = self.prepare_training_data()
        
        if dataset is None or len(dataset) == 0:
            print("❌ No training data")
            return
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="models/checkpoints",
            num_train_epochs=1,
            per_device_train_batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=4,
            save_steps=config['model']['save_every'],
            logging_steps=50,
            learning_rate=config['training']['learning_rate'],
            warmup_steps=100,
            save_total_limit=2,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # CPU training
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, 
                mlm=False
            ),
        )
        
        print(f"🚀 Starting training...")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Time limit: {config['training']['chunk_minutes']} minutes")
        
        # Train with timeout
        start_time = time.time()
        timeout = config['training']['chunk_minutes'] * 60
        
        trainer.train()
        
        # Save adapter
        self.save_adapter()
        
        elapsed = (time.time() - start_time) / 60
        print(f"✅ Training complete in {elapsed:.1f} minutes")
        
        # Save training metrics
        metrics = {
            'timestamp': time.time(),
            'dataset_size': len(dataset),
            'training_time_minutes': elapsed
        }
        
        os.makedirs('models', exist_ok=True)
        with open('models/last_train.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def generate(self, prompt, context=None, max_length=None):
        """Generate a response"""
        if max_length is None:
            max_length = config['model']['max_length']
        
        # Build full prompt with context
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"{config['personality']['system_prompt']}\n\nUser: {prompt}\n\nPullbot:"
        
        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_length,
                temperature=config['model']['temperature'],
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()

# Simple CLI for testing
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = 'train'
    
    bot = PullbotModel()
    
    if command == 'train':
        bot.train()
    elif command == 'generate':
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello!"
        response = bot.generate(prompt)
        print(f"\n🤖 Pullbot: {response}")
    else:
        print("Commands: train, generate 'your prompt'")
