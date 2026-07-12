"""
Pullbot Model
Loads DistilGPT2, trains with LoRA, generates responses.
"""

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import yaml
import json
import time

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class PullbotModel:
    def __init__(self, model_name=None):
        self.model_name = model_name or config['model']['base']
        self.device = "cpu"
        
        print(f"🤖 Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"]
        )
        
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.print_trainable_parameters()
        
        if os.path.exists("models/adapter/adapter_model.bin"):
            print("📂 Loading trained adapter...")
            self.model = PeftModel.from_pretrained(self.base_model, "models/adapter")
        
        print(f"✅ Model ready")
    
    def save_adapter(self, path="models/adapter"):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        print(f"💾 Adapter saved to {path}")
    
    def prepare_training_data(self):
        corpus_path = "data/processed/corpus.txt"
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
            output_dir="models/checkpoints",
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
        
        with open('models/last_train.json', 'w') as f:
            json.dump({'timestamp': time.time(), 'dataset_size': len(dataset), 'minutes': elapsed}, f)
    
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
