"""
Pullbot Model
Loads DistilGPT2 from chunked files in repo.
No internet needed after initial chunking.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import yaml
import os
import json
import time

# Fix path
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class PullbotModel:
    def __init__(self):
        self.device = "cpu"
        
        print("🤖 Loading Pullbot from repo chunks...")
        
        # Reassemble model from chunks
        self._reassemble_if_needed()
        
        # Load tokenizer from chunks dir
        self.tokenizer = AutoTokenizer.from_pretrained("models/chunks")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model config
        model_config = AutoConfig.from_pretrained("models/chunks")
        
        # Load model from reassembled file
        model_file = "models/chunks/model_reassembled.bin"
        if os.path.exists(model_file):
            print("📂 Loading from reassembled model...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "models/chunks",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        else:
            raise FileNotFoundError("Model chunks not found! Run chunk_model.py first.")
        
        # Apply LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['training']['lora_rank'] if 'lora_rank' in config.get('training', {}) else 8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn"]
        )
        
        self.model = get_peft_model(self.base_model, self.lora_config)
        
        # Load existing adapter if available
        if os.path.exists("models/adapter/adapter_model.bin"):
            print("📂 Loading trained adapter...")
            self.model = PeftModel.from_pretrained(self.base_model, "models/adapter")
        
        print(f"✅ Model ready! Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _reassemble_if_needed(self):
        """Reassemble chunks into full model file if not already done"""
        manifest_path = "models/chunks/manifest.json"
        reassembled = "models/chunks/model_reassembled.bin"
        
        if os.path.exists(reassembled):
            print("  ✅ Model already reassembled")
            return
        
        if not os.path.exists(manifest_path):
            print("  ⚠️ No manifest found, will try downloading...")
            return
        
        print("  🧩 Reassembling model from chunks...")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        with open(reassembled, 'wb') as outfile:
            for chunk_path in manifest['chunks']:
                with open(chunk_path, 'rb') as infile:
                    outfile.write(infile.read())
        
        print(f"  ✅ Reassembled {manifest['total_size']//1024//1024}MB model")
    
    def save_adapter(self, path="models/adapter"):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        print(f"💾 Adapter saved")
    
    def train(self):
        corpus_path = "data/processed/corpus.txt"
        if not os.path.exists(corpus_path):
            print("❌ No corpus! Run scrape first.")
            return
        
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        max_len = config['model']['max_length']
        chunks = [text[i:i+max_len] for i in range(0, len(text), max_len//2)]
        chunks = chunks[:config['training']['max_examples']]
        
        def tokenize_fn(examples):
            return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_len)
        
        dataset = Dataset.from_dict({'text': chunks})
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
        
        training_args = TrainingArguments(
            output_dir="models/checkpoints",
            num_train_epochs=config['training']['epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            save_steps=500,
            logging_steps=50,
            learning_rate=config['training']['learning_rate'],
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        print(f"🚀 Training on {len(dataset)} chunks...")
        trainer.train()
        self.save_adapter()
        print("✅ Training complete")
    
    def generate(self, prompt, context=None):
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"{config['personality']['intro']}\n\nUser: {prompt}\n\nPullbot:"
        
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
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()

if __name__ == '__main__':
    import sys
    bot = PullbotModel()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        bot.train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'generate':
        prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
        print(bot.generate(prompt))
