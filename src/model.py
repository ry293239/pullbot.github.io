"""
Pullbot Model - FULL FINE-TUNING + PRUNING + QUANTIZATION
Trains, prunes, then quantizes to 8-bit for free hosting.
Final model ~50-80MB RAM.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import torch
import torch.nn.utils.prune as torch_prune
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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
        print("🤖 PULLBOT MODEL LOADER")
        print("=" * 50)
        
        self._reassemble_if_needed()
        
        print("\n📂 Loading tokenizer from chunks...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.chunks_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"📂 Loading model from chunks...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.chunks_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model ready! {total_params:,} parameters")
    
    def _reassemble_if_needed(self):
        manifest_path = os.path.join(self.chunks_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            print("   ⚠️ No manifest found")
            return
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        weights_file = manifest.get('weights_filename', 'model.safetensors')
        reassembled_path = os.path.join(self.chunks_dir, weights_file)
        
        if os.path.exists(reassembled_path):
            expected_size = manifest.get('total_size', 0)
            actual_size = os.path.getsize(reassembled_path)
            if actual_size == expected_size:
                print(f"   ✅ Model reassembled ({actual_size//1024//1024}MB)")
                return
        
        print(f"   🧩 Reassembling from {manifest.get('num_chunks', 0)} chunks...")
        with open(reassembled_path, 'wb') as outfile:
            for chunk_rel_path in manifest.get('chunks', []):
                chunk_path = os.path.join(REPO_ROOT, chunk_rel_path)
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as infile:
                        outfile.write(infile.read())
        print(f"   ✅ Reassembled")
    
    def prepare_training_data(self):
        corpus_path = os.path.join(REPO_ROOT, "data", "processed", "corpus.txt")
        if not os.path.exists(corpus_path):
            print("❌ No corpus found!")
            return None
        
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        if len(text) < 500:
            print(f"❌ Corpus too small ({len(text)} chars)")
            return None
        
        max_length = config['model']['max_length']
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length//2)]
        chunks = chunks[:config['training']['max_examples']]
        
        print(f"\n📝 Training: {len(chunks)} chunks from {len(text):,} chars")
        
        def tokenize_fn(examples):
            return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)
        
        dataset = Dataset.from_dict({'text': chunks})
        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
        return tokenized
    
    def train(self):
        dataset = self.prepare_training_data()
        if dataset is None or len(dataset) == 0:
            return False
        
        print("\n" + "=" * 50)
        print("🚀 FULL MODEL TRAINING")
        print(f"   Examples: {len(dataset)}")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")
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
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        start = time.time()
        trainer.train()
        print(f"\n✅ Trained in {(time.time()-start)/60:.1f} min")
        
        with open(os.path.join(REPO_ROOT, "models", "last_train.json"), 'w') as f:
            json.dump({'timestamp': time.time(), 'dataset_size': len(dataset)}, f)
        
        return True
    
    # ============================================
    # PRUNING
    # ============================================
    
    def prune_model(self, target_sparsity=0.5):
        print("\n" + "=" * 50)
        print(f"✂️ PRUNING (target: {target_sparsity*100:.0f}%)")
        print("=" * 50)
        
        total_removed = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                total_params += weight.numel()
                flat = weight.abs().flatten()
                k = int(target_sparsity * flat.numel())
                
                if k > 0 and k < flat.numel():
                    threshold = torch.kthvalue(flat, k).values
                    mask = weight.abs() > threshold
                    removed = (mask == False).sum().item()
                    total_removed += removed
                    module.weight.data = weight * mask.float()
        
        sparsity = (total_removed / total_params * 100) if total_params > 0 else 0
        print(f"   Removed: {total_removed:,} / {total_params:,} ({sparsity:.1f}%)")
        return sparsity
    
    # ============================================
    # QUANTIZATION
    # ============================================
    
    def quantize_model(self):
        """Convert to 8-bit integers for 4x smaller RAM usage"""
        print("\n" + "=" * 50)
        print("🔧 QUANTIZING TO 8-BIT")
        print("=" * 50)
        
        # Quantize Linear layers to int8
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Count size reduction
        total_bytes = 0
        for param in self.model.parameters():
            if param.dtype == torch.qint8:
                total_bytes += param.numel() * 1  # 1 byte per int8
            else:
                total_bytes += param.numel() * 4  # 4 bytes per float32
        
        size_mb = total_bytes / (1024 * 1024)
        print(f"   Quantized size: {size_mb:.1f}MB")
        print(f"   RAM savings: ~75% vs 32-bit")
        return size_mb
    
    # ============================================
    # SIZE ESTIMATE
    # ============================================
    
    def get_model_size_estimate(self):
        non_zero = sum((p != 0).sum().item() for p in self.model.parameters() if p.dim() >= 2)
        total = sum(p.numel() for p in self.model.parameters())
        
        # Check if quantized
        is_quantized = any(p.dtype == torch.qint8 for p in self.model.parameters())
        bytes_per_param = 1 if is_quantized else 4
        
        ram_mb = non_zero * bytes_per_param / (1024 * 1024) * 3  # 3x for activations
        
        return {
            'total_params': total,
            'non_zero': non_zero,
            'sparsity_pct': (1 - non_zero/total)*100 if total > 0 else 0,
            'quantized': is_quantized,
            'estimated_ram_mb': ram_mb
        }
    
    # ============================================
    # SAVE & CHUNK
    # ============================================
    
    def save_and_chunk(self):
        print("\n💾 Saving model...")
        
        temp_dir = os.path.join(REPO_ROOT, "models", "temp_trained")
        os.makedirs(temp_dir, exist_ok=True)
        
        self.model.save_pretrained(temp_dir)
        self.tokenizer.save_pretrained(temp_dir)
        
        # Find weights
        weights_path = None
        for fname in ['model.safetensors', 'pytorch_model.bin']:
            path = os.path.join(temp_dir, fname)
            if os.path.exists(path):
                weights_path = path
                break
        
        if not weights_path:
            print("❌ No weights found!")
            return False
        
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"   File: {file_size_mb:.1f}MB")
        
        # Clear old chunks
        for old in glob.glob(os.path.join(self.chunks_dir, "model_chunk_*.bin")):
            os.remove(old)
        for old in glob.glob(os.path.join(self.chunks_dir, "*.safetensors")):
            os.remove(old)
        for old in glob.glob(os.path.join(self.chunks_dir, "pytorch_model.bin")):
            os.remove(old)
        
        # Split
        with open(weights_path, 'rb') as f:
            data = f.read()
        
        total_size = len(data)
        chunks = []
        for i in range(0, total_size, CHUNK_SIZE):
            chunk_data = data[i:i + CHUNK_SIZE]
            chunk_name = f"model_chunk_{i // CHUNK_SIZE:03d}.bin"
            chunk_path = os.path.join(self.chunks_dir, chunk_name)
            with open(chunk_path, 'wb') as f:
                f.write(chunk_data)
            chunks.append(f"models/chunks/{chunk_name}")
            print(f"   ✅ {chunk_name} ({len(chunk_data)/(1024*1024):.1f}MB)")
        
        # Copy configs
        for cfg in ['config.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                     'special_tokens_map.json', 'tokenizer.json']:
            src = os.path.join(temp_dir, cfg)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(self.chunks_dir, cfg))
        
        # Manifest
        stats = self.get_model_size_estimate()
        manifest = {
            "model_name": self.model_name,
            "total_size": total_size,
            "total_size_mb": round(total_size/(1024*1024), 1),
            "num_chunks": len(chunks),
            "chunks": chunks,
            "weights_filename": os.path.basename(weights_path),
            "fully_trained": True,
            "pruned": stats['sparsity_pct'] > 5,
            "quantized": stats['quantized'],
            "sparsity_pct": stats['sparsity_pct'],
            "estimated_ram_mb": stats['estimated_ram_mb'],
            "training_date": time.time()
        }
        with open(os.path.join(self.chunks_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        
        shutil.rmtree(temp_dir)
        if os.path.exists(os.path.join(REPO_ROOT, "models", "checkpoints")):
            shutil.rmtree(os.path.join(REPO_ROOT, "models", "checkpoints"))
        
        print(f"\n✅ Saved! {len(chunks)} chunks, {round(total_size/(1024*1024),1)}MB")
        print(f"   Sparsity: {stats['sparsity_pct']:.1f}%")
        print(f"   Quantized: {stats['quantized']}")
        print(f"   Est. RAM: {stats['estimated_ram_mb']:.0f}MB")
        return True

if __name__ == '__main__':
    import sys
    bot = PullbotModel()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            if bot.train():
                print("\n📊 Pre-optimization:")
                before = bot.get_model_size_estimate()
                print(f"   RAM: {before['estimated_ram_mb']:.0f}MB")
                
                # Prune
                sparsity = 0.5
                while True:
                    bot.prune_model(target_sparsity=sparsity)
                    after = bot.get_model_size_estimate()
                    if after['estimated_ram_mb'] < 400 or sparsity >= 0.9:
                        break
                    sparsity += 0.1
                
                # Quantize
                bot.quantize_model()
                
                final = bot.get_model_size_estimate()
                print(f"\n📊 Final: {final['estimated_ram_mb']:.0f}MB RAM")
                bot.save_and_chunk()
        
        elif command == 'prune':
            sparsity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
            bot.prune_model(target_sparsity=sparsity)
            bot.save_and_chunk()
        
        elif command == 'quantize':
            bot.quantize_model()
            bot.save_and_chunk()
        
        elif command == 'optimize':
            """Full optimization: prune + quantize"""
            sparsity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
            bot.prune_model(target_sparsity=sparsity)
            bot.quantize_model()
            stats = bot.get_model_size_estimate()
            print(f"\n📊 Optimized: {stats['estimated_ram_mb']:.0f}MB RAM")
            bot.save_and_chunk()
        
        elif command == 'size':
            stats = bot.get_model_size_estimate()
            print(f"\n📊 MODEL STATS")
            print(f"   Params: {stats['total_params']:,}")
            print(f"   Non-zero: {stats['non_zero']:,}")
            print(f"   Sparsity: {stats['sparsity_pct']:.1f}%")
            print(f"   Quantized: {stats['quantized']}")
            print(f"   Est. RAM: {stats['estimated_ram_mb']:.0f}MB")
        
        elif command == 'chunk':
            bot.save_and_chunk()
        
        else:
            print("Commands: train, prune [0.5-0.9], quantize, optimize [0.5-0.9], size, chunk")
    else:
        print("Commands: train, prune [0.5-0.9], quantize, optimize [0.5-0.9], size, chunk")
