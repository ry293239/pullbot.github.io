"""
Pullbot Model - FULL FINE-TUNING + PRUNING
Trains DistilGPT2 on scraped data, then prunes unnecessary weights.
Smaller model = less RAM = fits on free hosting.
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
        
        # Reassemble model from chunks if needed
        self._reassemble_if_needed()
        
        # Load tokenizer from chunks
        print("\n📂 Loading tokenizer from chunks...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.chunks_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model from chunks
        print(f"📂 Loading model from chunks...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.chunks_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model ready! {total_params:,} parameters")
    
    def _reassemble_if_needed(self):
        """Reassemble model chunks into full file if needed"""
        manifest_path = os.path.join(self.chunks_dir, "manifest.json")
        
        if not os.path.exists(manifest_path):
            print("   ⚠️ No manifest found - model may be incomplete")
            return
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        weights_file = manifest.get('weights_filename', 'model.safetensors')
        reassembled_path = os.path.join(self.chunks_dir, weights_file)
        
        if os.path.exists(reassembled_path):
            expected_size = manifest.get('total_size', 0)
            actual_size = os.path.getsize(reassembled_path)
            if actual_size == expected_size:
                print(f"   ✅ Model already reassembled ({actual_size//1024//1024}MB)")
                return
        
        print(f"   🧩 Reassembling model from {manifest.get('num_chunks', 0)} chunks...")
        
        with open(reassembled_path, 'wb') as outfile:
            for chunk_rel_path in manifest.get('chunks', []):
                chunk_path = os.path.join(REPO_ROOT, chunk_rel_path)
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as infile:
                        outfile.write(infile.read())
                else:
                    print(f"   ⚠️ Missing chunk: {chunk_path}")
        
        final_size = os.path.getsize(reassembled_path)
        print(f"   ✅ Reassembled {final_size//1024//1024}MB")
    
    def prepare_training_data(self):
        corpus_path = os.path.join(REPO_ROOT, "data", "processed", "corpus.txt")
        if not os.path.exists(corpus_path):
            print("❌ No corpus found! Run scrape first.")
            return None
        
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        if len(text) < 500:
            print(f"❌ Corpus too small ({len(text)} chars). Need >500.")
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
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
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
        elapsed = (time.time() - start) / 60
        
        print(f"\n✅ Training complete in {elapsed:.1f} minutes")
        
        metrics_path = os.path.join(REPO_ROOT, "models", "last_train.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'dataset_size': len(dataset),
                'minutes': elapsed,
                'type': 'full_fine_tune',
                'parameters': sum(p.numel() for p in self.model.parameters())
            }, f)
        
        return True
    
    # ============================================
    # PRUNING
    # ============================================
    
    def prune_model(self, target_sparsity=0.5):
        """Prune weights close to zero across the entire model.
        target_sparsity: fraction of weights to remove (0.5 = 50%, 0.8 = 80%)"""
        
        print("\n" + "=" * 50)
        print(f"✂️ PRUNING MODEL (target: {target_sparsity*100:.0f}%)")
        print("=" * 50)
        
        total_removed = 0
        total_params = 0
        layers_pruned = 0
        
        for name, module in self.model.named_modules():
            # Only prune Linear layers (these have the most weights)
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                total_params += weight.numel()
                
                # Calculate threshold for this layer
                flat = weight.abs().flatten()
                k = int(target_sparsity * flat.numel())
                
                if k > 0 and k < flat.numel():
                    threshold = torch.kthvalue(flat, k).values
                    mask = weight.abs() > threshold
                    
                    # Count removed weights
                    removed = (mask == False).sum().item()
                    total_removed += removed
                    
                    # Apply mask
                    module.weight.data = weight * mask.float()
                    layers_pruned += 1
                    
                    if layers_pruned <= 5:  # Show first 5 layers
                        pct = removed / weight.numel() * 100
                        print(f"   {name}: removed {removed:,} weights ({pct:.1f}%)")
        
        # Also prune LayerNorm and other small layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                weight = module.weight.data
                if weight.numel() > 10:
                    flat = weight.abs().flatten()
                    k = int(target_sparsity * flat.numel())
                    if k > 0 and k < flat.numel():
                        threshold = torch.kthvalue(flat, k).values
                        mask = weight.abs() > threshold
                        module.weight.data = weight * mask.float()
                        total_removed += (mask == False).sum().item()
                        total_params += weight.numel()
        
        actual_sparsity = (total_removed / total_params * 100) if total_params > 0 else 0
        
        print(f"\n   Layers pruned: {layers_pruned}")
        print(f"   Weights removed: {total_removed:,} / {total_params:,}")
        print(f"   Actual sparsity: {actual_sparsity:.1f}%")
        
        return actual_sparsity
    
    def get_model_size_estimate(self):
        """Estimate RAM and storage needs"""
        total = sum(p.numel() for p in self.model.parameters())
        non_zero = sum((p != 0).sum().item() for p in self.model.parameters() if p.dim() >= 2)
        
        fp32_mb = total * 4 / (1024 * 1024)
        sparse_mb = non_zero * 4 / (1024 * 1024)
        ram_est = sparse_mb * 3  # Model + activations + overhead
        
        return {
            'total_params': total,
            'non_zero_params': non_zero,
            'sparsity_pct': (1 - non_zero/total)*100 if total > 0 else 0,
            'file_size_mb': fp32_mb,
            'sparse_size_mb': sparse_mb,
            'estimated_ram_mb': ram_est
        }
    
    # ============================================
    # SAVE & CHUNK
    # ============================================
    
    def save_and_chunk(self):
        """Save trained/pruned model and split into chunks"""
        print("\n💾 Saving model...")
        
        temp_dir = os.path.join(REPO_ROOT, "models", "temp_trained")
        os.makedirs(temp_dir, exist_ok=True)
        
        self.model.save_pretrained(temp_dir)
        self.tokenizer.save_pretrained(temp_dir)
        
        # Find saved weights
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
        print(f"   Model file: {file_size_mb:.1f}MB")
        
        # Clear old chunks
        for old_file in glob.glob(os.path.join(self.chunks_dir, "model_chunk_*.bin")):
            os.remove(old_file)
        for old_file in glob.glob(os.path.join(self.chunks_dir, "*.safetensors")):
            os.remove(old_file)
        for old_file in glob.glob(os.path.join(self.chunks_dir, "pytorch_model.bin")):
            os.remove(old_file)
        
        # Split into chunks
        print(f"\n🔪 Splitting into {CHUNK_SIZE_MB}MB chunks...")
        
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
            
            chunk_size_mb = len(chunk_data) / (1024 * 1024)
            chunks.append(f"models/chunks/{chunk_name}")
            print(f"   ✅ {chunk_name} ({chunk_size_mb:.1f}MB)")
        
        # Copy config files
        config_files = ['config.json', 'tokenizer_config.json', 'vocab.json',
                        'merges.txt', 'special_tokens_map.json', 'tokenizer.json']
        for fname in config_files:
            src = os.path.join(temp_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(self.chunks_dir, fname)
                with open(src, 'rb') as fin:
                    with open(dst, 'wb') as fout:
                        fout.write(fin.read())
        
        # Get model stats
        stats = self.get_model_size_estimate()
        
        # Update manifest
        manifest = {
            "model_name": self.model_name,
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 1),
            "num_chunks": len(chunks),
            "chunks": chunks,
            "weights_filename": os.path.basename(weights_path),
            "fully_trained": True,
            "pruned": stats['sparsity_pct'] > 5,
            "sparsity_pct": stats['sparsity_pct'],
            "estimated_ram_mb": stats['estimated_ram_mb'],
            "training_date": time.time()
        }
        
        with open(os.path.join(self.chunks_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        # Remove checkpoint files
        ckpt_dir = os.path.join(REPO_ROOT, "models", "checkpoints")
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        
        print(f"\n✅ Chunked: {len(chunks)} files, {round(total_size/(1024*1024),1)}MB")
        print(f"   Sparsity: {stats['sparsity_pct']:.1f}%")
        print(f"   Est. RAM: {stats['estimated_ram_mb']:.0f}MB")
        return True

if __name__ == '__main__':
    import sys
    bot = PullbotModel()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            success = bot.train()
            if success:
                print("\n📊 Pre-pruning stats:")
                before = bot.get_model_size_estimate()
                print(f"   Est. RAM: {before['estimated_ram_mb']:.0f}MB")
                
                # Start with 50% pruning
                bot.prune_model(target_sparsity=0.5)
                
                after = bot.get_model_size_estimate()
                print(f"\n📊 After 50% prune: {after['estimated_ram_mb']:.0f}MB RAM")
                
                # If still over 600MB, prune more aggressively
                sparsity = 0.5
                while after['estimated_ram_mb'] > 600 and sparsity < 0.9:
                    sparsity += 0.1
                    print(f"\n⚠️ Still too big. Trying {sparsity*100:.0f}% pruning...")
                    bot.prune_model(target_sparsity=sparsity)
                    after = bot.get_model_size_estimate()
                    print(f"📊 After {sparsity*100:.0f}%: {after['estimated_ram_mb']:.0f}MB RAM")
                
                bot.save_and_chunk()
                
                if after['estimated_ram_mb'] <= 600:
                    print("\n🎉 MODEL FITS ON RENDER!")
                else:
                    print(f"\n⚠️ Still {after['estimated_ram_mb']:.0f}MB - may not fit Render")
        
        elif command == 'prune':
            sparsity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
            before = bot.get_model_size_estimate()
            print(f"\n📊 Before: {before['estimated_ram_mb']:.0f}MB RAM")
            bot.prune_model(target_sparsity=sparsity)
            after = bot.get_model_size_estimate()
            print(f"📊 After: {after['estimated_ram_mb']:.0f}MB RAM")
            bot.save_and_chunk()
        
        elif command == 'size':
            stats = bot.get_model_size_estimate()
            print(f"\n📊 MODEL STATS")
            print(f"   Total params: {stats['total_params']:,}")
            print(f"   Non-zero: {stats['non_zero_params']:,}")
            print(f"   Sparsity: {stats['sparsity_pct']:.1f}%")
            print(f"   File size: {stats['file_size_mb']:.0f}MB")
            print(f"   Sparse size: {stats['sparse_size_mb']:.0f}MB")
            print(f"   Est. RAM: {stats['estimated_ram_mb']:.0f}MB")
        
        elif command == 'chunk':
            bot.save_and_chunk()
        
        else:
            print("Commands: train, prune [0.5-0.9], size, chunk")
    else:
        print("Commands: train, prune [0.5-0.9], size, chunk")
