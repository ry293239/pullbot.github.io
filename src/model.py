"""
Pullbot Model - FULL FINE-TUNING + PRUNING + QUANTIZATION + ONNX EXPORT
Trains, saves checkpoints, prunes, quantizes, exports to ONNX for deployment.
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
        print("🤖 PULLBOT MODEL LOADER")
        print("=" * 50)
        
        self._reassemble_if_needed()
        
        print("\n📂 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.chunks_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("📂 Loading model...")
        self._load_model_safe()
        
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
    
    def _load_model_safe(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.chunks_dir,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            print("   ✅ Loaded via from_pretrained")
        except Exception as e:
            print(f"   ⚠️ Trying manual load...")
            try:
                model_config = AutoConfig.from_pretrained(self.chunks_dir)
                self.model = AutoModelForCausalLM.from_config(model_config)
                
                weights_path = None
                for fname in ['model.safetensors', 'pytorch_model.bin']:
                    path = os.path.join(self.chunks_dir, fname)
                    if os.path.exists(path):
                        weights_path = path
                        break
                
                if weights_path:
                    if weights_path.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        state_dict = load_file(weights_path)
                    else:
                        state_dict = torch.load(weights_path, map_location='cpu')
                    
                    cleaned = {k: v for k, v in state_dict.items() if not isinstance(v, tuple)}
                    self.model.load_state_dict(cleaned, strict=False)
                    print("   ✅ Loaded via state_dict")
            except Exception as e2:
                print(f"   ❌ Failed: {e2}")
                raise
    
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
    
    def train(self, resume_from_checkpoint=None):
        dataset = self.prepare_training_data()
        if dataset is None or len(dataset) == 0:
            return False
        
        print("\n" + "=" * 50)
        print("🚀 FULL MODEL TRAINING")
        print(f"   Examples: {len(dataset)}")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Batch: {config['training']['batch_size']}")
        print(f"   LR: {config['training']['learning_rate']}")
        if resume_from_checkpoint:
            print(f"   📂 Resuming from: {resume_from_checkpoint}")
        print("=" * 50 + "\n")
        
        training_args = TrainingArguments(
            output_dir=os.path.join(REPO_ROOT, "models", "checkpoints"),
            num_train_epochs=config['training']['epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=4,
            save_steps=500,
            save_total_limit=3,
            logging_steps=50,
            learning_rate=config['training']['learning_rate'],
            warmup_steps=50,
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
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        elapsed = (time.time() - start) / 60
        print(f"\n✅ Trained in {elapsed:.1f} min")
        
        with open(os.path.join(REPO_ROOT, "models", "last_train.json"), 'w') as f:
            json.dump({'timestamp': time.time(), 'dataset_size': len(dataset), 'minutes': elapsed}, f)
        
        return True
    
    def prune_model(self, target_sparsity=0.5):
        print("\n" + "=" * 50)
        print(f"✂️ PRUNING (target: {target_sparsity*100:.0f}%)")
        print("=" * 50)
        
        total_removed = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.data.float()
                total_params += weight.numel()
                flat = weight.abs().flatten()
                k = int(target_sparsity * flat.numel())
                
                if k > 0 and k < flat.numel():
                    threshold = torch.kthvalue(flat, k).values
                    mask = weight.abs() > threshold
                    total_removed += (mask == False).sum().item()
                    module.weight.data = (weight * mask.float()).to(module.weight.dtype)
        
        sparsity = (total_removed / total_params * 100) if total_params > 0 else 0
        print(f"   Removed: {total_removed:,} / {total_params:,} ({sparsity:.1f}%)")
        return sparsity
    
    def quantize_model(self):
        print("\n" + "=" * 50)
        print("🔧 QUANTIZING TO 8-BIT")
        print("=" * 50)
        
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            total_bytes = sum(p.numel() for p in self.model.parameters() if p.dtype == torch.qint8)
            total_bytes += sum(p.numel() * 4 for p in self.model.parameters() if p.dtype != torch.qint8)
            size_mb = total_bytes / (1024 * 1024)
            print(f"   Size: {size_mb:.1f}MB")
            return size_mb
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            return None
    
    def export_to_onnx(self, output_path=None):
        """Export model to ONNX format for lightweight deployment"""
        if output_path is None:
            output_path = os.path.join(REPO_ROOT, "models", "pullbot.onnx")
        
        print("\n" + "=" * 50)
        print("📤 EXPORTING TO ONNX")
        print("=" * 50)
        
        try:
            self.model.eval()
            self.model.to('cpu')
            
            dummy_input = torch.randint(0, 50257, (1, 64))
            dummy_mask = torch.ones(1, 64)
            
            torch.onnx.export(
                self.model,
                (dummy_input, dummy_mask),
                output_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'attention_mask': {0: 'batch', 1: 'sequence'},
                    'logits': {0: 'batch', 1: 'sequence'}
                },
                opset_version=14,
                do_constant_folding=True
            )
            
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   ✅ ONNX model: {size_mb:.1f}MB")
            return output_path
            
        except Exception as e:
            print(f"   ⚠️ Export with mask failed: {e}")
            try:
                dummy_input = torch.randint(0, 50257, (1, 64))
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    output_path,
                    input_names=['input_ids'],
                    output_names=['logits'],
                    dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}},
                    opset_version=14,
                    do_constant_folding=True
                )
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   ✅ ONNX model (no mask): {size_mb:.1f}MB")
                return output_path
            except Exception as e2:
                print(f"   ❌ ONNX export failed: {e2}")
                return None
    
    def get_model_size_estimate(self):
        non_zero = sum((p != 0).sum().item() for p in self.model.parameters() if p.dim() >= 2)
        total = sum(p.numel() for p in self.model.parameters())
        is_quantized = any(p.dtype == torch.qint8 for p in self.model.parameters())
        bytes_per_param = 1 if is_quantized else 4
        ram_mb = non_zero * bytes_per_param / (1024 * 1024) * 3
        
        return {
            'total_params': total,
            'non_zero': non_zero,
            'sparsity_pct': (1 - non_zero/total)*100 if total > 0 else 0,
            'quantized': is_quantized,
            'estimated_ram_mb': ram_mb
        }
    
    def save_and_chunk(self):
        print("\n💾 Saving model...")
        
        temp_dir = os.path.join(REPO_ROOT, "models", "temp_trained")
        os.makedirs(temp_dir, exist_ok=True)
        
        self.tokenizer.save_pretrained(temp_dir)
        
        try:
            self.model.save_pretrained(temp_dir)
            print("   Saved via save_pretrained")
        except:
            print("   Using torch.save for quantized model...")
            torch.save(self.model.state_dict(), os.path.join(temp_dir, "pytorch_model.bin"))
            if hasattr(self.model, 'config'):
                self.model.config.save_pretrained(temp_dir)
        
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
        
        for old in glob.glob(os.path.join(self.chunks_dir, "model_chunk_*.bin")):
            os.remove(old)
        for old in glob.glob(os.path.join(self.chunks_dir, "*.safetensors")):
            os.remove(old)
        for old in glob.glob(os.path.join(self.chunks_dir, "pytorch_model.bin")):
            os.remove(old)
        
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
        
        config_files = ['config.json', 'tokenizer_config.json', 'vocab.json',
                        'merges.txt', 'special_tokens_map.json', 'tokenizer.json']
        for cfg in config_files:
            src = os.path.join(temp_dir, cfg)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(self.chunks_dir, cfg))
        
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
        ckpt_dir = os.path.join(REPO_ROOT, "models", "checkpoints")
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        
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
            resume = None
            if '--resume' in sys.argv:
                idx = sys.argv.index('--resume')
                if idx + 1 < len(sys.argv):
                    resume = sys.argv[idx + 1]
            
            if bot.train(resume_from_checkpoint=resume):
                print("\n📊 Pre-optimization:")
                before = bot.get_model_size_estimate()
                print(f"   RAM: {before['estimated_ram_mb']:.0f}MB")
                
                sparsity = 0.5
                while True:
                    bot.prune_model(target_sparsity=sparsity)
                    after = bot.get_model_size_estimate()
                    if after['estimated_ram_mb'] < 400 or sparsity >= 0.9:
                        break
                    sparsity += 0.1
                
                bot.quantize_model()
                final = bot.get_model_size_estimate()
                print(f"\n📊 Final: {final['estimated_ram_mb']:.0f}MB RAM")
                bot.save_and_chunk()
                
                # Export to ONNX
                bot.export_to_onnx()
                
                print("\n🎉 TRAINED, PRUNED, QUANTIZED & EXPORTED!")
        
        elif command == 'prune':
            sparsity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
            bot.prune_model(target_sparsity=sparsity)
            bot.save_and_chunk()
        
        elif command == 'quantize':
            bot.quantize_model()
            bot.save_and_chunk()
        
        elif command == 'optimize':
            sparsity = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
            before = bot.get_model_size_estimate()
            print(f"\n📊 Before: {before['estimated_ram_mb']:.0f}MB")
            bot.prune_model(target_sparsity=sparsity)
            bot.quantize_model()
            final = bot.get_model_size_estimate()
            print(f"📊 After: {final['estimated_ram_mb']:.0f}MB")
            bot.save_and_chunk()
            bot.export_to_onnx()
        
        elif command == 'onnx':
            bot.export_to_onnx()
        
        elif command == 'size':
            stats = bot.get_model_size_estimate()
            print(f"\n📊 Params: {stats['total_params']:,}")
            print(f"   Non-zero: {stats['non_zero']:,}")
            print(f"   Sparsity: {stats['sparsity_pct']:.1f}%")
            print(f"   Quantized: {stats['quantized']}")
            print(f"   RAM: {stats['estimated_ram_mb']:.0f}MB")
        
        elif command == 'chunk':
            bot.save_and_chunk()
        
        else:
            print("Commands: train, prune [0.5-0.9], quantize, optimize [0.5-0.9], onnx, size, chunk")
    else:
        print("Commands: train, prune [0.5-0.9], quantize, optimize [0.5-0.9], onnx, size, chunk")
