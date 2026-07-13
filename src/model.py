"""
Pullbot Model - FULL FINE-TUNING + ACTUAL SHRINKING + COMPACT BINARY + ONNX
- Pruning actually shrinks layers (not just zeroing)
- Exports compact binary format (not CSV)
- Streaming chunking
- ONNX export for deployment
- RAG-ready: knowledge separate from generation
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import torch
import torch.nn as nn
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
import struct
import numpy as np

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
        manifest_path = os.path.join(self.chunks_dir, "manifest.json")
        is_quantized = False
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                is_quantized = json.load(f).get('quantized', False)
        
        try:
            if is_quantized:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.chunks_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
                )
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("   ✅ Loaded quantized model")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.chunks_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
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
            model=self.model, args=training_args, train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        start = time.time()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        elapsed = (time.time() - start) / 60
        print(f"\n✅ Trained in {elapsed:.1f} min")
        with open(os.path.join(REPO_ROOT, "models", "last_train.json"), 'w') as f:
            json.dump({'timestamp': time.time(), 'dataset_size': len(dataset), 'minutes': elapsed}, f)
        return True
    
    # ============================================
    # ACTUAL SHRINKING PRUNING (removes neurons)
    # ============================================
    
    def shrink_model(self, target_sparsity=0.5):
        """Actually shrink layers by removing unimportant neurons"""
        print("\n" + "=" * 50)
        print(f"✂️ SHRINKING MODEL ({target_sparsity*100:.0f}%)")
        print("=" * 50)
        
        total_before = sum(p.numel() for p in self.model.parameters())
        layers_shrunk = 0
        
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and module.weight.shape[0] > 100:
                weight = module.weight.data.float()
                
                # Score each output neuron by its total absolute weight
                importance = weight.abs().sum(dim=1)
                k = max(10, int((1 - target_sparsity) * importance.numel()))
                
                if k >= importance.numel():
                    continue
                
                threshold = torch.kthvalue(importance, importance.numel() - k).values
                keep_mask = importance >= threshold
                
                if keep_mask.sum() == 0:
                    continue
                
                # Create smaller layer
                new_out = keep_mask.sum().item()
                new_linear = nn.Linear(weight.shape[1], new_out, bias=module.bias is not None)
                new_linear.weight.data = weight[keep_mask].clone()
                if module.bias is not None:
                    new_linear.bias.data = module.bias.data[keep_mask].clone()
                
                # Replace in model
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                if parent_name:
                    parent = dict(self.model.named_modules()).get(parent_name)
                    if parent:
                        setattr(parent, attr_name, new_linear)
                        layers_shrunk += 1
        
        total_after = sum(p.numel() for p in self.model.parameters())
        reduction = (1 - total_after/total_before) * 100
        print(f"   Layers shrunk: {layers_shrunk}")
        print(f"   Params: {total_before:,} → {total_after:,} ({reduction:.1f}% smaller)")
        return reduction
    
    # ============================================
    # QUANTIZATION
    # ============================================
    
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
            print(f"   Quantized size: {size_mb:.1f}MB")
            return size_mb
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            return None
    
    # ============================================
    # COMPACT BINARY EXPORT (not CSV)
    # ============================================
    
    def export_compact(self, output_path=None):
        """Export weights as compact binary with only non-zero values"""
        if output_path is None:
            output_path = os.path.join(REPO_ROOT, "models", "pullbot.bin")
        
        print("\n" + "=" * 50)
        print("📦 EXPORTING COMPACT BINARY")
        print("=" * 50)
        
        with open(output_path, 'wb') as f:
            # Header
            f.write(b'PULLBOT')  # Magic number
            f.write(struct.pack('I', 1))  # Version
            
            layer_count = 0
            for name, param in self.model.named_parameters():
                if param.dim() >= 2:
                    layer_count += 1
            
            f.write(struct.pack('I', layer_count))
            
            for name, param in self.model.named_parameters():
                if param.dim() < 2:
                    continue
                
                weights = param.data.cpu().numpy()
                
                # Only non-zero
                mask = np.abs(weights) > 0.0001
                indices = np.where(mask)
                values = weights[mask]
                
                name_bytes = name.encode()
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('II', *weights.shape))
                f.write(struct.pack('I', len(values)))
                
                # Write sparse indices and values
                for r, c, v in zip(indices[0], indices[1], values):
                    f.write(struct.pack('IIf', int(r), int(c), float(v)))
        
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        # Count total non-zero
        non_zero = sum(
            (np.abs(p.data.cpu().numpy()) > 0.0001).sum()
            for p in self.model.parameters() if p.dim() >= 2
        )
        total = sum(p.numel() for p in self.model.parameters() if p.dim() >= 2)
        sparsity = (1 - non_zero/total) * 100 if total > 0 else 0
        
        print(f"   ✅ Compact binary: {size_mb:.1f}MB")
        print(f"   Non-zero weights: {non_zero:,} / {total:,} ({sparsity:.1f}% sparse)")
        return output_path
    
    # ============================================
    # ONNX EXPORT
    # ============================================
    
    def export_to_onnx(self, output_path=None):
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
                self.model, (dummy_input, dummy_mask), output_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'attention_mask': {0: 'batch', 1: 'sequence'},
                    'logits': {0: 'batch', 1: 'sequence'}
                },
                opset_version=14, do_constant_folding=True
            )
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   ✅ ONNX: {size_mb:.1f}MB")
            return output_path
        except:
            try:
                dummy_input = torch.randint(0, 50257, (1, 64))
                torch.onnx.export(
                    self.model, dummy_input, output_path,
                    input_names=['input_ids'], output_names=['logits'],
                    dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}},
                    opset_version=14, do_constant_folding=True
                )
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"   ✅ ONNX (no mask): {size_mb:.1f}MB")
                return output_path
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return None
    
    # ============================================
    # SIZE ESTIMATE
    # ============================================
    
    def get_model_size_estimate(self):
        non_zero = sum((p != 0).sum().item() for p in self.model.parameters() if p.dim() >= 2)
        total = sum(p.numel() for p in self.model.parameters())
        is_quantized = any(p.dtype == torch.qint8 for p in self.model.parameters())
        bytes_per_param = 1 if is_quantized else 4
        ram_mb = non_zero * bytes_per_param / (1024 * 1024) * 3
        return {
            'total_params': total, 'non_zero': non_zero,
            'sparsity_pct': (1 - non_zero/total)*100 if total > 0 else 0,
            'quantized': is_quantized, 'estimated_ram_mb': ram_mb
        }
    
    # ============================================
    # STREAMING SAVE & CHUNK
    # ============================================
    
    def save_and_chunk(self):
        print("\n💾 Saving model...")
        temp_dir = os.path.join(REPO_ROOT, "models", "temp_trained")
        os.makedirs(temp_dir, exist_ok=True)
        self.tokenizer.save_pretrained(temp_dir)
        try:
            self.model.save_pretrained(temp_dir)
        except:
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
        
        for old in glob.glob(os.path.join(self.chunks_dir, "model_chunk_*.bin")):
            os.remove(old)
        for old in glob.glob(os.path.join(self.chunks_dir, "*.safetensors")):
            os.remove(old)
        for old in glob.glob(os.path.join(self.chunks_dir, "pytorch_model.bin")):
            os.remove(old)
        
        total_size = os.path.getsize(weights_path)
        chunks = []
        print(f"\n🔪 Streaming chunks ({CHUNK_SIZE_MB}MB each)...")
        with open(weights_path, 'rb') as f:
            chunk_idx = 0
            while True:
                chunk_data = f.read(CHUNK_SIZE)
                if not chunk_data:
                    break
                chunk_name = f"model_chunk_{chunk_idx:03d}.bin"
                chunk_path = os.path.join(self.chunks_dir, chunk_name)
                with open(chunk_path, 'wb') as out:
                    out.write(chunk_data)
                chunks.append(f"models/chunks/{chunk_name}")
                print(f"   ✅ {chunk_name} ({len(chunk_data)/(1024*1024):.1f}MB)")
                chunk_idx += 1
                del chunk_data
        
        config_files = ['config.json', 'tokenizer_config.json', 'vocab.json',
                        'merges.txt', 'special_tokens_map.json', 'tokenizer.json']
        for cfg in config_files:
            src = os.path.join(temp_dir, cfg)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(self.chunks_dir, cfg))
        
        stats = self.get_model_size_estimate()
        manifest = {
            "model_name": self.model_name, "total_size": total_size,
            "total_size_mb": round(total_size/(1024*1024), 1),
            "num_chunks": len(chunks), "chunks": chunks,
            "weights_filename": os.path.basename(weights_path),
            "fully_trained": True, "pruned": stats['sparsity_pct'] > 5,
            "quantized": stats['quantized'], "sparsity_pct": stats['sparsity_pct'],
            "estimated_ram_mb": stats['estimated_ram_mb'], "training_date": time.time()
        }
        with open(os.path.join(self.chunks_dir, "manifest.json"), 'w') as f:
            json.dump(manifest, f, indent=2)
        
        shutil.rmtree(temp_dir)
        ckpt_dir = os.path.join(REPO_ROOT, "models", "checkpoints")
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        
        print(f"\n✅ Saved! {len(chunks)} chunks, {round(total_size/(1024*1024),1)}MB")
        print(f"   Sparsity: {stats['sparsity_pct']:.1f}% | Quantized: {stats['quantized']} | RAM: {stats['estimated_ram_mb']:.0f}MB")
        return True

if __name__ == '__main__':
    import sys
    bot = PullbotModel()
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == 'train':
            resume = None
            if '--resume' in sys.argv:
                idx = sys.argv.index('--resume')
                if idx + 1 < len(sys.argv):
                    resume = sys.argv[idx + 1]
            if bot.train(resume_from_checkpoint=resume):
                before = bot.get_model_size_estimate()
                print(f"\n📊 Before: {before['total_params']:,} params, {before['estimated_ram_mb']:.0f}MB RAM")
                
                # Shrink progressively
                for sp in [0.3, 0.5, 0.7]:
                    bot.shrink_model(target_sparsity=sp)
                    after = bot.get_model_size_estimate()
                    print(f"   After {sp*100:.0f}%: {after['total_params']:,} params, {after['estimated_ram_mb']:.0f}MB RAM")
                
                bot.quantize_model()
                final = bot.get_model_size_estimate()
                print(f"\n📊 Final: {final['total_params']:,} params, {final['estimated_ram_mb']:.0f}MB RAM")
                bot.save_and_chunk()
                bot.export_compact()
                bot.export_to_onnx()
                print("\n🎉 TRAINED, SHRUNK, QUANTIZED & EXPORTED!")
        
        elif cmd == 'shrink':
            sp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
            bot.shrink_model(target_sparsity=sp)
            bot.save_and_chunk()
        
        elif cmd == 'quantize':
            bot.quantize_model()
            bot.save_and_chunk()
        
        elif cmd == 'optimize':
            sp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
            before = bot.get_model_size_estimate()
            print(f"\n📊 Before: {before['total_params']:,} params")
            bot.shrink_model(target_sparsity=sp)
            bot.quantize_model()
            final = bot.get_model_size_estimate()
            print(f"📊 After: {final['total_params']:,} params, {final['estimated_ram_mb']:.0f}MB RAM")
            bot.save_and_chunk()
            bot.export_compact()
            bot.export_to_onnx()
        
        elif cmd == 'onnx':
            bot.export_to_onnx()
        
        elif cmd == 'compact':
            bot.export_compact()
        
        elif cmd == 'size':
            s = bot.get_model_size_estimate()
            print(f"\n📊 {s['total_params']:,} params | Non-zero: {s['non_zero']:,} | Sparsity: {s['sparsity_pct']:.1f}% | RAM: {s['estimated_ram_mb']:.0f}MB")
        
        elif cmd == 'chunk':
            bot.save_and_chunk()
        
        else:
            print("Commands: train, shrink [0.3-0.9], quantize, optimize, onnx, compact, size, chunk")
    else:
        print("Commands: train, shrink [0.3-0.9], quantize, optimize, onnx, compact, size, chunk")
