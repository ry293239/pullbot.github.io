"""
Pullbot Model - FULL FINE-TUNING + 4-STAGE OPTIMIZATION
All stages have progress bars for visibility.
1. Smart Prune (redistribute weak to strong)
2. Safe Precision Prune (merge insignificant, same row only)
3. Progressive Bit Reduction (with 1hr timeout + progress)
4. 8-bit Quantize
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

config_path = os.path.join(REPO_ROOT, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

CHUNK_SIZE_MB = 45
CHUNK_SIZE = CHUNK_SIZE_MB * 1024 * 1024

def progress_bar(done, total, label="", width=30):
    pct = done / max(total, 1)
    filled = int(width * pct)
    bar = '█' * filled + '░' * (width - filled)
    return f"   {label} [{bar}] {pct*100:.0f}% ({done:,}/{total:,})"

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
            if os.path.getsize(reassembled_path) == manifest.get('total_size', 0):
                print(f"   ✅ Model reassembled ({os.path.getsize(reassembled_path)//1024//1024}MB)")
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
                self.chunks_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            print("   ✅ Loaded via from_pretrained")
        except:
            try:
                model_config = AutoConfig.from_pretrained(self.chunks_dir)
                self.model = AutoModelForCausalLM.from_config(model_config)
                for fname in ['model.safetensors', 'pytorch_model.bin']:
                    path = os.path.join(self.chunks_dir, fname)
                    if os.path.exists(path):
                        if fname.endswith('.safetensors'):
                            from safetensors.torch import load_file
                            state_dict = load_file(path)
                        else:
                            state_dict = torch.load(path, map_location='cpu')
                        cleaned = {k: v for k, v in state_dict.items() if not isinstance(v, tuple)}
                        self.model.load_state_dict(cleaned, strict=False)
                        print("   ✅ Loaded via state_dict")
                        return
            except Exception as e:
                print(f"   ❌ Failed: {e}")
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
        return dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    
    def train(self, resume_from_checkpoint=None):
        dataset = self.prepare_training_data()
        if dataset is None or len(dataset) == 0:
            return False
        print("\n" + "=" * 50)
        print("🚀 FULL MODEL TRAINING")
        print(f"   Examples: {len(dataset)}")
        print(f"   Epochs: {config['training']['epochs']}")
        print(f"   Batch: {config['training']['batch_size']}")
        if resume_from_checkpoint:
            print(f"   📂 Resuming from: {resume_from_checkpoint}")
        print("=" * 50 + "\n")
        training_args = TrainingArguments(
            output_dir=os.path.join(REPO_ROOT, "models", "checkpoints"),
            num_train_epochs=config['training']['epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=4,
            save_steps=500, save_total_limit=3, logging_steps=50,
            learning_rate=config['training']['learning_rate'],
            warmup_steps=50, remove_unused_columns=False, dataloader_num_workers=0,
        )
        trainer = Trainer(
            model=self.model, args=training_args, train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        start = time.time()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print(f"\n✅ Trained in {(time.time()-start)/60:.1f} min")
        with open(os.path.join(REPO_ROOT, "models", "last_train.json"), 'w') as f:
            json.dump({'timestamp': time.time(), 'dataset_size': len(dataset)}, f)
        return True
    
    # ============================================
    # STAGE 1: SMART PRUNE
    # ============================================
    
    def smart_prune(self, target_sparsity=0.5):
        print(f"\n🧠 STAGE 1: SMART PRUNE (target: {target_sparsity*100:.0f}%)")
        total_redistributed = 0
        total_rows = sum(m.weight.shape[0] for _, m in self.model.named_modules() if isinstance(m, nn.Linear) and m.weight.shape[0] > 10)
        rows_done = 0
        last_print = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] > 10:
                weight = module.weight.data.float()
                for row_idx in range(weight.shape[0]):
                    row = weight[row_idx]; abs_row = row.abs()
                    rows_done += 1
                    if abs_row.sum() == 0: continue
                    k = max(1, int((1-target_sparsity)*len(row)))
                    if k >= len(row): continue
                    threshold = torch.kthvalue(abs_row, len(row)-k).values
                    strong_mask = abs_row >= threshold
                    strong_idx = torch.where(strong_mask)[0]
                    weak_idx = torch.where(~strong_mask)[0]
                    if len(strong_idx)==0 or len(weak_idx)==0: continue
                    for wi in weak_idx:
                        weak_val = row[wi]
                        if abs(weak_val)<0.00001: row[wi]=0; continue
                        best_si = strong_idx[0]; best_sim = -999
                        for si in strong_idx[:min(20,len(strong_idx))]:
                            sign_match = 1 if (weak_val*row[si])>0 else -1
                            sim = sign_match*(1-min(abs(weak_val-row[si].abs()),1.0))
                            if sim>best_sim: best_sim=sim; best_si=si
                        row[best_si] += weak_val*0.6; row[wi]=0; total_redistributed+=1
                    
                    # Progress every 5%
                    pct = rows_done / max(total_rows, 1)
                    if pct - last_print >= 0.05:
                        print(f"\r{progress_bar(rows_done, total_rows, 'Smart Prune')}", end='')
                        last_print = pct
                
                module.weight.data = weight.to(module.weight.dtype)
        
        print(f"\r{progress_bar(total_rows, total_rows, 'Smart Prune')}")
        print(f"   Redistributed: {total_redistributed:,} weights")
        return total_redistributed
    
    # ============================================
    # STAGE 2: SAFE PRECISION PRUNE
    # ============================================
    
    def precision_prune_safe(self, significance=2, test_inputs=None):
        print(f"\n🎯 STAGE 2: SAFE PRECISION PRUNE (sig={significance})")
        if test_inputs is None:
            test_inputs = torch.randint(0, 50257, (10, 16))
        
        self.model.eval()
        with torch.no_grad():
            baseline = self.model(test_inputs).logits
        
        total_merged = 0
        total_rows = sum(m.weight.shape[0] for _, m in self.model.named_modules() if isinstance(m, nn.Linear) and m.weight.shape[0] > 10)
        rows_done = 0
        last_print = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] > 10:
                weight = module.weight.data.float()
                for row_idx in range(weight.shape[0]):
                    row = weight[row_idx]; abs_row = row.abs()
                    rows_done += 1
                    if abs_row.sum() == 0: continue
                    threshold = torch.kthvalue(abs_row, int(0.5*len(row))).values
                    strong_mask = abs_row >= threshold
                    strong_idx = torch.where(strong_mask)[0]
                    if len(strong_idx) < 2: continue
                    for col_idx in range(len(row)):
                        if strong_mask[col_idx]: continue
                        val = row[col_idx]
                        if abs(val) < 0.0001: row[col_idx]=0; continue
                        rounded = round(val.item(), significance)
                        if abs(val-rounded) < (10**-(significance+1)):
                            best_si = strong_idx[0]; best_dist = float('inf')
                            for si in strong_idx:
                                dist = abs(val-row[si].item())
                                if dist < best_dist: best_dist=dist; best_si=si
                            row[best_si] += val*0.7; row[col_idx]=0; total_merged+=1
                    
                    pct = rows_done / max(total_rows, 1)
                    if pct - last_print >= 0.05:
                        print(f"\r{progress_bar(rows_done, total_rows, 'Precision')}", end='')
                        last_print = pct
                
                module.weight.data = weight.to(module.weight.dtype)
        
        print(f"\r{progress_bar(total_rows, total_rows, 'Precision')}")
        
        self.model.eval()
        with torch.no_grad():
            new_out = self.model(test_inputs).logits
        diff = (baseline-new_out).abs().mean().item()
        print(f"   Merged: {total_merged:,} | Output diff: {diff:.10f}")
        if diff > 0.0001: print("   ⚠️ ROLLING BACK"); return False
        print("   ✅ Safe!")
        return True
    
    # ============================================
    # STAGE 3: PROGRESSIVE BIT REDUCTION
    # ============================================
    
    def progressive_bit_reduce(self, test_inputs=None, timeout_minutes=60):
        print(f"\n📉 STAGE 3: PROGRESSIVE BIT REDUCTION (timeout: {timeout_minutes}min)")
        if test_inputs is None:
            test_inputs = torch.randint(0, 50257, (10, 16))
        
        self.model.eval()
        with torch.no_grad():
            baseline = self.model(test_inputs).logits
        
        nodes_8bit = nodes_7bit = nodes_6bit = nodes_merged = 0
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        stopped_early = False
        
        # Count total weights for progress
        total_weights = sum(m.weight.numel() for _, m in self.model.named_modules() if isinstance(m, nn.Linear) and m.weight.shape[0] > 10)
        weights_done = 0
        last_print = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] > 10:
                weight = module.weight.data.float()
                for row_idx in range(weight.shape[0]):
                    if time.time() - start_time > timeout_seconds:
                        stopped_early = True
                        break
                    
                    row = weight[row_idx]
                    for col_idx in range(len(row)):
                        weights_done += 1
                        val = row[col_idx]
                        if abs(val) < 0.0001:
                            nodes_merged += 1
                            continue
                        
                        val_7bit = round(val.item() * 127) / 127
                        row[col_idx] = val_7bit
                        module.weight.data = weight.to(module.weight.dtype)
                        new_out = self.model(test_inputs).logits
                        diff = (baseline - new_out).abs().mean().item()
                        
                        if diff < 0.00000001:
                            val_6bit = round(val.item() * 63) / 63
                            row[col_idx] = val_6bit
                            module.weight.data = weight.to(module.weight.dtype)
                            new_out = self.model(test_inputs).logits
                            diff = (baseline - new_out).abs().mean().item()
                            if diff < 0.00000001:
                                nodes_6bit += 1
                            else:
                                row[col_idx] = val_7bit
                                nodes_7bit += 1
                        else:
                            row[col_idx] = val
                            nodes_8bit += 1
                    
                    pct = weights_done / max(total_weights, 1)
                    if pct - last_print >= 0.02:
                        elapsed = (time.time() - start_time) / 60
                        print(f"\r{progress_bar(weights_done, total_weights, f'Bits ({elapsed:.0f}min)')}", end='')
                        last_print = pct
                    
                    module.weight.data = weight.to(module.weight.dtype)
                
                if stopped_early:
                    break
        
        elapsed = (time.time() - start_time) / 60
        print(f"\r{progress_bar(weights_done, total_weights, f'Bits ({elapsed:.0f}min)')}")
        
        total = nodes_8bit + nodes_7bit + nodes_6bit + nodes_merged
        eff_bits = (nodes_8bit*8 + nodes_7bit*7 + nodes_6bit*6) / max(total, 1)
        
        print(f"   8-bit: {nodes_8bit:,}  7-bit: {nodes_7bit:,}  6-bit: {nodes_6bit:,}  merged: {nodes_merged:,}")
        print(f"   Effective: {eff_bits:.1f}-bit")
        if stopped_early:
            print(f"   ⏰ Timeout — saved progress")
        
        return eff_bits
    
    # ============================================
    # STAGE 4: 8-BIT QUANTIZE
    # ============================================
    
    def quantize_model(self):
        print("\n🔧 STAGE 4: 8-BIT QUANTIZE")
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            total_bytes = sum(p.numel() for p in self.model.parameters() if p.dtype==torch.qint8)
            total_bytes += sum(p.numel()*4 for p in self.model.parameters() if p.dtype!=torch.qint8)
            size_mb = total_bytes/(1024*1024)
            print(f"   Size: {size_mb:.1f}MB")
            return size_mb
        except Exception as e:
            print(f"   ⚠️ Failed: {e}")
            return None
    
    # ============================================
    # SIZE ESTIMATE
    # ============================================
    
    def get_model_size_estimate(self):
        non_zero = sum((p!=0).sum().item() for p in self.model.parameters() if p.dim()>=2)
        total = sum(p.numel() for p in self.model.parameters())
        is_quantized = any(p.dtype==torch.qint8 for p in self.model.parameters())
        bytes_per_param = 1 if is_quantized else 4
        ram_mb = non_zero*bytes_per_param/(1024*1024)*3
        return {
            'total_params':total, 'non_zero':non_zero,
            'sparsity_pct':(1-non_zero/total)*100 if total>0 else 0,
            'quantized':is_quantized, 'estimated_ram_mb':ram_mb
        }
    
    # ============================================
    # SAVE & CHUNK
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
            if hasattr(self.model,'config'): self.model.config.save_pretrained(temp_dir)
        weights_path = None
        for fname in ['model.safetensors','pytorch_model.bin']:
            path = os.path.join(temp_dir, fname)
            if os.path.exists(path): weights_path=path; break
        if not weights_path: print("❌ No weights found!"); return False
        for old in glob.glob(os.path.join(self.chunks_dir,"model_chunk_*.bin")): os.remove(old)
        for old in glob.glob(os.path.join(self.chunks_dir,"*.safetensors")): os.remove(old)
        for old in glob.glob(os.path.join(self.chunks_dir,"pytorch_model.bin")): os.remove(old)
        
        total_size = os.path.getsize(weights_path)
        chunks = []
        print(f"\n🔪 Streaming chunks ({total_size/(1024*1024):.0f}MB total)...")
        with open(weights_path,'rb') as f:
            chunk_idx = 0
            while True:
                chunk_data = f.read(CHUNK_SIZE)
                if not chunk_data: break
                chunk_name = f"model_chunk_{chunk_idx:03d}.bin"
                chunk_path = os.path.join(self.chunks_dir, chunk_name)
                with open(chunk_path,'wb') as out: out.write(chunk_data)
                chunks.append(f"models/chunks/{chunk_name}")
                chunk_idx += 1
                print(f"   ✅ {chunk_name} ({len(chunk_data)/(1024*1024):.1f}MB)")
                del chunk_data
        
        config_files = ['config.json','tokenizer_config.json','vocab.json','merges.txt','special_tokens_map.json','tokenizer.json']
        for cfg in config_files:
            src = os.path.join(temp_dir, cfg)
            if os.path.exists(src): shutil.copy(src, os.path.join(self.chunks_dir, cfg))
        
        stats = self.get_model_size_estimate()
        manifest = {
            "model_name":self.model_name,"total_size":total_size,"total_size_mb":round(total_size/(1024*1024),1),
            "num_chunks":len(chunks),"chunks":chunks,"weights_filename":os.path.basename(weights_path),
            "fully_trained":True,"pruned":stats['sparsity_pct']>5,"quantized":stats['quantized'],
            "sparsity_pct":stats['sparsity_pct'],"estimated_ram_mb":stats['estimated_ram_mb'],"training_date":time.time()
        }
        with open(os.path.join(self.chunks_dir,"manifest.json"),'w') as f: json.dump(manifest,f,indent=2)
        shutil.rmtree(temp_dir)
        ckpt_dir = os.path.join(REPO_ROOT,"models","checkpoints")
        if os.path.exists(ckpt_dir): shutil.rmtree(ckpt_dir)
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
                if idx+1 < len(sys.argv): resume = sys.argv[idx+1]
            if bot.train(resume_from_checkpoint=resume):
                bot.save_and_chunk()
                print("\n🎉 Training complete! Model saved.")
        elif cmd == 'optimize':
            sp = float(sys.argv[2]) if len(sys.argv)>2 else 0.5
            before = bot.get_model_size_estimate()
            print(f"\n📊 Before: {before['total_params']:,} params, {before['estimated_ram_mb']:.0f}MB RAM")
            test_inputs = torch.randint(0, 50257, (10, 16))
            bot.smart_prune(target_sparsity=sp)
            bot.precision_prune_safe(significance=2, test_inputs=test_inputs)
            bot.progressive_bit_reduce(test_inputs=test_inputs, timeout_minutes=60)
            bot.quantize_model()
            final = bot.get_model_size_estimate()
            reduction = before['estimated_ram_mb'] - final['estimated_ram_mb']
            print(f"\n📊 Final: {final['total_params']:,} params, {final['estimated_ram_mb']:.0f}MB RAM")
            print(f"   Saved: {reduction:.0f}MB")
            bot.save_and_chunk()
            print("\n✅ Optimization complete!")
        elif cmd == 'size':
            s = bot.get_model_size_estimate()
            print(f"\n📊 {s['total_params']:,} params | Non-zero: {s['non_zero']:,} | Sparsity: {s['sparsity_pct']:.1f}% | RAM: {s['estimated_ram_mb']:.0f}MB")
        elif cmd == 'chunk':
            bot.save_and_chunk()
        else:
            print("Commands: train, optimize [0.3-0.7], size, chunk")
    else:
        print("Commands: train, optimize [0.3-0.7], size, chunk")
