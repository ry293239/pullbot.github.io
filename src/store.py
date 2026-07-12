"""
Pullbot Knowledge Store
Loads embedding model from repo chunks - NO internet needed.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import torch
import numpy as np
import json
import yaml
from transformers import AutoModel, AutoTokenizer, AutoConfig
import time
import glob

config_path = os.path.join(REPO_ROOT, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

class KnowledgeStore:
    def __init__(self):
        print("🧠 Initializing knowledge store...")
        
        chunks_dir = os.path.join(REPO_ROOT, "models", "chunks")
        
        # Reassemble model from chunks if needed
        self._reassemble_if_needed(chunks_dir)
        
        # Load tokenizer and model FROM CHUNKS (no internet)
        print(f"   Loading from repo chunks...")
        self.tokenizer = AutoTokenizer.from_pretrained(chunks_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.embedder = AutoModel.from_pretrained(
            chunks_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.embedder.eval()
        
        self.chunks = []
        self.store_path = os.path.join(REPO_ROOT, "knowledge", "store.json")
        self.load()
        print(f"   ✅ Knowledge store ready")
    
    def _reassemble_if_needed(self, chunks_dir):
        """Reassemble model chunks into full file if needed"""
        manifest_path = os.path.join(chunks_dir, "manifest.json")
        
        if not os.path.exists(manifest_path):
            print("   ⚠️ No manifest found - model chunks may be incomplete")
            return
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        weights_file = manifest.get('weights_filename', 'model.safetensors')
        reassembled_path = os.path.join(chunks_dir, weights_file)
        
        # Check if already reassembled and complete
        if os.path.exists(reassembled_path):
            expected_size = manifest.get('total_size', 0)
            actual_size = os.path.getsize(reassembled_path)
            if actual_size == expected_size:
                print(f"   ✅ Model already reassembled ({actual_size//1024//1024}MB)")
                return
        
        # Need to reassemble
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
    
    def embed_text(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        )
        with torch.no_grad():
            outputs = self.embedder(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def add_text(self, text, source="unknown"):
        chunk_size = config['knowledge']['chunk_size']
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk) < 20:
                continue
            embedding = self.embed_text(chunk)
            self.chunks.append({
                'text': chunk,
                'embedding': embedding.tolist(),
                'source': source,
                'added': int(time.time())
            })
        
        max_chunks = config['knowledge']['max_chunks']
        if len(self.chunks) > max_chunks:
            self.chunks = self.chunks[-max_chunks:]
        
        print(f"  📚 Knowledge store: {len(self.chunks)} chunks")
    
    def search(self, query, top_k=None):
        if top_k is None:
            top_k = config['knowledge']['top_k']
        if not self.chunks:
            return []
        
        query_emb = self.embed_text(query)
        scores = []
        for i, chunk in enumerate(self.chunks):
            chunk_emb = np.array(chunk['embedding'])
            sim = np.dot(query_emb, chunk_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-8
            )
            scores.append((sim, i))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return [self.chunks[i] for _, i in scores[:top_k]]
    
    def get_context(self, query, top_k=None):
        results = self.search(query, top_k)
        if not results:
            return None
        return "\n\n".join([r['text'] for r in results])
    
    def save(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        save_data = [
            {
                'text': c['text'], 
                'source': c['source'], 
                'added': c['added']
            } 
            for c in self.chunks
        ]
        with open(self.store_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"  💾 Saved {len(self.chunks)} chunks")
    
    def load(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, 'r') as f:
                save_data = json.load(f)
            print(f"  📂 Loading {len(save_data)} stored chunks...")
            for item in save_data:
                embedding = self.embed_text(item['text'])
                self.chunks.append({
                    'text': item['text'],
                    'embedding': embedding.tolist(),
                    'source': item.get('source', 'unknown'),
                    'added': item.get('added', 0)
                })
            print(f"  ✅ Rebuilt {len(self.chunks)} chunks with embeddings")
        else:
            print("  🆕 Fresh knowledge store")
    
    def build_from_corpus(self):
        corpus_path = os.path.join(REPO_ROOT, "data", "processed", "corpus.txt")
        if not os.path.exists(corpus_path):
            print("❌ No corpus found at", corpus_path)
            return
        
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        print(f"  📄 Corpus: {len(text):,} chars")
        
        sections = text.split('\n\n---\n\n')
        for section in sections:
            if section.strip():
                self.add_text(section.strip())
        
        self.save()
        print(f"✅ Built knowledge store: {len(self.chunks)} chunks")

if __name__ == '__main__':
    import sys
    store = KnowledgeStore()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'build':
            store.build_from_corpus()
        elif sys.argv[1] == 'search':
            query = sys.argv[2] if len(sys.argv) > 2 else "test"
            results = store.search(query)
            for r in results:
                print(f"\n📄 Score: {r[0]:.3f}\n{r[1]['text'][:200]}...")
