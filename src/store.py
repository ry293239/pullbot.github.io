"""
Pullbot Knowledge Store
Simple vector storage and retrieval.
"""

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
import yaml
from transformers import AutoModel, AutoTokenizer
import time

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class KnowledgeStore:
    def __init__(self):
        print("🧠 Initializing knowledge store...")
        
        self.model_name = config['model']['base']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedder = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.embedder.eval()
        
        self.chunks = []
        self.store_path = "knowledge/store.json"
        self.load()
    
    def embed_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
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
        
        if len(self.chunks) > config['knowledge']['max_chunks']:
            self.chunks = self.chunks[-config['knowledge']['max_chunks']:]
        
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
            sim = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb) + 1e-8)
            scores.append((sim, i))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return [self.chunks[i] for _, i in scores[:top_k]]
    
    def get_context(self, query, top_k=None):
        results = self.search(query, top_k)
        if not results:
            return None
        return "\n\n".join([r['text'] for r in results])
    
    def save(self):
        os.makedirs('knowledge', exist_ok=True)
        save_data = [{'text': c['text'], 'source': c['source'], 'added': c['added']} for c in self.chunks]
        with open(self.store_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    def load(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, 'r') as f:
                save_data = json.load(f)
            print(f"📂 Loading {len(save_data)} stored chunks...")
            for item in save_data:
                embedding = self.embed_text(item['text'])
                self.chunks.append({
                    'text': item['text'],
                    'embedding': embedding.tolist(),
                    'source': item.get('source', 'unknown'),
                    'added': item.get('added', 0)
                })
            print(f"  ✅ Rebuilt {len(self.chunks)} chunks")
        else:
            print("  🆕 Fresh knowledge store")
    
    def build_from_corpus(self, corpus_path="data/processed/corpus.txt"):
        if not os.path.exists(corpus_path):
            print("❌ No corpus found")
            return
        with open(corpus_path, 'r') as f:
            text = f.read()
        sections = text.split('\n\n---\n\n')
        for section in sections:
            if section.strip():
                self.add_text(section.strip())
        self.save()
        print(f"✅ Built knowledge store: {len(self.chunks)} chunks")

if __name__ == '__main__':
    import sys
    store = KnowledgeStore()
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        store.build_from_corpus()
    elif len(sys.argv) > 1 and sys.argv[1] == 'search':
        query = sys.argv[2] if len(sys.argv) > 2 else "test"
        results = store.search(query)
        for r in results:
            print(f"\n📄 {r['text'][:200]}...")
