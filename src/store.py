"""
Pullbot Knowledge Store
Simple vector storage and retrieval using embeddings.
No external APIs needed - uses DistilGPT2's embeddings.
"""

import torch
import numpy as np
import json
import os
import yaml
from transformers import AutoModel, AutoTokenizer

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class KnowledgeStore:
    def __init__(self):
        print("🧠 Initializing knowledge store...")
        
        # Use sentence-transformers style model for embeddings
        # But we'll use DistilGPT2's hidden states to keep it simple
        self.model_name = config['model']['base']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedder = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.embedder.eval()
        
        self.chunks = []  # List of {'text': ..., 'embedding': ..., 'source': ...}
        self.store_path = "knowledge/store.json"
        
        self.load()
    
    def embed_text(self, text):
        """Convert text to embedding vector"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.embedder(**inputs)
            # Use mean of last hidden state as embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embedding
    
    def add_text(self, text, source="unknown"):
        """Add text to knowledge store"""
        chunk_size = config['knowledge']['chunk_size']
        
        # Split into chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk) < 20:  # Skip tiny chunks
                continue
            
            embedding = self.embed_text(chunk)
            
            self.chunks.append({
                'text': chunk,
                'embedding': embedding.tolist(),  # Convert to list for JSON
                'source': source,
                'added': int(__import__('time').time())
            })
        
        # Prune old chunks if too many
        max_chunks = config['knowledge']['max_stored_chunks']
        if len(self.chunks) > max_chunks:
            self.chunks = self.chunks[-max_chunks:]
        
        print(f"  📚 Knowledge store: {len(self.chunks)} chunks")
    
    def search(self, query, top_k=None):
        """Find most relevant chunks for a query"""
        if top_k is None:
            top_k = config['knowledge']['top_k']
        
        if len(self.chunks) == 0:
            return []
        
        query_embedding = self.embed_text(query)
        
        # Compute cosine similarities
        scores = []
        for i, chunk in enumerate(self.chunks):
            chunk_embedding = np.array(chunk['embedding'])
            
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-8
            )
            
            scores.append((similarity, i))
        
        # Get top k
        scores.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [self.chunks[i] for _, i in scores[:top_k]]
        
        return top_chunks
    
    def get_context(self, query, top_k=None):
        """Get concatenated context for a query"""
        results = self.search(query, top_k)
        if not results:
            return None
        
        context = "\n\n".join([r['text'] for r in results])
        return context
    
    def save(self):
        """Save knowledge store to disk"""
        os.makedirs('knowledge', exist_ok=True)
        
        # Save without embeddings to keep file small (they're regenerated)
        save_data = []
        for chunk in self.chunks:
            save_data.append({
                'text': chunk['text'],
                'source': chunk['source'],
                'added': chunk['added']
            })
        
        with open(self.store_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 Saved {len(self.chunks)} chunks")
    
    def load(self):
        """Load knowledge store from disk"""
        if os.path.exists(self.store_path):
            with open(self.store_path, 'r') as f:
                save_data = json.load(f)
            
            print(f"📂 Loading {len(save_data)} stored chunks...")
            
            # Rebuild embeddings
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
    
    def build_from_corpus(self, corpus_path="data/processed/corpus.txt"):
        """Build knowledge store from corpus file"""
        if not os.path.exists(corpus_path):
            print("❌ No corpus found")
            return
        
        with open(corpus_path, 'r') as f:
            text = f.read()
        
        # Split by the delimiter we used in scrape.py
        sections = text.split('\n\n---\n\n')
        
        for section in sections:
            if section.strip():
                self.add_text(section.strip())
        
        self.save()
        print(f"✅ Built knowledge store with {len(self.chunks)} chunks")

if __name__ == '__main__':
    store = KnowledgeStore()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        store.build_from_corpus()
    elif len(sys.argv) > 1 and sys.argv[1] == 'search':
        query = sys.argv[2] if len(sys.argv) > 2 else "What is machine learning?"
        results = store.search(query)
        for r in results:
            print(f"\n📄 {r['text'][:200]}...")
