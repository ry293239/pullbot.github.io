"""Nuclear clean the corpus and rebuild knowledge store"""

import os, sys, re, json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')

if not os.path.exists(corpus_path):
    print("No corpus found")
    sys.exit(0)

with open(corpus_path, 'r') as f:
    text = f.read()

before = len(text)

# Nuclear cleaning
text = re.sub(r'<[^>]+>', '', text)
text = re.sub(r'https?://\S+', '', text)
text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
text = re.sub(r'`[^`]+`', '', text)
text = re.sub(r'#[^\n]*', '', text)
text = re.sub(r'\n{3,}', '\n\n', text)
text = re.sub(r'[=\-*]{3,}', '', text)
text = re.sub(r'\|[^|]+\|', '', text)
text = re.sub(r'badge|shield|svg|build status|coverage|discord|chat|join|releases|linux|macos|windows', '', text, flags=re.IGNORECASE)

# Remove short lines
lines = [l for l in text.split('\n') if len(l.strip()) > 30 or l.strip() == '---']
text = '\n'.join(lines)

with open(corpus_path, 'w') as f:
    f.write(text)

after = len(text)
print(f'Cleaned corpus: {before:,} -> {after:,} chars ({((1-after/before)*100):.0f}% removed)')

# Wipe knowledge store
store_path = os.path.join(REPO_ROOT, 'knowledge', 'store.json')
with open(store_path, 'w') as f:
    json.dump([], f)
print('Knowledge store wiped')

# Rebuild
print('Rebuilding knowledge...')
from store import KnowledgeStore
store = KnowledgeStore()
store.build_from_corpus()

# Show sample
with open(store_path) as f:
    data = json.load(f)
print(f'\nKnowledge chunks: {len(data)}')
print('\n--- First 3 chunks ---')
for i, chunk in enumerate(data[:3]):
    print(f'\n{i+1}. {chunk["text"][:200]}...')
