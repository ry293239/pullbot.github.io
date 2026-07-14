"""Clean corpus text of markdown, HTML, URLs"""
import re, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
if not os.path.exists(corpus_path):
    print("No corpus found")
    sys.exit(0)

with open(corpus_path, 'r') as f:
    text = f.read()

text = re.sub(r'<[^>]+>', '', text)
text = re.sub(r'https?://\S+', '', text)
text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
text = re.sub(r'`[^`]+`', '', text)
text = re.sub(r'#[^\n]*', '', text)
text = re.sub(r'\n{3,}', '\n\n', text)
text = re.sub(r'[=\-*]{3,}', '', text)

with open(corpus_path, 'w') as f:
    f.write(text)

print(f'Cleaned corpus: {len(text):,} chars')
