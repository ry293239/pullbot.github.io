"""
Generate sentence-completion training examples from wordbank.
Filters out code, YAML, and markdown junk.
"""

import os, sys, re, json, random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def is_clean_text(text):
    """Filter out code, YAML, markdown-heavy junk"""
    junk_patterns = [
        r'```', r'apiVersion', r'kind:', r'metadata:', r'│', r'├',
        r'docker', r'kubernetes', r'nginx', r'yaml', r'namespace',
        r'kubectl', r'deployment', r'persistentvolume', r'configmap',
        r'FIXED RESOURCE CATALOG', r'SAMPLE YAML', r'DIFFICULTY MODIFIERS',
        r'Core Resources', r'badge', r'shield', r'build status'
    ]
    text_lower = text.lower()
    for pattern in junk_patterns:
        if pattern.lower() in text_lower:
            return False
    return True

def extract_key_phrases(text, max_phrases=100):
    phrases = set()
    words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
    phrases.update(words[:50])
    patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
        r'\b[a-z]+ [a-z]+ [a-z]+\b',
        r'\b(artificial|neural|deep|machine|computer|binary|operating|data) [a-z]+\b',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        phrases.update(matches[:30])
    return list(phrases)[:max_phrases]

def generate_completion_examples(text, phrase):
    examples = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        if phrase.lower() in sentence.lower() and len(sentence) > 30:
            idx = sentence.lower().find(phrase.lower())
            if idx > 0:
                prompt = sentence[:idx + len(phrase)].strip()
                completion = sentence[idx + len(phrase):].strip()
                if len(prompt) > 10 and len(completion) > 5:
                    if is_clean_text(prompt) and is_clean_text(completion):
                        examples.append({
                            'prompt': prompt,
                            'completion': completion,
                            'full': sentence.strip()
                        })
    return examples[:3]

def build_training_data(corpus_path, output_path, max_examples=500):
    print("=" * 50)
    print("📝 GENERATING TRAINING EXAMPLES")
    print("=" * 50)
    
    if not os.path.exists(corpus_path):
        print("No corpus found")
        return
    
    with open(corpus_path, 'r') as f:
        text = f.read()
    
    # Filter out junk lines
    lines = text.split('\n')
    clean_lines = [l for l in lines if is_clean_text(l)]
    text = '\n'.join(clean_lines)
    
    print(f"Corpus: {len(text):,} chars (junk removed)")
    
    phrases = extract_key_phrases(text)
    print(f"Phrases: {len(phrases)}")
    
    all_examples = []
    for phrase in phrases[:50]:
        examples = generate_completion_examples(text, phrase)
        all_examples.extend(examples)
    
    # Add word-definition pairs from wordbank
    wordbank_path = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
    if os.path.exists(wordbank_path):
        with open(wordbank_path) as f:
            bank = json.load(f)
        defined = [(w, info) for w, info in bank['words'].items() if info.get('has_definition')]
        for word, info in defined[-100:]:
            all_examples.append({
                'prompt': f"{word} is",
                'completion': info['definition'],
                'full': f"{word} is {info['definition']}"
            })
    
    random.shuffle(all_examples)
    all_examples = all_examples[:max_examples]
    
    training_lines = []
    for ex in all_examples:
        if is_clean_text(ex['full']):
            training_lines.append(ex['full'])
    
    training_text = '\n'.join(training_lines)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a') as f:
        f.write('\n\n---\n\n' + training_text)
    
    print(f"\n✅ Generated {len(training_lines)} clean training examples")
    print("\n--- Sample ---")
    for ex in all_examples[:3]:
        if is_clean_text(ex['full']):
            print(f"   {ex['prompt']} → {ex['completion'][:60]}...")

if __name__ == '__main__':
    corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    output_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    build_training_data(corpus_path, output_path)
