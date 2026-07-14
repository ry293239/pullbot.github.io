"""
Generate sentence-completion training examples from corpus.
Extracts key phrases, creates "complete the sentence" pairs.
"""

import os, sys, re, json, random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def extract_key_phrases(text, max_phrases=100):
    """Extract important words and multi-word phrases"""
    phrases = set()
    
    # Single important words (capitalized, longer words)
    words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
    phrases.update(words[:50])
    
    # Multi-word phrases (2-3 words starting with capital or common patterns)
    patterns = [
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper nouns: "Machine Learning"
        r'\b[a-z]+ [a-z]+ [a-z]+\b',       # 3-word phrases
        r'\b(artificial|neural|deep|machine|computer|binary|operating|data) [a-z]+\b',  # Tech phrases
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        phrases.update(matches[:30])
    
    return list(phrases)[:max_phrases]

def generate_completion_examples(text, phrase):
    """Generate 'complete the sentence' examples for a phrase"""
    examples = []
    
    # Find sentences containing the phrase
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if phrase.lower() in sentence.lower() and len(sentence) > 30:
            # Split at the phrase to create completion task
            idx = sentence.lower().find(phrase.lower())
            if idx > 0:
                prompt = sentence[:idx + len(phrase)].strip()
                completion = sentence[idx + len(phrase):].strip()
                
                if len(prompt) > 10 and len(completion) > 5:
                    examples.append({
                        'prompt': prompt,
                        'completion': completion,
                        'full': sentence.strip()
                    })
    
    return examples[:3]  # Max 3 examples per phrase

def build_training_data(corpus_path, output_path, max_examples=500):
    """Main function: build structured training examples"""
    print("=" * 50)
    print("📝 GENERATING TRAINING EXAMPLES")
    print("=" * 50)
    
    if not os.path.exists(corpus_path):
        print("No corpus found")
        return
    
    with open(corpus_path, 'r') as f:
        text = f.read()
    
    print(f"Corpus: {len(text):,} chars")
    
    # Extract key phrases
    phrases = extract_key_phrases(text)
    print(f"Phrases found: {len(phrases)}")
    
    # Generate completion examples
    all_examples = []
    for phrase in phrases[:50]:  # Top 50 phrases
        examples = generate_completion_examples(text, phrase)
        all_examples.extend(examples)
    
    # Also add simple word-definition pairs
    if os.path.exists(os.path.join(REPO_ROOT, 'data', 'definitions.json')):
        with open(os.path.join(REPO_ROOT, 'data', 'definitions.json')) as f:
            definitions = json.load(f)
        
        for d in definitions[-100:]:
            word = d['word']
            definition = d['definition']
            all_examples.append({
                'prompt': f"{word} is",
                'completion': definition,
                'full': f"{word} is {definition}"
            })
    
    # Shuffle and limit
    random.shuffle(all_examples)
    all_examples = all_examples[:max_examples]
    
    # Save as training corpus
    training_lines = []
    for ex in all_examples:
        training_lines.append(ex['full'])
    
    training_text = '\n'.join(training_lines)
    
    # Append to corpus
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a') as f:
        f.write('\n\n---\n\n' + training_text)
    
    print(f"\n✅ Generated {len(all_examples)} training examples")
    print(f"   Appended to {output_path}")
    
    # Show samples
    print("\n--- Sample examples ---")
    for ex in all_examples[:5]:
        print(f"   Prompt: {ex['prompt']}")
        print(f"   Completion: {ex['completion'][:80]}...")
        print()

if __name__ == '__main__':
    corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    output_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    build_training_data(corpus_path, output_path)
