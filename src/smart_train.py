"""
Smart Training System
- Only define complex words (skip "the", "and", "is" etc)
- Cap definitions to 5-10 words max
- Detect common phrases (words that appear together)
- Train model to string words/phrases into answers
"""

import os, sys, json, re, random, requests, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Words that don't need definitions
SKIP_WORDS = {
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
    'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'from',
    'they', 'this', 'that', 'with', 'will', 'each', 'about',
    'many', 'some', 'them', 'then', 'than', 'what', 'when', 'where',
    'which', 'who', 'whom', 'how', 'said', 'been', 'also', 'into',
    'its', 'may', 'other', 'new', 'just', 'like', 'over', 'after',
    'very', 'most', 'these', 'those', 'only', 'even', 'still',
    'would', 'could', 'should', 'shall', 'must', 'might', 'every',
    'any', 'such', 'more', 'make', 'made', 'know', 'take', 'see',
    'come', 'think', 'say', 'get', 'go', 'look', 'use', 'find',
    'give', 'tell', 'work', 'call', 'try', 'ask', 'need', 'feel'
}

def is_complex_word(word):
    """Only define words that actually need it"""
    if len(word) < 4:
        return False
    if word.lower() in SKIP_WORDS:
        return False
    if not word.isalpha():
        return False
    return True

def find_phrases(text, min_occurrences=2, min_length=2, max_length=4):
    """Find word sequences that frequently appear together"""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    phrases = {}
    
    for length in range(min_length, max_length + 1):
        for i in range(len(words) - length):
            phrase = ' '.join(words[i:i+length])
            if all(w in SKIP_WORDS for w in phrase.split()):
                continue
            phrases[phrase] = phrases.get(phrase, 0) + 1
    
    common_phrases = {
        phrase: count for phrase, count in phrases.items()
        if count >= min_occurrences
    }
    
    return common_phrases

def get_tiny_definition(word):
    """Get a very short definition (5-10 words max)"""
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for entry in data[:1]:
                for meaning in entry.get('meanings', [])[:1]:
                    for d in meaning.get('definitions', [])[:1]:
                        full_def = d.get('definition', '')
                        words_list = full_def.split()
                        short_def = ' '.join(words_list[:8])
                        if len(words_list) > 8:
                            short_def += '...'
                        return short_def
    except:
        pass
    return None

def build_string_training_examples(wordbank_path, corpus_path, output_path):
    """Build training examples that teach sentence construction"""
    print("=" * 50)
    print("🧠 SMART TRAINING GENERATOR")
    print("=" * 50)
    
    if not os.path.exists(wordbank_path):
        print("No wordbank found")
        return
    
    with open(wordbank_path) as f:
        bank = json.load(f)
    
    corpus = ""
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            corpus = f.read()
    
    # Find common phrases
    print("Finding common phrases...")
    phrases = {}
    if corpus:
        phrases = find_phrases(corpus, min_occurrences=2)
    top_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)[:100]
    print(f"   Found {len(phrases)} phrases, keeping top {len(top_phrases)}")
    
    # Get definitions for complex undefined words
    print("Getting short definitions for complex words...")
    complex_undefined = []
    for word, info in bank['words'].items():
        if isinstance(info, dict):
            if is_complex_word(word) and not info.get('has_definition', False):
                complex_undefined.append(word)
        else:
            # Old format - string instead of dict
            if is_complex_word(word):
                complex_undefined.append(word)
    complex_undefined = complex_undefined[:30]
    
    defined_count = 0
    for word in complex_undefined:
        definition = get_tiny_definition(word)
        if definition:
            if word not in bank['words'] or not isinstance(bank['words'].get(word), dict):
                bank['words'][word] = {}
            bank['words'][word]['has_definition'] = True
            bank['words'][word]['definition'] = definition
            defined_count += 1
        time.sleep(0.2)
    
    bank['total_defined'] = sum(
        1 for w in bank['words'].values()
        if isinstance(w, dict) and w.get('has_definition')
    )
    
    with open(wordbank_path, 'w') as f:
        json.dump(bank, f, indent=2)
    
    print(f"   Defined {defined_count} new complex words")
    print(f"   Total defined: {bank['total_defined']}")
    
    # Build training examples
    print("Building string-together examples...")
    examples = []
    
    # Collect defined words safely
    defined_words = []
    for word, info in bank['words'].items():
        if isinstance(info, dict) and info.get('has_definition'):
            definition = info.get('definition', '')
            if definition and is_complex_word(word):
                defined_words.append((word, definition))
    
    # Type 1: "Use these words to answer"
    for _ in range(50):
        if len(defined_words) < 3:
            break
        selected = random.sample(defined_words, min(3, len(defined_words)))
        words_str = ', '.join([f"{w} ({d})" for w, d in selected])
        question_templates = [
            f"Explain how {selected[0][0]} relates to {selected[-1][0]}",
            f"What is the connection between {selected[0][0]} and {selected[1][0]}?",
            f"Describe {selected[0][0]} and {selected[-1][0]}",
        ]
        question = random.choice(question_templates)
        answer = f"{selected[0][0]} is {selected[0][1]}. This relates to {selected[-1][0]} which means {selected[-1][1]}."
        examples.append(f"Using these words: {words_str}\nQuestion: {question}\nAnswer: {answer}")
    
    # Type 2: Phrase completion
    for phrase, count in top_phrases[:30]:
        words_in_phrase = phrase.split()
        if len(words_in_phrase) >= 2:
            examples.append(f"Complete the phrase: {' '.join(words_in_phrase[:-1])}... → {phrase}")
    
    # Type 3: Definition Q&A
    for word, info in bank['words'].items():
        if isinstance(info, dict) and info.get('has_definition'):
            definition = info.get('definition', '')
            if definition and is_complex_word(word):
                examples.append(f"Q: What does {word} mean?\nA: {word} is {definition}.")
        if len([e for e in examples if e.startswith('Q:')]) >= 100:
            break
    
    # Type 4: Word grouping
    if len(defined_words) >= 5:
        for _ in range(20):
            group = random.sample(defined_words, min(5, len(defined_words)))
            words_only = [w for w, d in group]
            examples.append(f"Group these words: {', '.join(words_only)}")
    
    # Save
    training_text = '\n\n---\n\n'.join(examples)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'a') as f:
        f.write('\n\n---\n\n' + training_text)
    
    print(f"\n✅ Generated {len(examples)} smart training examples")
    print(f"   Defined: {defined_count} | Phrases: {len(top_phrases)} | Q&A: {min(100, len(defined_words))}")
    
    print("\n--- Sample ---")
    for ex in examples[:3]:
        preview = ex[:120].replace('\n', ' | ')
        print(f"   {preview}...")
        print()

if __name__ == '__main__':
    wordbank_path = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
    corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    output_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    build_string_training_examples(wordbank_path, corpus_path, output_path)
