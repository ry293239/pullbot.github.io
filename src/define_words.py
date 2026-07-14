"""
Definition Lookup for Wordbank
Takes words from wordbank, looks up definitions.
Stores word + definition pairs for training.
"""

import os, sys, json, time, requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

WORD_BANK_PATH = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
DEFINITIONS_PATH = os.path.join(REPO_ROOT, 'data', 'definitions.json')

def load_wordbank():
    with open(WORD_BANK_PATH, 'r') as f:
        return json.load(f)

def save_wordbank(bank):
    with open(WORD_BANK_PATH, 'w') as f:
        json.dump(bank, f, indent=2)

def load_definitions():
    if os.path.exists(DEFINITIONS_PATH):
        with open(DEFINITIONS_PATH, 'r') as f:
            return json.load(f)
    return []

def save_definitions(defs):
    os.makedirs(os.path.dirname(DEFINITIONS_PATH), exist_ok=True)
    with open(DEFINITIONS_PATH, 'w') as f:
        json.dump(defs, f, indent=2)

def lookup_word(word):
    """Look up definition for a single word"""
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for entry in data[:1]:
                for meaning in entry.get('meanings', [])[:1]:
                    for d in meaning.get('definitions', [])[:1]:
                        return d.get('definition', '')
    except:
        pass
    return None

def run_define_words(limit=100):
    """Look up definitions for words that don't have them yet"""
    print("=" * 50)
    print(f"📖 DEFINITION LOOKUP")
    print(f"   Limit: {limit} words")
    print("=" * 50)
    
    bank = load_wordbank()
    definitions = load_definitions()
    defined_count = 0
    
    # Find words without definitions
    undefined = [
        (word, info) for word, info in bank['words'].items()
        if not info.get('has_definition', False)
    ][:limit]
    
    print(f"   Undefined words: {len(undefined):,}")
    
    for i, (word, info) in enumerate(undefined):
        definition = lookup_word(word)
        
        if definition:
            bank['words'][word]['has_definition'] = True
            bank['words'][word]['definition'] = definition
            
            definitions.append({
                'word': word,
                'definition': definition,
                'first_seen': info.get('first_seen', 'unknown')
            })
            
            defined_count += 1
            
            if defined_count % 20 == 0:
                print(f"   {defined_count}/{len(undefined)} defined")
        
        time.sleep(0.2)
    
    save_wordbank(bank)
    save_definitions(definitions)
    
    print(f"\n✅ Defined {defined_count} words")
    print(f"   Total definitions: {len(definitions):,}")
    return definitions

if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_define_words(limit)
