"""
Definition Lookup - Rich Vocabulary Builder
Stores definitions with examples, related words, and generated questions.
Creates multiple training examples per word.
"""

import os, sys, json, time, requests, random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORD_BANK_PATH = os.path.join(REPO_ROOT, 'data', 'wordbank.json')

def load_wordbank():
    if os.path.exists(WORD_BANK_PATH):
        with open(WORD_BANK_PATH, 'r') as f:
            return json.load(f)
    return {"words": {}, "total_articles": 0, "total_words": 0, "total_defined": 0}

def save_wordbank(bank):
    os.makedirs(os.path.dirname(WORD_BANK_PATH), exist_ok=True)
    with open(WORD_BANK_PATH, 'w') as f:
        json.dump(bank, f, indent=2)

def lookup_word(word):
    """Look up definition for a single word"""
    try:
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for entry in data[:1]:
                for meaning in entry.get('meanings', [])[:1]:
                    part_of_speech = meaning.get('partOfSpeech', '')
                    for d in meaning.get('definitions', [])[:1]:
                        definition = d.get('definition', '')
                        example = d.get('example', '')
                        return definition, part_of_speech, example
    except:
        pass
    return None, None, None

def generate_examples(word, definition):
    """Generate example sentences using the word"""
    templates = [
        f"{word.capitalize()} is an important concept in its field.",
        f"Scientists study {word} to understand more about the world.",
        f"The term {word} refers to {definition[:50]}.",
        f"{word.capitalize()} can be explained as {definition[:60]}.",
    ]
    return templates

def generate_questions(word):
    """Generate questions about the word"""
    return [
        f"What is {word}?",
        f"Why is {word} important?",
        f"Can you explain {word}?",
        f"How does {word} work?",
    ]

def find_related_words(word, bank, max_related=5):
    """Find words with similar definitions"""
    if word not in bank['words']:
        return []
    
    current_def = bank['words'][word].get('definition', '')
    if not current_def:
        return []
    
    related = []
    current_words = set(current_def.lower().split())
    
    for other_word, info in bank['words'].items():
        if other_word == word:
            continue
        if not isinstance(info, dict):
            continue
        
        other_def = info.get('definition', '')
        if not other_def:
            continue
        
        other_words = set(other_def.lower().split())
        overlap = len(current_words & other_words)
        
        if overlap >= 3:
            related.append((other_word, overlap))
    
    related.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in related[:max_related]]

def create_rich_entry(word, definition, part_of_speech, example):
    """Create a rich vocabulary entry"""
    return {
        "definition": definition,
        "part_of_speech": part_of_speech or "unknown",
        "has_definition": True,
        "examples": generate_examples(word, definition),
        "related": [],
        "questions": generate_questions(word),
        "original_example": example or "",
    }

def upgrade_old_entry(word, existing_info):
    """Upgrade old format entries to rich format"""
    if isinstance(existing_info, dict) and 'definition' in existing_info:
        definition = existing_info.get('definition', '')
        return create_rich_entry(word, definition, '', '')
    
    if isinstance(existing_info, str):
        return create_rich_entry(word, existing_info, '', '')
    
    return create_rich_entry(word, '', '', '')

def run_define_words(limit=100):
    """Look up definitions and create rich entries"""
    print("=" * 50)
    print(f"📖 RICH DEFINITION LOOKUP (limit: {limit})")
    print("=" * 50)
    
    bank = load_wordbank()
    
    # Upgrade old entries first
    upgraded = 0
    for word, info in bank['words'].items():
        if isinstance(info, str) or (isinstance(info, dict) and 'examples' not in info):
            bank['words'][word] = upgrade_old_entry(word, info)
            upgraded += 1
    
    if upgraded > 0:
        print(f"   Upgraded {upgraded} old entries to rich format")
    
    # Find undefined words
    undefined = [
        word for word, info in bank['words'].items()
        if isinstance(info, dict) and not info.get('has_definition', False)
    ][:limit]
    
    # Also try to define words with short definitions
    short_defs = [
        word for word, info in bank['words'].items()
        if isinstance(info, dict) and info.get('has_definition') and len(info.get('definition', '')) < 20
    ][:limit // 2]
    
    all_to_define = undefined + short_defs
    defined_count = 0
    
    print(f"   Undefined: {len(undefined)} | Short defs: {len(short_defs)}")
    
    for word in all_to_define[:limit]:
        definition, pos, example = lookup_word(word)
        
        if definition:
            bank['words'][word] = create_rich_entry(word, definition, pos, example)
            defined_count += 1
            
            if defined_count % 20 == 0:
                print(f"   {defined_count}/{len(all_to_define[:limit])} defined")
        
        time.sleep(0.2)
    
    # Find related words
    print(f"\n   Finding related words...")
    relation_count = 0
    defined_words = [
        word for word, info in bank['words'].items()
        if isinstance(info, dict) and info.get('has_definition') and not info.get('related')
    ][:50]
    
    for word in defined_words:
        related = find_related_words(word, bank)
        if related:
            bank['words'][word]['related'] = related
            relation_count += 1
    
    # Update totals
    bank['total_defined'] = sum(
        1 for w in bank['words'].values()
        if isinstance(w, dict) and w.get('has_definition')
    )
    bank['total_words'] = len(bank['words'])
    
    save_wordbank(bank)
    
    print(f"\n✅ Defined: {defined_count} | Relations found: {relation_count}")
    print(f"   Total defined: {bank['total_defined']:,} / {bank['total_words']:,}")
    
    # Show sample rich entry
    defined = [(w, i) for w, i in bank['words'].items() if isinstance(i, dict) and i.get('has_definition')]
    if defined:
        word, info = defined[-1]
        print(f"\n   Sample rich entry for '{word}':")
        print(f"   Definition: {info.get('definition', 'N/A')[:80]}...")
        print(f"   Questions: {info.get('questions', [])[:2]}")
        print(f"   Related: {info.get('related', [])[:3]}")
    
    return defined_count

if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_define_words(limit)
