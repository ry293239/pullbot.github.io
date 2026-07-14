"""
Mass Wikipedia Word Extractor + Auto-Definer
Scrapes full articles, extracts every unique word,
immediately looks up definitions for new words.
"""

import os, sys, json, time, re, requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

WORD_BANK_PATH = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
DEFINITIONS_PATH = os.path.join(REPO_ROOT, 'data', 'definitions.json')

def load_wordbank():
    if os.path.exists(WORD_BANK_PATH):
        with open(WORD_BANK_PATH, 'r') as f:
            return json.load(f)
    return {"words": {}, "total_articles": 0, "total_words": 0}

def save_wordbank(bank):
    os.makedirs(os.path.dirname(WORD_BANK_PATH), exist_ok=True)
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

def extract_words(text):
    """Extract clean English words from text"""
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = []
    for w in text.split():
        w = w.lower().strip()
        if len(w) > 2 and w.isalpha():
            words.append(w)
    return list(set(words))

def scrape_article():
    """Scrape one random Wikipedia article"""
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/random/summary",
            timeout=15,
            headers={'User-Agent': 'Pullbot/1.0 (https://pullbot-ai.github.io)'}
        )
        if r.status_code == 200:
            data = r.json()
            return data.get('extract', ''), data.get('title', 'unknown')
    except:
        pass
    return "", ""

def lookup_definition(word):
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

def process_article(text, title, bank, definitions, article_num):
    """Process one article: extract words, add new ones, define them"""
    
    # Extract all words from article
    all_words = extract_words(text)
    
    new_words = []
    skipped = 0
    
    for word in all_words:
        if word not in bank['words']:
            # New word! Add it
            bank['words'][word] = {
                'first_seen': title,
                'has_definition': False,
                'definition': ''
            }
            new_words.append(word)
        else:
            skipped += 1
    
    # Look up definitions for new words
    defined_count = 0
    for word in new_words[:20]:  # Max 20 definitions per article
        definition = lookup_definition(word)
        if definition:
            bank['words'][word]['has_definition'] = True
            bank['words'][word]['definition'] = definition
            definitions.append({
                'word': word,
                'definition': definition,
                'first_seen': title
            })
            defined_count += 1
        time.sleep(0.2)  # Rate limit for dictionary API
    
    return new_words, skipped, defined_count

def run_mass_scrape(num_articles=50):
    """Scrape articles, extract words, define new ones"""
    print("=" * 50)
    print(f"📚 MASS WIKIPEDIA WORD EXTRACTOR + DEFINER")
    print(f"   Target: {num_articles} articles")
    print("=" * 50)
    
    bank = load_wordbank()
    definitions = load_definitions()
    
    starting_words = len(bank['words'])
    total_new = 0
    total_skipped = 0
    total_defined = 0
    errors = 0
    
    for i in range(num_articles):
        text, title = scrape_article()
        
        if not text:
            errors += 1
            continue
        
        # Debug first article
        if i == 0:
            print(f"   First article: {title}")
            print(f"   Length: {len(text)} chars")
        
        new_words, skipped, defined = process_article(text, title, bank, definitions, i)
        
        total_new += len(new_words)
        total_skipped += skipped
        total_defined += defined
        bank['total_articles'] += 1
        
        # Progress every 5 articles
        if (i + 1) % 5 == 0:
            total = len(bank['words'])
            print(f"   {i+1}/{num_articles} | {total:,} words | +{total_new} new | {total_defined} defined | {errors} err")
        
        time.sleep(0.3)
    
    bank['total_words'] = len(bank['words'])
    save_wordbank(bank)
    save_definitions(definitions)
    
    print(f"\n✅ Done!")
    print(f"   Articles: {num_articles}")
    print(f"   Words before: {starting_words:,}")
    print(f"   Words after: {len(bank['words']):,}")
    print(f"   New words added: {total_new:,}")
    print(f"   Words already known: {total_skipped:,}")
    print(f"   New definitions: {total_defined:,}")
    print(f"   Errors: {errors}")
    
    # Show sample new words
    all_words = list(bank['words'].keys())
    if all_words:
        print(f"\n   Latest words: {all_words[-20:]}")

if __name__ == '__main__':
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_mass_scrape(num)
