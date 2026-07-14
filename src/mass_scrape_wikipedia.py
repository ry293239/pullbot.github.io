"""
Mass Wikipedia Word Extractor
Scrapes full articles, extracts every unique word.
Builds a clean English wordbank.
"""

import os, sys, json, time, re, requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

WORD_BANK_PATH = os.path.join(REPO_ROOT, 'data', 'wordbank.json')

def load_wordbank():
    if os.path.exists(WORD_BANK_PATH):
        with open(WORD_BANK_PATH, 'r') as f:
            return json.load(f)
    return {"words": {}, "total_articles": 0, "total_words": 0}

def save_wordbank(bank):
    os.makedirs(os.path.dirname(WORD_BANK_PATH), exist_ok=True)
    with open(WORD_BANK_PATH, 'w') as f:
        json.dump(bank, f, indent=2)

def extract_words(text):
    """Extract clean English words from text"""
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Split into words, lowercase, filter short words
    words = [w.lower() for w in text.split() if len(w) > 2 and w.isalpha()]
    return list(set(words))  # Unique only

def scrape_article():
    """Scrape one random Wikipedia article"""
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/random/summary",
            timeout=15
        )
        if r.status_code == 200:
            data = r.json()
            return data.get('extract', ''), data.get('title', 'unknown')
    except:
        pass
    return "", ""

def run_mass_scrape(num_articles=50):
    """Scrape multiple articles, extract all unique words"""
    print("=" * 50)
    print(f"📚 MASS WIKIPEDIA WORD EXTRACTOR")
    print(f"   Target: {num_articles} articles")
    print("=" * 50)
    
    bank = load_wordbank()
    starting_words = len(bank['words'])
    new_words = 0
    
    for i in range(num_articles):
        text, title = scrape_article()
        if not text:
            continue
        
        words = extract_words(text)
        
        for word in words:
            if word not in bank['words']:
                bank['words'][word] = {
                    'first_seen': title,
                    'has_definition': False,
                    'definition': ''
                }
                new_words += 1
        
        bank['total_articles'] += 1
        bank['total_words'] = len(bank['words'])
        
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{num_articles} articles | {len(bank['words']):,} total words | +{new_words} new")
        
        time.sleep(0.3)
    
    save_wordbank(bank)
    
    print(f"\n✅ Done!")
    print(f"   Articles scraped: {num_articles}")
    print(f"   Words before: {starting_words:,}")
    print(f"   Words after: {len(bank['words']):,}")
    print(f"   New words added: {new_words:,}")
    return bank

if __name__ == '__main__':
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_mass_scrape(num)
