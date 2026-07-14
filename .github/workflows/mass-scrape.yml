"""
Mass Wikipedia Word Extractor
Scrapes full articles, extracts every unique English word.
Builds a clean vocabulary wordbank.
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
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Split, lowercase, filter
    words = []
    for w in text.split():
        w = w.lower().strip()
        if len(w) > 2 and w.isalpha():
            words.append(w)
    return list(set(words))

def scrape_article():
    """Scrape one random Wikipedia article with proper headers"""
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/random/summary",
            timeout=15,
            headers={'User-Agent': 'Pullbot/1.0 (https://pullbot-ai.github.io)'}
        )
        if r.status_code == 200:
            data = r.json()
            return data.get('extract', ''), data.get('title', 'unknown')
        else:
            print(f"   HTTP {r.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
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
    errors = 0
    empty = 0
    
    for i in range(num_articles):
        text, title = scrape_article()
        
        if not text:
            empty += 1
            continue
        
        # Debug first successful article
        if i == 0 or (starting_words == 0 and new_words == 0 and i == 0):
            print(f"   First article: {title}")
            print(f"   Text length: {len(text)} chars")
            print(f"   Preview: {text[:100]}...")
        
        words = extract_words(text)
        
        # Debug first extraction
        if i == 0 and words:
            print(f"   Words extracted: {len(words)}")
            print(f"   Sample: {words[:10]}")
        
        added_this_article = 0
        for word in words:
            if word not in bank['words']:
                bank['words'][word] = {
                    'first_seen': title,
                    'has_definition': False,
                    'definition': ''
                }
                new_words += 1
                added_this_article += 1
        
        bank['total_articles'] += 1
        
        if (i + 1) % 10 == 0:
            total = len(bank['words'])
            print(f"   {i+1}/{num_articles} articles | {total:,} total words | +{new_words} new ({added_this_article} this batch) | {errors} err | {empty} empty")
        
        time.sleep(0.3)
    
    bank['total_words'] = len(bank['words'])
    save_wordbank(bank)
    
    print(f"\n✅ Done!")
    print(f"   Articles scraped: {num_articles}")
    print(f"   Words before: {starting_words:,}")
    print(f"   Words after: {len(bank['words']):,}")
    print(f"   New words added: {new_words:,}")
    print(f"   Errors: {errors}")
    print(f"   Empty articles: {empty}")
    
    # Show some words
    all_words = list(bank['words'].keys())
    if all_words:
        print(f"\n   Sample words: {all_words[:20]}")
    
    return bank

if __name__ == '__main__':
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_mass_scrape(num)
