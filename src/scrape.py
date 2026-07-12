"""
Pullbot Scraper
Grabs text from GitHub trending repos and Wikipedia.
No API keys needed.
"""

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import yaml
from bs4 import BeautifulSoup
import time

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def scrape_github_trending():
    print("📊 Scraping GitHub trending...")
    
    url = "https://github.com/trending"
    headers = {'User-Agent': 'Pullbot/1.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        repos = []
        for article in soup.find_all('article', class_='Box-row')[:10]:
            h2 = article.find('h2')
            if h2:
                link = h2.find('a')
                if link:
                    repos.append(link['href'].strip('/'))
        
        texts = []
        for repo in repos[:config['scrape']['github_readmes']]:
            for branch in ['main', 'master']:
                readme_url = f"https://raw.githubusercontent.com/{repo}/{branch}/README.md"
                try:
                    r = requests.get(readme_url, headers=headers, timeout=15)
                    if r.status_code == 200:
                        texts.append({
                            'source': repo,
                            'text': r.text[:5000],
                            'type': 'code_readme'
                        })
                        break
                except:
                    continue
            time.sleep(0.5)
        
        print(f"  ✅ Got {len(texts)} READMEs")
        return texts
    except Exception as e:
        print(f"  ❌ GitHub scrape failed: {e}")
        return []

def scrape_wikipedia():
    print("📚 Scraping Wikipedia...")
    
    texts = []
    base_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    
    for i in range(config['scrape']['wikipedia_articles']):
        try:
            r = requests.get(base_url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                text = data.get('extract', '')
                title = data.get('title', 'unknown')
                texts.append({
                    'source': f"wiki:{title}",
                    'text': text,
                    'type': 'wiki_article'
                })
            time.sleep(0.3)
        except:
            continue
    
    print(f"  ✅ Got {len(texts)} articles")
    return texts

def save_raw_data(texts):
    os.makedirs('data/raw', exist_ok=True)
    timestamp = int(time.time())
    filepath = f"data/raw/scrape_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(texts, f, indent=2)
    
    os.makedirs('data/processed', exist_ok=True)
    clean_text = '\n\n---\n\n'.join([t['text'] for t in texts])
    
    with open('data/processed/corpus.txt', 'a') as f:
        f.write(clean_text + '\n')
    
    corpus_path = 'data/processed/corpus.txt'
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            content = f.read()
        max_chars = config['scrape']['max_chars']
        if len(content) > max_chars:
            content = content[-max_chars:]
            with open(corpus_path, 'w') as f:
                f.write(content)
    
    print(f"✅ Scrape complete! Corpus: {len(open(corpus_path).read()) if os.path.exists(corpus_path) else 0:,} chars")

def run_scrape():
    all_texts = []
    
    if 'github_trending' in config['scrape']['sources']:
        all_texts.extend(scrape_github_trending())
    
    if 'wikipedia_random' in config['scrape']['sources']:
        all_texts.extend(scrape_wikipedia())
    
    seen = set()
    unique = []
    for t in all_texts:
        key = t['text'][:100]
        if key not in seen:
            seen.add(key)
            unique.append(t)
    
    save_raw_data(unique)
    return unique

if __name__ == '__main__':
    run_scrape()
