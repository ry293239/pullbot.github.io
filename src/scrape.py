"""
Pullbot Scraper
Grabs text from GitHub trending repos and Wikipedia.
Includes random sentence scraping for variety.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

import requests
import json
import yaml
from bs4 import BeautifulSoup
import time
import random
import re

config_path = os.path.join(REPO_ROOT, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def scrape_github_trending():
    """Get trending repos and their READMEs"""
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
    """Get random Wikipedia article summaries"""
    print("📚 Scraping Wikipedia summaries...")
    
    texts = []
    base_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    
    for i in range(config['scrape']['wikipedia_articles']):
        try:
            r = requests.get(base_url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                text = data.get('extract', '')
                title = data.get('title', 'unknown')
                if len(text) > 50:
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

def scrape_wikipedia_random_sentences(num_sentences=50):
    """Scrape random sentences from random Wikipedia articles.
    This gives WAY more variety than just summaries."""
    
    print("🎲 Scraping random Wikipedia sentences...")
    
    all_sentences = []
    articles_checked = 0
    max_articles = num_sentences * 3
    
    base_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    
    while len(all_sentences) < num_sentences and articles_checked < max_articles:
        try:
            r = requests.get(base_url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                title = data.get('title', 'unknown')
                extract = data.get('extract', '')
                
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', extract)
                
                # Pick 1-3 random sentences from this article
                if len(sentences) > 2:
                    num_to_take = min(random.randint(1, 3), len(sentences))
                    selected = random.sample(sentences, num_to_take)
                    
                    for sent in selected:
                        sent = sent.strip()
                        if len(sent) > 30:
                            all_sentences.append({
                                'source': f"wiki_random:{title}",
                                'text': sent,
                                'type': 'wiki_random_sentence'
                            })
                
            articles_checked += 1
            time.sleep(0.3)
            
        except Exception as e:
            articles_checked += 1
            continue
    
    print(f"  ✅ Got {len(all_sentences)} random sentences from {articles_checked} articles")
    return all_sentences

def save_raw_data(texts):
    """Save scraped text to files"""
    os.makedirs(os.path.join(REPO_ROOT, 'data', 'raw'), exist_ok=True)
    timestamp = int(time.time())
    filepath = os.path.join(REPO_ROOT, 'data', 'raw', f'scrape_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(texts, f, indent=2)
    
    # Append to corpus
    os.makedirs(os.path.join(REPO_ROOT, 'data', 'processed'), exist_ok=True)
    clean_text = '\n\n---\n\n'.join([t['text'] for t in texts])
    
    corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    with open(corpus_path, 'a') as f:
        f.write(clean_text + '\n')
    
    # Trim corpus if too large
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            content = f.read()
        max_chars = config['scrape']['max_chars']
        if len(content) > max_chars:
            content = content[-max_chars:]
            with open(corpus_path, 'w') as f:
                f.write(content)
    
    corpus_size = len(open(corpus_path).read()) if os.path.exists(corpus_path) else 0
    print(f"✅ Scrape complete! Corpus: {corpus_size:,} chars")

def run_scrape():
    all_texts = []
    
    if 'github_trending' in config['scrape']['sources']:
        all_texts.extend(scrape_github_trending())
    
    if 'wikipedia_random' in config['scrape']['sources']:
        all_texts.extend(scrape_wikipedia())
        all_texts.extend(scrape_wikipedia_random_sentences(50))
    
    # Deduplicate
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
