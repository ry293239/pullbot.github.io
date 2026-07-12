"""
Pullbot Scraper
Grabs text from GitHub trending repos and Wikipedia.
No API keys needed. All public sources.
"""

import requests
import json
import os
import yaml
from bs4 import BeautifulSoup
import time
import re

# Load config
with open('config.yaml', 'r') as f:
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
        articles = soup.find_all('article', class_='Box-row')[:10]
        
        for article in articles:
            h2 = article.find('h2')
            if h2:
                link = h2.find('a')
                if link:
                    repo_path = link['href'].strip('/')
                    repos.append(repo_path)
        
        texts = []
        readme_limit = config['scrape']['github_readmes']
        
        for repo in repos[:readme_limit]:
            # Get README raw content
            readme_url = f"https://raw.githubusercontent.com/{repo}/main/README.md"
            try:
                r = requests.get(readme_url, headers=headers, timeout=15)
                if r.status_code == 200:
                    texts.append({
                        'source': repo,
                        'text': r.text[:5000],  # Cap at 5k chars per README
                        'type': 'code_readme'
                    })
                else:
                    # Try master branch
                    readme_url = f"https://raw.githubusercontent.com/{repo}/master/README.md"
                    r = requests.get(readme_url, headers=headers, timeout=15)
                    if r.status_code == 200:
                        texts.append({
                            'source': repo,
                            'text': r.text[:5000],
                            'type': 'code_readme'
                        })
            except:
                continue
            
            time.sleep(1)  # Be polite
        
        print(f"  ✅ Got {len(texts)} READMEs")
        return texts
        
    except Exception as e:
        print(f"  ❌ GitHub scrape failed: {e}")
        return []

def scrape_wikipedia():
    """Get random Wikipedia articles"""
    print("📚 Scraping Wikipedia...")
    
    texts = []
    base_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    
    for i in range(config['scrape']['wikipedia_random']):
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
                
                # Follow links for depth
                if config['scrape']['wikipedia_depth'] > 1:
                    page_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
                    # Add related article titles for next scrape
                
            time.sleep(0.5)
        except Exception as e:
            continue
    
    print(f"  ✅ Got {len(texts)} articles")
    return texts

def save_raw_data(texts, output_dir="data/raw"):
    """Save scraped text to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = int(time.time())
    filepath = f"{output_dir}/scrape_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(texts, f, indent=2)
    
    # Check total size and prune old files
    total_chars = sum(len(t['text']) for t in texts)
    print(f"  💾 Saved {total_chars:,} chars to {filepath}")
    
    # If we have too much data, delete oldest files
    files = sorted(os.listdir(output_dir))
    while len(files) > 20:  # Keep last 20 scrapes
        os.remove(f"{output_dir}/{files[0]}")
        files.pop(0)

def run_scrape():
    """Main scrape function"""
    all_texts = []
    
    if config['scrape']['github_trending']:
        all_texts.extend(scrape_github_trending())
    
    if config['scrape']['wikipedia_random'] > 0:
        all_texts.extend(scrape_wikipedia())
    
    # Deduplicate
    seen = set()
    unique_texts = []
    for t in all_texts:
        key = t['text'][:100]  # First 100 chars as dedup key
        if key not in seen:
            seen.add(key)
            unique_texts.append(t)
    
    save_raw_data(unique_texts)
    
    # Also save as clean text file for training
    os.makedirs('data/processed', exist_ok=True)
    clean_text = '\n\n---\n\n'.join([t['text'] for t in unique_texts])
    
    with open('data/processed/corpus.txt', 'a') as f:
        f.write(clean_text + '\n')
    
    # Trim corpus if too large
    max_chars = config['scrape']['max_total_chars']
    corpus_path = 'data/processed/corpus.txt'
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            content = f.read()
        if len(content) > max_chars:
            content = content[-max_chars:]  # Keep last max_chars
            with open(corpus_path, 'w') as f:
                f.write(content)
    
    print(f"\n✅ Scrape complete! Total corpus: {len(open(corpus_path).read()) if os.path.exists(corpus_path) else 0:,} chars")
    return unique_texts

if __name__ == '__main__':
    run_scrape()
