"""
Pullbot Scraper
Grabs text from GitHub trending, Wikipedia, open source datasets, 
English dictionary, and vocabulary builders.
Cleans all text before saving.
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
import csv
import io

config_path = os.path.join(REPO_ROOT, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text):
    """Remove markdown, HTML, URLs, badges, and code blocks"""
    if not text:
        return ""
    
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
    text = re.sub(r'#{1,6}\s', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^>\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[=\-*#]{3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\|.+\|$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text

# ============================================
# SCRAPERS
# ============================================

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
                        cleaned = clean_text(r.text[:3000])
                        if len(cleaned) > 50:
                            texts.append({
                                'source': repo,
                                'text': cleaned,
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
    print("📚 Scraping Wikipedia summaries...")
    texts = []
    base_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    
    for i in range(config['scrape']['wikipedia_articles']):
        try:
            r = requests.get(base_url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                text = clean_text(data.get('extract', ''))
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
                extract = clean_text(data.get('extract', ''))
                sentences = re.split(r'(?<=[.!?])\s+', extract)
                
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
        except:
            articles_checked += 1
            continue
    
    print(f"  ✅ Got {len(all_sentences)} random sentences")
    return all_sentences

def scrape_chatgpt_prompts():
    print("💬 Scraping ChatGPT prompts dataset...")
    texts = []
    
    try:
        url = "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv"
        r = requests.get(url, timeout=30)
        
        if r.status_code == 200:
            reader = csv.reader(io.StringIO(r.text))
            next(reader)
            
            for row in reader:
                if len(row) >= 2:
                    prompt = row[1].strip()
                    if len(prompt) > 30:
                        texts.append({
                            'source': 'chatgpt_prompts',
                            'text': f"User asked: {prompt}",
                            'type': 'conversation_template'
                        })
            
            print(f"  ✅ Got {len(texts)} prompt templates")
    except Exception as e:
        print(f"  ⚠️ ChatGPT prompts failed: {e}")
    
    return texts

def scrape_tiny_shakespeare():
    print("🎭 Scraping Shakespeare dataset...")
    texts = []
    
    try:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        r = requests.get(url, timeout=30)
        
        if r.status_code == 200:
            full_text = clean_text(r.text)
            chunk_size = 500
            num_chunks = min(10, len(full_text) // chunk_size)
            
            for _ in range(num_chunks):
                start = random.randint(0, max(0, len(full_text) - chunk_size))
                chunk = full_text[start:start + chunk_size].strip()
                if len(chunk) > 50:
                    texts.append({
                        'source': 'shakespeare',
                        'text': chunk,
                        'type': 'literature'
                    })
            
            print(f"  ✅ Got {len(texts)} Shakespeare chunks")
    except Exception as e:
        print(f"  ⚠️ Shakespeare failed: {e}")
    
    return texts

def scrape_github_ai_repos():
    print("🤖 Scraping AI/ML repo READMEs...")
    texts = []
    
    ai_repos = [
        "huggingface/transformers",
        "pytorch/pytorch",
        "tensorflow/tensorflow",
        "scikit-learn/scikit-learn",
        "openai/openai-python",
        "langchain-ai/langchain",
        "thomasahle/sunfish",
        "official-stockfish/Stockfish",
    ]
    
    headers = {'User-Agent': 'Pullbot/1.0'}
    
    for repo in random.sample(ai_repos, min(4, len(ai_repos))):
        for branch in ['main', 'master']:
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/README.md"
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    cleaned = clean_text(r.text[:2000])
                    if len(cleaned) > 50:
                        texts.append({
                            'source': f"ai_repo:{repo}",
                            'text': f"Open source project {repo}: {cleaned}",
                            'type': 'ai_readme'
                        })
                    break
            except:
                continue
        time.sleep(0.5)
    
    print(f"  ✅ Got {len(texts)} AI READMEs")
    return texts

def scrape_full_dictionary():
    """Download English dictionary for vocabulary building"""
    print("📚 Downloading English dictionary...")
    texts = []
    
    # Try multiple dictionary sources
    dict_urls = [
        "https://raw.githubusercontent.com/adambom/dictionary/master/dictionary.json",
        "https://raw.githubusercontent.com/matthewreagan/WebstersEnglishDictionary/master/dictionary.json"
    ]
    
    for url in dict_urls:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                data = r.json()
                words = list(data.items())[:2000]  # Take 2000 words
                
                for word, definition in words:
                    if isinstance(definition, str) and len(definition) > 15:
                        clean_def = clean_text(definition[:250])
                        texts.append({
                            'source': f'dict:{word}',
                            'text': f'{word}: {clean_def}',
                            'type': 'vocabulary'
                        })
                
                print(f"  ✅ Got {len(texts)} dictionary entries")
                return texts
        except:
            continue
    
    # Fallback: common word definitions via API
    print("  ⚠️ Full dictionary failed, trying word-by-word...")
    return scrape_word_definitions()

def scrape_word_definitions():
    """Scrape individual word definitions"""
    texts = []
    
    # Common English words across categories
    words = [
        # Tech
        "algorithm", "function", "variable", "data", "network", "system",
        "process", "method", "object", "class", "code", "program",
        # Science
        "energy", "matter", "force", "cell", "atom", "species",
        "theory", "experiment", "observe", "calculate", "measure",
        # General
        "create", "develop", "design", "implement", "analyze",
        "understand", "explain", "describe", "compare", "evaluate",
        "learn", "teach", "write", "read", "think", "know",
        "language", "communication", "information", "knowledge"
    ]
    
    for word in words:
        try:
            url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                for entry in data[:1]:
                    for meaning in entry.get('meanings', [])[:1]:
                        part_of_speech = meaning.get('partOfSpeech', '')
                        for definition in meaning.get('definitions', [])[:1]:
                            def_text = definition.get('definition', '')
                            if def_text:
                                texts.append({
                                    'source': f'dict:{word}',
                                    'text': f'{word} ({part_of_speech}): {def_text}',
                                    'type': 'vocabulary'
                                })
            time.sleep(0.2)
        except:
            continue
    
    print(f"  ✅ Got {len(texts)} word definitions")
    return texts

def scrape_wikipedia_full_articles(num_articles=2):
    print("📖 Scraping full Wikipedia articles...")
    texts = []
    
    for _ in range(num_articles * 2):
        try:
            r = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                timeout=15
            )
            if r.status_code == 200:
                title = r.json().get('title', '')
                if title:
                    full_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&explaintext=true&format=json"
                    fr = requests.get(full_url, timeout=15)
                    if fr.status_code == 200:
                        pages = fr.json().get('query', {}).get('pages', {})
                        for page in pages.values():
                            extract = clean_text(page.get('extract', ''))
                            if len(extract) > 500:
                                texts.append({
                                    'source': f"wiki_full:{title}",
                                    'text': extract[:3000],
                                    'type': 'wiki_full_article'
                                })
                                break
            time.sleep(0.5)
        except:
            continue
        
        if len(texts) >= num_articles:
            break
    
    print(f"  ✅ Got {len(texts)} full articles")
    return texts

# ============================================
# SAVE
# ============================================

def save_raw_data(texts):
    os.makedirs(os.path.join(REPO_ROOT, 'data', 'raw'), exist_ok=True)
    timestamp = int(time.time())
    filepath = os.path.join(REPO_ROOT, 'data', 'raw', f'scrape_{timestamp}.json')
    
    with open(filepath, 'w') as f:
        json.dump(texts, f, indent=2)
    
    os.makedirs(os.path.join(REPO_ROOT, 'data', 'processed'), exist_ok=True)
    clean_text = '\n\n---\n\n'.join([t['text'] for t in texts])
    
    corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    with open(corpus_path, 'a') as f:
        f.write(clean_text + '\n')
    
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
    
    # GitHub and Wikipedia
    if 'github_trending' in config['scrape']['sources']:
        all_texts.extend(scrape_github_trending())
    
    if 'wikipedia_random' in config['scrape']['sources']:
        all_texts.extend(scrape_wikipedia())
        all_texts.extend(scrape_wikipedia_random_sentences(30))
    
    # Vocabulary building (NEW - makes Pullbot understand words)
    all_texts.extend(scrape_full_dictionary())
    
    # Language and conversation
    all_texts.extend(scrape_chatgpt_prompts())
    all_texts.extend(scrape_tiny_shakespeare())
    
    # Tech knowledge
    all_texts.extend(scrape_github_ai_repos())
    
    # Deep knowledge
    all_texts.extend(scrape_wikipedia_full_articles(2))
    
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
