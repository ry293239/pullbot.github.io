"""
Rich Training Data Generator
Wikipedia → Summarize → Generate Questions → Store Everything
Documentation → Extract examples
Q&A pairs → Conversational training
"""

import os, sys, json, time, re, random, requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def scrape_wikipedia_rich(num_articles=10):
    """Scrape Wikipedia and generate rich training examples"""
    print("=" * 50)
    print("📚 RICH WIKIPEDIA SCRAPER")
    print("=" * 50)
    
    examples = []
    
    for i in range(num_articles):
        try:
            r = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/random/summary",
                timeout=15,
                headers={'User-Agent': 'Pullbot/1.0'}
            )
            if r.status_code != 200:
                continue
            
            data = r.json()
            title = data.get('title', 'Unknown')
            extract = data.get('extract', '')
            
            if len(extract) < 100:
                continue
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in extract.split('\n') if len(p.strip()) > 50]
            
            for para in paragraphs[:3]:
                # 1. Original paragraph
                examples.append(f"{para}")
                
                # 2. Generate Q&A from paragraph
                sentences = re.split(r'(?<=[.!?])\s+', para)
                if len(sentences) >= 2:
                    first_sent = sentences[0].strip()
                    if len(first_sent) > 30:
                        # Turn the first sentence into a question
                        q = first_sent.rstrip('.')
                        q = re.sub(r'^(The|A|An)\s+', '', q)
                        q = q[0].lower() + q[1:] if q else q
                        
                        examples.append(f"Q: What is {q}?\nA: {first_sent}")
                        examples.append(f"Q: Explain {q}.\nA: {para[:300]}")
                
                # 3. Sentence completion
                words = para.split()
                if len(words) > 10:
                    mid = len(words) // 2
                    prompt = ' '.join(words[:mid])
                    examples.append(f"Complete: {prompt}... → {para}")
            
            if (i + 1) % 3 == 0:
                print(f"   {i+1}/{num_articles} articles processed")
            
            time.sleep(0.3)
            
        except Exception as e:
            continue
    
    print(f"   Generated {len(examples)} rich training examples")
    return examples

def scrape_documentation():
    """Scrape programming documentation"""
    print("\n📖 SCRAPING DOCUMENTATION")
    
    examples = []
    
    # Python docs snippets
    docs = [
        ("Python functions", "A function is a block of organized, reusable code that performs a single action. Functions provide better modularity and code reusability."),
        ("Python lists", "A list is a collection which is ordered and changeable. Lists allow duplicate members and are created using square brackets."),
        ("Python dictionaries", "A dictionary is a collection which is unordered, changeable, and indexed. Dictionaries have keys and values."),
        ("Flask routes", "A route in Flask maps a URL to a function. The @app.route() decorator tells Flask which URL triggers the function."),
        ("HTML basics", "HTML is the standard markup language for web pages. HTML elements are the building blocks of web pages."),
    ]
    
    for topic, content in docs:
        examples.append(f"Topic: {topic}\n{content}")
        examples.append(f"Q: What is {topic.lower()}?\nA: {content}")
        examples.append(f"Q: How do I use {topic.lower()}?\nA: {content}")
        examples.append(f"Complete: {topic} is... → {content}")
    
    print(f"   Generated {len(examples)} documentation examples")
    return examples

def generate_qa_from_wordbank():
    """Generate Q&A pairs from the wordbank"""
    print("\n💬 GENERATING Q&A FROM WORDBANK")
    
    wordbank_path = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
    if not os.path.exists(wordbank_path):
        print("   No wordbank found")
        return []
    
    with open(wordbank_path) as f:
        bank = json.load(f)
    
    examples = []
    defined = [
        (word, info) for word, info in bank['words'].items()
        if isinstance(info, dict) and info.get('has_definition')
    ]
    
    for word, info in defined[:100]:
        definition = info['definition']
        
        # Multiple question formats
        examples.append(f"Q: What is {word}?\nA: {word} is {definition}.")
        examples.append(f"Q: Define {word}.\nA: {definition}")
        examples.append(f"Q: Explain {word} in simple terms.\nA: {word} means {definition}")
        examples.append(f"Complete: {word} is... → {word} is {definition}.")
    
    print(f"   Generated {len(examples)} Q&A pairs")
    return examples

def save_rich_corpus(examples):
    """Save rich training data to corpus"""
    corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    
    text = '\n\n---\n\n'.join(examples)
    
    with open(corpus_path, 'a') as f:
        f.write('\n\n---\n\n' + text)
    
    print(f"\n✅ Saved {len(examples)} rich examples to corpus")
    return len(examples)

def run_rich_scrape():
    all_examples = []
    
    all_examples.extend(scrape_wikipedia_rich(10))
    all_examples.extend(scrape_documentation())
    all_examples.extend(generate_qa_from_wordbank())
    
    # Shuffle for variety
    random.shuffle(all_examples)
    
    save_rich_corpus(all_examples)
    
    print(f"\n🎉 Total: {len(all_examples)} rich training examples generated!")

if __name__ == '__main__':
    run_rich_scrape()
