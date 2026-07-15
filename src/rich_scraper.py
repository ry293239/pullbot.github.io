"""
Rich Training Data Generator
Converts Wikipedia articles into Q&A pairs, continuations, and raw text.
Target ratio: 40% Q&A, 30% continuations, 20% raw, 10% vocabulary
"""

import os, sys, json, time, re, random, requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def paragraph_to_qa(paragraph):
    """Convert a paragraph into Question-Answer training pairs"""
    examples = []
    
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    if len(sentences) < 1:
        return examples
    
    main_fact = sentences[0].strip()
    if len(main_fact) < 30:
        return examples
    
    # Remove leading "The", "A", "An" for question generation
    topic = re.sub(r'^(The|A|An)\s+', '', main_fact)
    topic = topic.rstrip('.')
    
    # Q&A format questions
    questions = [
        f"What is {topic.lower()}?",
        f"Explain {topic.lower()}.",
        f"Tell me about {topic.lower()}.",
        f"Describe {topic.lower()}.",
    ]
    
    for q in questions:
        examples.append({
            'text': f"Question: {q}\nAnswer: {main_fact}",
            'type': 'qa_pair',
            'source': 'wikipedia_qa'
        })
    
    # Continuation format
    words = main_fact.split()
    if len(words) > 6:
        mid = len(words) // 2
        prompt = ' '.join(words[:mid])
        examples.append({
            'text': f"Complete: {prompt}... → {main_fact}",
            'type': 'continuation',
            'source': 'wikipedia_continuation'
        })
    
    # Also keep raw paragraph
    examples.append({
        'text': paragraph,
        'type': 'raw_text',
        'source': 'wikipedia_raw'
    })
    
    return examples

def paragraph_to_continuations(paragraph):
    """Generate multiple continuation examples from a paragraph"""
    examples = []
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    
    for i in range(len(sentences) - 1):
        prompt = ' '.join(sentences[:i+1])
        completion = ' '.join(sentences[i+1:i+3])
        if len(prompt) > 20 and len(completion) > 10:
            examples.append({
                'text': f"Continue: {prompt} → {completion}",
                'type': 'continuation',
                'source': 'wikipedia_continuation'
            })
    
    return examples

def scrape_wikipedia_rich(num_articles=15):
    """Scrape Wikipedia and generate rich training examples"""
    print("=" * 50)
    print("📚 RICH WIKIPEDIA Q&A SCRAPER")
    print(f"   Target: {num_articles} articles")
    print("=" * 50)
    
    all_examples = []
    
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
            
            for para in paragraphs[:5]:
                # 40% Q&A
                qa_examples = paragraph_to_qa(para)
                all_examples.extend(qa_examples[:4])  # Keep ratio
                
                # 30% Continuations
                cont_examples = paragraph_to_continuations(para)
                all_examples.extend(cont_examples[:2])
            
            if (i + 1) % 5 == 0:
                print(f"   {i+1}/{num_articles} articles → {len(all_examples)} examples")
            
            time.sleep(0.3)
            
        except Exception as e:
            continue
    
    print(f"   ✅ Generated {len(all_examples)} total examples")
    
    # Show breakdown
    types = {}
    for ex in all_examples:
        t = ex.get('type', 'unknown')
        types[t] = types.get(t, 0) + 1
    for t, c in types.items():
        print(f"      {t}: {c} ({c/len(all_examples)*100:.0f}%)")
    
    return all_examples

def scrape_documentation():
    """Scrape programming documentation as Q&A"""
    print("\n📖 DOCUMENTATION Q&A")
    
    docs = [
        ("Python functions", "A function is a reusable block of code that performs a specific task. Functions help organize code and avoid repetition."),
        ("Python lists", "A list is an ordered collection of items in Python. Lists are mutable, meaning you can change them after creation."),
        ("Python dictionaries", "A dictionary stores key-value pairs. Each key is unique and maps to a specific value, like a real dictionary maps words to definitions."),
        ("Flask routes", "A route in Flask connects a URL to a Python function. When someone visits that URL, the function runs and returns a response."),
        ("HTML elements", "HTML elements are the building blocks of web pages. Each element has an opening tag, content, and a closing tag."),
        ("CSS selectors", "CSS selectors target HTML elements for styling. You can select by tag name, class, ID, or other attributes."),
        ("JavaScript variables", "Variables in JavaScript store data values. You can declare them with let, const, or var."),
        ("Git commits", "A commit in Git saves changes to your repository. Each commit has a unique hash and a message describing the changes."),
    ]
    
    examples = []
    for topic, content in docs:
        examples.append({'text': f"Question: What are {topic.lower()}?\nAnswer: {content}", 'type': 'qa_pair'})
        examples.append({'text': f"Question: Explain {topic.lower()}.\nAnswer: {content}", 'type': 'qa_pair'})
        examples.append({'text': content, 'type': 'raw_text'})
    
    print(f"   ✅ Generated {len(examples)} documentation examples")
    return examples

def generate_vocab_qa():
    """Generate Q&A pairs from wordbank"""
    print("\n📖 VOCABULARY Q&A")
    
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
    
    for word, info in defined[:200]:
        definition = info['definition']
        
        examples.append({
            'text': f"Question: What is {word}?\nAnswer: {word} is {definition}.",
            'type': 'qa_pair'
        })
        examples.append({
            'text': f"Question: Define {word}.\nAnswer: {definition}",
            'type': 'qa_pair'
        })
        examples.append({
            'text': f"Complete: {word} is... → {word} is {definition}.",
            'type': 'continuation'
        })
    
    print(f"   ✅ Generated {len(examples)} vocabulary examples")
    return examples

def save_rich_corpus(examples):
    """Save rich training data to corpus"""
    corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
    
    random.shuffle(examples)
    text = '\n\n---\n\n'.join([ex['text'] for ex in examples])
    
    with open(corpus_path, 'a') as f:
        f.write('\n\n---\n\n' + text)
    
    print(f"\n💾 Saved {len(examples)} examples to corpus")
    return len(examples)

def run_rich_scrape():
    all_examples = []
    
    # 40% Q&A + continuations from Wikipedia
    all_examples.extend(scrape_wikipedia_rich(15))
    
    # 30% Documentation Q&A
    all_examples.extend(scrape_documentation())
    
    # 20% Vocabulary Q&A
    all_examples.extend(generate_vocab_qa())
    
    save_rich_corpus(all_examples)
    
    print(f"\n🎉 Total: {len(all_examples)} rich training examples generated!")

if __name__ == '__main__':
    run_rich_scrape()
