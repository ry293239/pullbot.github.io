"""
Generate high-quality Q&A training data using GitHub Models (free).
Uses GPT-4o or Llama 3.1 to create perfect training examples for Pullbot.
"""

import os, sys, json, time, requests, random, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

GITHUB_MODELS_URL = "https://models.inference.ai.azure.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

def generate_qa(topic, model="gpt-4o"):
    """Use GitHub Models to generate Q&A pairs about a topic"""
    if not GITHUB_TOKEN:
        return []
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Generate 3 question-answer pairs about "{topic}".
Format each pair exactly as:
Q: [question]
A: [answer]

Make answers concise (1-2 sentences). Use simple, clear language.
Keep questions varied: one definition, one explanation, one example."""

    try:
        r = requests.post(
            f"{GITHUB_MODELS_URL}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            },
            timeout=30
        )
        
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            
            # Parse Q&A pairs from the response
            qa_pairs = []
            current_q = None
            
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('Q:') or line.startswith('Question:'):
                    if current_q:
                        qa_pairs.append(current_q)
                    current_q = line
                elif line.startswith('A:') or line.startswith('Answer:'):
                    if current_q:
                        qa_pairs.append(current_q)
                    qa_pairs.append(line)
                    current_q = None
            
            if current_q:
                qa_pairs.append(current_q)
            
            return qa_pairs
        else:
            print(f"   API status: {r.status_code}")
    except Exception as e:
        print(f"   Error: {e}")
    
    return []

def generate_continuation(topic, model="gpt-4o"):
    """Generate continuation examples"""
    if not GITHUB_TOKEN:
        return []
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Create 2 sentence completion examples about "{topic}".
Format each as:
Complete: [first half of sentence]... → [full sentence]

Example:
Complete: Machine learning is a field of... → Machine learning is a field of artificial intelligence that enables computers to learn from data."""

    try:
        r = requests.post(
            f"{GITHUB_MODELS_URL}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 200
            },
            timeout=30
        )
        
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            lines = [l.strip() for l in text.split('\n') if l.strip().startswith('Complete:')]
            return lines
    except:
        pass
    
    return []

def generate_from_wordbank(model="gpt-4o", max_topics=20):
    """Generate Q&A and continuations for words in the wordbank"""
    print("=" * 50)
    print(f"🤖 SYNTHETIC Q&A GENERATOR (via GitHub Models)")
    print(f"   Model: {model}")
    print("=" * 50)
    
    if not GITHUB_TOKEN:
        print("❌ No GITHUB_TOKEN set. Skipping synthetic generation.")
        return []
    
    wordbank_path = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
    if not os.path.exists(wordbank_path):
        print("❌ No wordbank found")
        return []
    
    with open(wordbank_path) as f:
        bank = json.load(f)
    
    # Get defined words
    defined = [
        word for word, info in bank['words'].items()
        if isinstance(info, dict) and info.get('has_definition')
    ]
    
    # Shuffle and pick random topics
    random.shuffle(defined)
    topics = defined[:max_topics]
    
    all_qa = []
    qa_count = 0
    
    for i, word in enumerate(topics):
        print(f"   {i+1}/{len(topics)}: {word}")
        
        # Generate Q&A
        qa = generate_qa(word, model)
        if qa:
            all_qa.extend(qa)
            qa_count += len(qa)
        
        # Generate continuations
        cont = generate_continuation(word, model)
        if cont:
            all_qa.extend(cont)
        
        # Progress
        if (i + 1) % 5 == 0:
            print(f"      Progress: {qa_count} lines generated")
        
        time.sleep(0.5)  # Rate limit
    
    # Save to corpus
    if all_qa:
        corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        
        text = '\n'.join(all_qa)
        with open(corpus_path, 'a') as f:
            f.write('\n\n---\n\n' + text)
        
        print(f"\n✅ Added {len(all_qa)} synthetic training lines to corpus")
        
        # Show samples
        print("\n--- Sample outputs ---")
        for line in all_qa[:6]:
            print(f"   {line[:100]}")
    
    return all_qa

def generate_from_topics(topics, model="gpt-4o"):
    """Generate Q&A from a list of topics"""
    print("=" * 50)
    print(f"🤖 SYNTHETIC Q&A FROM TOPICS")
    print(f"   Topics: {len(topics)} | Model: {model}")
    print("=" * 50)
    
    if not GITHUB_TOKEN:
        print("❌ No GITHUB_TOKEN set")
        return []
    
    all_qa = []
    
    for i, topic in enumerate(topics):
        print(f"   {i+1}/{len(topics)}: {topic}")
        qa = generate_qa(topic, model)
        if qa:
            all_qa.extend(qa)
        time.sleep(0.5)
    
    if all_qa:
        corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        
        text = '\n'.join(all_qa)
        with open(corpus_path, 'a') as f:
            f.write('\n\n---\n\n' + text)
        
        print(f"\n✅ Added {len(all_qa)} lines to corpus")
    
    return all_qa

if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
    
    # Try wordbank first, then fall back to hardcoded topics
    result = generate_from_wordbank(model=model)
    
    if not result:
        # Fallback topics if wordbank is empty
        fallback_topics = [
            "machine learning", "artificial intelligence", "python programming",
            "climate change", "solar system", "human brain",
            "world war 2", "renaissance art", "quantum physics",
            "natural selection", "internet technology", "ancient egypt"
        ]
        print("\n⚠️ No wordbank results, using fallback topics...")
        generate_from_topics(fallback_topics, model=model)
