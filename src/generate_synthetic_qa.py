"""
Generate high-quality Q&A training data using GitHub Models (free).
Uses GPT-4o or Llama 3.1 to create training examples AND grade Pullbot's answers.
"""

import os, sys, json, time, requests, random, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

GITHUB_MODELS_URL = "https://models.inference.ai.azure.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
PULLBOT_API = "https://pullbot-api.onrender.com"

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
Complete: [first half of sentence]... → [full sentence]"""

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
            return [l.strip() for l in text.split('\n') if l.strip().startswith('Complete:')]
    except:
        pass
    return []

def grade_pullbot_answer(question, pullbot_answer, model="gpt-4o"):
    """Use GitHub Models to grade Pullbot's response and suggest improvements"""
    if not GITHUB_TOKEN or not pullbot_answer:
        return None
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Grade this AI response on a scale of 1-5.

Question: {question}
AI Response: {pullbot_answer}

Return ONLY a JSON object (no other text):
{{"score": <1-5>, "feedback": "<one sentence>", "improved": "<better version>"}}"""

    try:
        r = requests.post(
            f"{GITHUB_MODELS_URL}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 250
            },
            timeout=30
        )
        
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            # Try to parse JSON
            try:
                # Find JSON in response
                json_match = re.search(r'\{[^}]+\}', text)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
    except Exception as e:
        print(f"   Grade error: {e}")
    return None

def ask_pullbot(question):
    """Get Pullbot's current answer for a question"""
    try:
        r = requests.get(f"{PULLBOT_API}/ask?q={question}", timeout=60)
        if r.status_code == 200:
            return r.json().get('response', '')
    except:
        pass
    return ""

def generate_and_grade(model="gpt-4o", max_topics=10):
    """Generate Q&A, test Pullbot, grade responses, save improvements"""
    print("=" * 50)
    print(f"🤖 SYNTHETIC Q&A + GRADING (via GitHub Models)")
    print(f"   Model: {model}")
    print("=" * 50)
    
    if not GITHUB_TOKEN:
        print("❌ No GITHUB_TOKEN set")
        return
    
    wordbank_path = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
    topics = []
    
    if os.path.exists(wordbank_path):
        with open(wordbank_path) as f:
            bank = json.load(f)
        defined = [
            word for word, info in bank['words'].items()
            if isinstance(info, dict) and info.get('has_definition')
        ]
        random.shuffle(defined)
        topics = defined[:max_topics]
    
    if not topics:
        topics = ["machine learning", "artificial intelligence", "python programming",
                  "climate change", "solar system", "quantum physics", "natural selection"]
    
    all_qa = []
    grades = []
    
    for i, word in enumerate(topics):
        print(f"\n   {i+1}/{len(topics)}: {word}")
        
        # Generate perfect Q&A
        qa = generate_qa(word, model)
        if qa:
            # Extract first question
            for line in qa:
                if line.startswith('Q:'):
                    question = line.replace('Q:', '').strip()
                    
                    # Ask Pullbot the same question
                    print(f"      Asking Pullbot: {question[:60]}...")
                    pullbot_answer = ask_pullbot(question)
                    
                    if pullbot_answer:
                        # Grade Pullbot's answer
                        print(f"      Grading response...")
                        grade = grade_pullbot_answer(question, pullbot_answer, model)
                        
                        if grade:
                            score = grade.get('score', 0)
                            improved = grade.get('improved', '')
                            feedback = grade.get('feedback', '')
                            
                            print(f"      Score: {score}/5 | {feedback}")
                            grades.append({
                                'question': question,
                                'pullbot_answer': pullbot_answer[:200],
                                'score': score,
                                'feedback': feedback,
                                'improved': improved
                            })
                            
                            # Add the improved version to training data
                            if improved and score < 4:
                                all_qa.append(f"Q: {question}")
                                all_qa.append(f"A: {improved}")
                    
                    break  # Only test first question per topic
            
            # Add all generated Q&A to training data
            all_qa.extend(qa)
        
        time.sleep(1)
    
    # Save grades
    if grades:
        grades_path = os.path.join(REPO_ROOT, 'data', 'pullbot_grades.json')
        existing = []
        if os.path.exists(grades_path):
            with open(grades_path) as f:
                existing = json.load(f)
        existing.extend(grades)
        with open(grades_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        avg = sum(g['score'] for g in grades) / len(grades)
        print(f"\n📊 Average score: {avg:.1f}/5 across {len(grades)} tests")
    
    # Save training data
    if all_qa:
        corpus_path = os.path.join(REPO_ROOT, 'data', 'processed', 'corpus.txt')
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        text = '\n'.join(all_qa)
        with open(corpus_path, 'a') as f:
            f.write('\n\n---\n\n' + text)
        print(f"✅ Added {len(all_qa)} lines to corpus")

if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
    generate_and_grade(model=model)
