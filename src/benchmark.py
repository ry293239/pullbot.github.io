"""
Pullbot Benchmark
Tests model against fixed prompts. Better scoring rewards actual answers.
"""

import os, sys, json, time, requests, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

API_URL = "https://pullbot-api.onrender.com"
BENCHMARK_PATH = os.path.join(REPO_ROOT, "data", "benchmark.json")
SCORES_PATH = os.path.join(REPO_ROOT, "data", "benchmark_scores.json")

def load_benchmark():
    with open(BENCHMARK_PATH) as f:
        return json.load(f)

def load_scores():
    if os.path.exists(SCORES_PATH):
        with open(SCORES_PATH) as f:
            return json.load(f)
    return {"runs": []}

def save_scores(scores):
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    with open(SCORES_PATH, 'w') as f:
        json.dump(scores, f, indent=2)

def test_prompt(prompt):
    try:
        r = requests.get(f"{API_URL}/ask?q={prompt}", timeout=90)
        if r.status_code == 200:
            return r.json().get('response', '')
    except:
        pass
    return ""

def score_response(response, prompt):
    """Better scoring: rewards actual answers, penalizes fragments"""
    score = 0
    
    if not response or len(response) < 5:
        return 0
    
    # 1. Complete sentence (ends with punctuation)
    if response.rstrip()[-1] in '.!?':
        score += 1
    
    # 2. Has substance (more than a few words)
    words = response.split()
    if len(words) > 5:
        score += 1
    
    # 3. No fragment patterns like "of the... of it..."
    if not re.search(r'\bof \w+\.\s*\bof \w+', response):
        score += 1
    
    # 4. Contains content words (not just filler)
    filler = {'the', 'of', 'it', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'is', 'was', 'are', 'this', 'that', 'with', 'from', 'they', 'have', 'been'}
    content_words = [w for w in words if w.lower() not in filler and len(w) > 2]
    if len(content_words) >= 2:
        score += 1
    
    # 5. Not just repeating itself
    unique_words = len(set(w.lower() for w in words))
    if unique_words > len(words) * 0.4:
        score += 1
    
    return score

def run_benchmark():
    print("=" * 50)
    print("🧪 PULLBOT BENCHMARK")
    print("=" * 50)
    
    print("Waking up Render...")
    try:
        requests.get(f"{API_URL}/health", timeout=30)
    except:
        pass
    time.sleep(5)
    
    benchmark = load_benchmark()
    scores = load_scores()
    
    prompts = benchmark['prompts'][:10]
    results = []
    total_score = 0
    max_score = len(prompts) * 5
    
    for i, prompt in enumerate(prompts):
        print(f"   {i+1}/{len(prompts)}: {prompt[:60]}...")
        response = test_prompt(prompt)
        score = score_response(response, prompt)
        total_score += score
        
        results.append({
            'prompt': prompt,
            'response': response[:200] if response else '(no response)',
            'score': score
        })
        
        print(f"      Score: {score}/5")
        if response:
            print(f"      Response: {response[:100]}...")
        
        if i < len(prompts) - 1:
            time.sleep(5)
    
    percentage = (total_score / max_score) * 100 if max_score > 0 else 0
    
    scores['runs'].append({
        'timestamp': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
        'score': total_score,
        'max': max_score,
        'percentage': round(percentage, 1),
        'results': results
    })
    
    scores['runs'] = scores['runs'][-30:]
    save_scores(scores)
    
    print(f"\n✅ Score: {total_score}/{max_score} ({percentage:.1f}%)")
    
    if len(scores['runs']) >= 2:
        prev = scores['runs'][-2]['percentage']
        change = percentage - prev
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"   Previous: {prev:.1f}% | Change: {direction} {change:+.1f}%")
    
    return results

if __name__ == '__main__':
    run_benchmark()
