"""
Pullbot Benchmark
Tests the model against fixed prompts after every training run.
Tracks scores over time to measure improvement.
"""

import os, sys, json, time, requests

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
    with open(SCORES_PATH, 'w') as f:
        json.dump(scores, f, indent=2)

def test_prompt(prompt):
    """Get model response for a prompt"""
    try:
        r = requests.get(f"{API_URL}/ask?q={prompt}", timeout=60)
        if r.status_code == 200:
            return r.json().get('response', '')
    except:
        pass
    return ""

def score_response(response):
    """Basic auto-scoring"""
    score = 0
    if not response:
        return 0
    
    # Coherent: has actual words
    words = response.split()
    if len(words) > 3:
        score += 1
    
    # Not repetitive
    lines = response.split('\n')
    unique_lines = len(set(lines))
    if unique_lines > len(lines) * 0.5:
        score += 1
    
    # Not too short, not too long
    if 20 < len(response) < 1000:
        score += 1
    
    # Contains at least one period (proper sentence)
    if '.' in response:
        score += 1
    
    # No obvious junk patterns
    junk = ['badge', 'shield', 'svg', 'catch2', 'response = bias']
    if not any(j in response.lower() for j in junk):
        score += 1
    
    return score

def run_benchmark():
    print("=" * 50)
    print("🧪 PULLBOT BENCHMARK")
    print("=" * 50)
    
    benchmark = load_benchmark()
    scores = load_scores()
    
    prompts = benchmark['prompts']
    results = []
    total_score = 0
    max_score = len(prompts) * 5
    
    for i, prompt in enumerate(prompts):
        print(f"   {i+1}/{len(prompts)}: {prompt[:50]}...")
        response = test_prompt(prompt)
        score = score_response(response)
        total_score += score
        
        results.append({
            'prompt': prompt,
            'response': response[:200],
            'score': score
        })
        
        time.sleep(2)  # Rate limit
    
    percentage = (total_score / max_score) * 100
    
    scores['runs'].append({
        'timestamp': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
        'score': total_score,
        'max': max_score,
        'percentage': round(percentage, 1),
        'results': results
    })
    
    # Keep only last 20 runs
    scores['runs'] = scores['runs'][-20:]
    
    save_scores(scores)
    
    print(f"\n✅ Score: {total_score}/{max_score} ({percentage:.1f}%)")
    
    # Show improvement
    if len(scores['runs']) >= 2:
        prev = scores['runs'][-2]['percentage']
        change = percentage - prev
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"   Previous: {prev:.1f}% | Change: {direction} {change:+.1f}%")
    
    return results

if __name__ == '__main__':
    run_benchmark()
