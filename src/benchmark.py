"""
Pullbot Benchmark
Tests the model against fixed prompts after every training run.
Tracks scores over time to measure improvement.
Lightweight: 10 prompts, 5s delay, 90s timeout.
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
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)
    with open(SCORES_PATH, 'w') as f:
        json.dump(scores, f, indent=2)

def test_prompt(prompt):
    """Get model response for a prompt"""
    try:
        r = requests.get(f"{API_URL}/ask?q={prompt}", timeout=90)
        if r.status_code == 200:
            return r.json().get('response', '')
    except Exception as e:
        print(f"      Error: {e}")
    return ""

def score_response(response):
    """Basic auto-scoring (0-5 points)"""
    if not response:
        return 0
    
    score = 0
    words = response.split()
    
    # 1. Coherent: has actual words
    if len(words) > 3:
        score += 1
    
    # 2. Not repetitive (check for repeated lines)
    lines = response.split('\n')
    unique_lines = len(set(lines))
    if unique_lines > len(lines) * 0.5:
        score += 1
    
    # 3. Good length (20-1000 chars)
    if 20 < len(response) < 1000:
        score += 1
    
    # 4. Contains proper punctuation (at least one period)
    if '.' in response:
        score += 1
    
    # 5. No obvious junk patterns
    junk = ['badge', 'shield', 'svg', 'catch2', 'response = bias', '```']
    if not any(j in response.lower() for j in junk):
        score += 1
    
    return score

def run_benchmark():
    print("=" * 50)
    print("🧪 PULLBOT BENCHMARK")
    print("=" * 50)
    
    # Wake up Render
    print("Waking up Render...")
    try:
        r = requests.get(f"{API_URL}/health", timeout=30)
        if r.status_code == 200:
            print("   Render is awake!")
        else:
            print(f"   Render status: {r.status_code}")
    except Exception as e:
        print(f"   Render unreachable: {e}")
        # Save empty run and exit
        scores = load_scores()
        scores['runs'].append({
            'timestamp': time.time(),
            'date': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
            'score': 0,
            'max': 50,
            'percentage': 0,
            'error': str(e),
            'results': []
        })
        save_scores(scores)
        return []
    
    time.sleep(5)
    
    benchmark = load_benchmark()
    scores = load_scores()
    
    prompts = benchmark['prompts'][:10]  # Only 10 prompts
    results = []
    total_score = 0
    max_score = len(prompts) * 5
    
    for i, prompt in enumerate(prompts):
        print(f"   {i+1}/{len(prompts)}: {prompt[:60]}...")
        response = test_prompt(prompt)
        score = score_response(response)
        total_score += score
        
        results.append({
            'prompt': prompt,
            'response': response[:200] if response else '(no response)',
            'score': score
        })
        
        print(f"      Score: {score}/5")
        
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
    
    # Keep only last 30 runs
    scores['runs'] = scores['runs'][-30:]
    
    save_scores(scores)
    
    print(f"\n✅ Score: {total_score}/{max_score} ({percentage:.1f}%)")
    
    # Show improvement
    if len(scores['runs']) >= 2:
        prev = scores['runs'][-2]['percentage']
        change = percentage - prev
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"   Previous: {prev:.1f}% | Change: {direction} {change:+.1f}%")
    
    # Show sample responses
    print("\n--- Sample Responses ---")
    for r in results[:3]:
        print(f"\n   Q: {r['prompt']}")
        print(f"   A: {r['response'][:150]}...")
        print(f"   Score: {r['score']}/5")
    
    return results

if __name__ == '__main__':
    run_benchmark()
