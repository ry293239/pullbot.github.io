"""
Pullbot Benchmark - AI Graded
Triggers GitHub Actions responder, polls for response, grades with GPT-4o.
"""

import os, sys, json, time, requests, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

GH_REPO = "pullbot-ai/pullbot-ai.github.io"
BENCHMARK_PATH = os.path.join(REPO_ROOT, "data", "benchmark.json")
SCORES_PATH = os.path.join(REPO_ROOT, "data", "benchmark_scores.json")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

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

def ask_github(question):
    """Trigger GitHub responder and poll for response"""
    # Trigger the workflow
    try:
        requests.post(
            f"https://api.github.com/repos/{GH_REPO}/actions/workflows/respond.yml/dispatches",
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"
            },
            json={"ref": "main", "inputs": {"question": question}},
            timeout=10
        )
    except:
        return ""
    
    # Poll for response
    for i in range(30):
        time.sleep(10)
        try:
            r = requests.get(
                f"https://raw.githubusercontent.com/{GH_REPO}/main/response.json?t={time.time()}",
                headers={"Cache-Control": "no-cache"}
            )
            if r.status_code == 200:
                data = r.json()
                if data.get('question') == question and data.get('response'):
                    return data['response']
        except:
            pass
    
    return ""

def ai_grade(question, response):
    """Use GitHub Models to grade the response"""
    if not GITHUB_TOKEN or not response:
        return 0
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""Grade this AI response from 1-5. Only return the number.

Question: {question}
AI Response: {response}

Score (1=terrible, 3=ok, 5=excellent):"""

    try:
        r = requests.post(
            "https://models.inference.ai.azure.com/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 5
            },
            timeout=30
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            nums = re.findall(r'\d+', text)
            if nums:
                score = int(nums[0])
                return max(1, min(5, score))
    except:
        pass
    return 0

def run_benchmark():
    print("=" * 50)
    print("🧪 PULLBOT BENCHMARK (GitHub Actions + AI Graded)")
    print("=" * 50)
    
    if not GITHUB_TOKEN:
        print("❌ No GITHUB_TOKEN set")
        return
    
    benchmark = load_benchmark()
    scores = load_scores()
    
    prompts = benchmark['prompts'][:10]
    results = []
    total_score = 0
    max_score = len(prompts) * 5
    
    for i, prompt in enumerate(prompts):
        print(f"   {i+1}/{len(prompts)}: {prompt[:60]}...")
        print(f"      Triggering responder...")
        response = ask_github(prompt)
        
        if response:
            print(f"      Response: {response[:100]}...")
            score = ai_grade(prompt, response)
            print(f"      AI Score: {score}/5")
        else:
            print(f"      No response")
            score = 0
        
        total_score += score
        results.append({
            'prompt': prompt,
            'response': response[:200] if response else '(no response)',
            'score': score
        })
    
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
    
    print(f"\n✅ AI Score: {total_score}/{max_score} ({percentage:.1f}%)")
    
    if len(scores['runs']) >= 2:
        prev = scores['runs'][-2]['percentage']
        change = percentage - prev
        direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"   Previous: {prev:.1f}% | Change: {direction} {change:+.1f}%")
    
    return results

if __name__ == '__main__':
    run_benchmark()
