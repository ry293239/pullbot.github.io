"""Watchdog - checks and revives dead workflows"""
import sys, json, os, requests

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO = "pullbot-ai/pullbot-ai.github.io"

def check_workflow(wf_file, wf_name, ping_file):
    url = f"https://api.github.com/repos/{REPO}/actions/workflows/{wf_file}/runs?per_page=3"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    try:
        r = requests.get(url, headers=headers)
        data = r.json()
        runs = data.get('workflow_runs', [])
        
        statuses = [run.get('status') for run in runs]
        
        if 'in_progress' in statuses or 'queued' in statuses:
            status = 'active'
        elif runs and runs[0].get('conclusion') == 'success':
            status = 'healthy'
        elif runs and runs[0].get('conclusion') == 'failure':
            status = 'dead'
        else:
            status = 'unknown'
        
        print(f"   {wf_name}: {status}")
        
        if status == 'dead':
            print(f"   ⚡ Reviving {wf_name}...")
            dispatch_url = f"https://api.github.com/repos/{REPO}/actions/workflows/{ping_file}/dispatches"
            requests.post(dispatch_url, headers=headers, json={"ref": "main"})
            print(f"   ✅ Ping sent to {ping_file}")
        
        return status
    except Exception as e:
        print(f"   {wf_name}: error ({e})")
        return 'error'

if __name__ == '__main__':
    print(f"🔍 WATCHDOG BETA")
    print("=" * 50)
    
    check_workflow("train.yml", "Train", "ping.yml")
    check_workflow("mass-scrape.yml", "Mass Scrape", "mass-scrape-ping.yml")
    check_workflow("quick-train.yml", "Quick Train", "quick-train-ping.yml")
    check_workflow("optimise.yml", "Optimize", "ping.yml")
    
    print("=" * 50)
    print("✅ Watchdog complete")
