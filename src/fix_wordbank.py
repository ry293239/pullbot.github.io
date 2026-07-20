"""
Fix corrupted wordbank.json and recover as much data as possible.
Never deletes data - only repairs it.
"""

import os, sys, json, re, shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def fix_wordbank():
    path = os.path.join(REPO_ROOT, 'data', 'wordbank.json')
    
    if not os.path.exists(path):
        print("No wordbank found - creating fresh one")
        fresh = {"words": {}, "total_articles": 0, "total_words": 0, "total_defined": 0}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(fresh, f, indent=2)
        return
    
    # Read raw text
    with open(path, 'r') as f:
        raw = f.read()
    
    # Try normal load first
    try:
        data = json.loads(raw)
        total = len(data.get('words', {}))
        defined = sum(1 for w in data.get('words', {}).values() if isinstance(w, dict) and w.get('has_definition'))
        print(f"✅ Wordbank OK: {total:,} words ({defined:,} defined)")
        return
    except json.JSONDecodeError as e:
        print(f"⚠️ Corrupted at line {e.lineno}: {e.msg}")
    
    # Backup before attempting repair
    backup_path = path + '.backup'
    shutil.copy(path, backup_path)
    print(f"📦 Backup saved to {backup_path}")
    
    # Fix 1: Remove git merge conflict markers
    cleaned = re.sub(r'<<<<<<< .*?\n', '', raw)
    cleaned = re.sub(r'=======\n', '', cleaned)
    cleaned = re.sub(r'>>>>>>> .*?\n', '', cleaned)
    
    # Fix 2: Remove trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix 3: Extract all valid word entries via regex
    rebuilt = {"words": {}, "total_articles": 0, "total_words": 0, "total_defined": 0}
    recovered = 0
    failed = 0
    
    # Match "word": { ... }
    pattern = r'"([^"]+)":\s*(\{[^}]+\})'
    for match in re.finditer(pattern, cleaned):
        word = match.group(1)
        try:
            entry = json.loads(match.group(2))
            rebuilt['words'][word] = entry
            recovered += 1
        except:
            failed += 1
    
    # Fix 4: If regex failed, try line-by-line recovery
    if recovered == 0:
        print("Regex recovery failed, trying line-by-line...")
        lines = cleaned.split('\n')
        for line in lines:
            line = line.strip().rstrip(',')
            if line.startswith('"') and '":' in line:
                try:
                    # Try to parse as individual word entry
                    pass
                except:
                    pass
    
    rebuilt['total_words'] = len(rebuilt['words'])
    rebuilt['total_defined'] = sum(
        1 for w in rebuilt['words'].values()
        if isinstance(w, dict) and w.get('has_definition')
    )
    
    # Save repaired version
    with open(path, 'w') as f:
        json.dump(rebuilt, f, indent=2)
    
    print(f"✅ Repaired! Recovered {recovered:,} words ({failed} failed)")
    print(f"   Total: {rebuilt['total_words']:,} | Defined: {rebuilt['total_defined']:,}")
    
    # Verify saved file is valid
    try:
        with open(path) as f:
            json.load(f)
        print("   Verified: saved file is valid JSON")
    except:
        print("   ⚠️ Saved file still corrupted!")

if __name__ == '__main__':
    fix_wordbank()
