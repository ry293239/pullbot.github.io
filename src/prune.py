"""Prune an existing model to reduce size for free hosting"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

from model import PullbotModel

target_mb = 500  # Target RAM for Render

if len(sys.argv) > 1:
    sparsity = float(sys.argv[1])
else:
    sparsity = 0.5

print("=" * 50)
print("✂️ PULLBOT PRUNING TOOL")
print(f"   Target: <{target_mb}MB RAM for free hosting")
print("=" * 50)

bot = PullbotModel()

# Initial size
before = bot.get_model_size_estimate()
print(f"\n📊 Current size: {before['estimated_ram_mb']:.0f}MB RAM")
print(f"   Sparsity: {before['sparsity_pct']:.1f}%")

# Try increasing sparsity until under target
current_sparsity = sparsity
while True:
    print(f"\n🔧 Trying {current_sparsity*100:.0f}% pruning...")
    bot.prune_model(target_sparsity=current_sparsity)
    
    after = bot.get_model_size_estimate()
    print(f"   Result: {after['estimated_ram_mb']:.0f}MB RAM, {after['sparsity_pct']:.1f}% sparse")
    
    if after['estimated_ram_mb'] <= target_mb:
        print(f"\n✅ {after['estimated_ram_mb']:.0f}MB <= {target_mb}MB target!")
        break
    
    if current_sparsity >= 0.95:
        print(f"\n⚠️ Even 95% pruning won't fit. Model may be too degraded.")
        break
    
    current_sparsity += 0.1

# Save
bot.save_and_chunk()
print("\n💾 Pruned model saved to models/chunks/")
print("   Push to GitHub and it's ready for free hosting!")
