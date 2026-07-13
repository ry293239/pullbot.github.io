"""Full optimization: prune + quantize an existing model"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

from model import PullbotModel

target_sparsity = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7

print("=" * 50)
print("🔧 PULLBOT OPTIMIZER")
print(f"   Target sparsity: {target_sparsity*100:.0f}%")
print("=" * 50)

bot = PullbotModel()

# Before
before = bot.get_model_size_estimate()
print(f"\n📊 Before: {before['estimated_ram_mb']:.0f}MB RAM")

# Prune
bot.prune_model(target_sparsity=target_sparsity)
after_prune = bot.get_model_size_estimate()
print(f"📊 After prune: {after_prune['estimated_ram_mb']:.0f}MB RAM")

# Quantize
bot.quantize_model()
final = bot.get_model_size_estimate()
print(f"📊 After quantize: {final['estimated_ram_mb']:.0f}MB RAM")

# Save
bot.save_and_chunk()

print("\n" + "=" * 50)
print(f"✅ DONE! {before['estimated_ram_mb']:.0f}MB → {final['estimated_ram_mb']:.0f}MB")
print(f"   Savings: {before['estimated_ram_mb'] - final['estimated_ram_mb']:.0f}MB")
print("=" * 50)
