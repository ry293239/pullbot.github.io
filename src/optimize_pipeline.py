"""
Full Optimization Pipeline
Smart Prune → Precision Prune → Progressive Bits (recently changed weights) → Quantize
"""

import os, sys, torch, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from model import PullbotModel

print('='*50)
print('🔧 OPTIMIZATION PIPELINE')
print('='*50)

bot = PullbotModel()
before = bot.get_model_size_estimate()
print(f'Before: {before["total_params"]:,} params, {before["estimated_ram_mb"]:.0f}MB RAM')

# Stage 1: Smart Prune
print('\n--- Stage 1: Smart Prune ---')
bot.smart_prune(target_sparsity=0.5)
s1 = bot.get_model_size_estimate()
print(f'After: {s1["total_params"]:,} params, {s1["estimated_ram_mb"]:.0f}MB RAM')

# Stage 2: Precision Prune
print('\n--- Stage 2: Safe Precision Prune ---')
test_inputs = torch.randint(0, 50257, (10, 16))
bot.precision_prune_safe(significance=2, test_inputs=test_inputs)
s2 = bot.get_model_size_estimate()
print(f'After: {s2["total_params"]:,} params, {s2["estimated_ram_mb"]:.0f}MB RAM')

# Stage 3: Progressive Bit Reduction (only on weights changed by prune + extra margin)
print('\n--- Stage 3: Progressive Bits (changed weights only) ---')
bot.progressive_bit_reduce_changed(test_inputs=test_inputs, timeout_minutes=25)
s3 = bot.get_model_size_estimate()
print(f'After: {s3["total_params"]:,} params, {s3["estimated_ram_mb"]:.0f}MB RAM')

# Stage 4: 8-bit Quantize
print('\n--- Stage 4: 8-bit Quantize ---')
bot.quantize_model()
final = bot.get_model_size_estimate()
print(f'Final: {final["total_params"]:,} params, {final["estimated_ram_mb"]:.0f}MB RAM')

bot.save_and_chunk()

reduction = before['estimated_ram_mb'] - final['estimated_ram_mb']
print(f'\n✅ Optimization complete!')
print(f'   RAM: {before["estimated_ram_mb"]:.0f}MB → {final["estimated_ram_mb"]:.0f}MB (saved {reduction:.0f}MB)')
print(f'   Sparsity: {before["sparsity_pct"]:.1f}% → {final["sparsity_pct"]:.1f}%')
