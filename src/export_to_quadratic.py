"""
Export pruned + quantized model weights to CSV files for Quadratic.
Only exports NON-ZERO weights (skips pruned ones).
"""

import os, sys, json, csv, time, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

from model import PullbotModel

def export_weights():
    print("=" * 50)
    print("📤 EXPORTING WEIGHTS FOR QUADRATIC")
    print("=" * 50)
    
    bot = PullbotModel()
    
    export_dir = os.path.join(REPO_ROOT, "data", "quadratic")
    os.makedirs(export_dir, exist_ok=True)
    
    # Define which layers go to which sheets
    layer_groups = {
        'embedding': ['transformer.wte', 'transformer.wpe'],
        'output': ['lm_head'],
    }
    
    # Add transformer layers
    for i in range(6):
        layer_groups[f'layer_{i}_attn'] = [f'transformer.h.{i}.attn']
        layer_groups[f'layer_{i}_ffn'] = [f'transformer.h.{i}.mlp']
    
    total_nonzero = 0
    total_params = 0
    sheets_created = 0
    
    for sheet_name, layer_patterns in layer_groups.items():
        rows = []
        
        for name, param in bot.model.named_parameters():
            if any(p in name for p in layer_patterns) and param.dim() >= 2:
                weights = param.data.cpu().numpy()
                total_params += weights.size
                
                # Find non-zero positions
                nonzero_mask = np.abs(weights) > 0.0001
                nonzero_indices = np.where(nonzero_mask)
                nonzero_count = len(nonzero_indices[0])
                total_nonzero += nonzero_count
                
                # Store as (row, col, value)
                for r, c in zip(nonzero_indices[0], nonzero_indices[1]):
                    rows.append([int(r), int(c), float(weights[r, c])])
        
        if rows:
            csv_path = os.path.join(export_dir, f'{sheet_name}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['row', 'col', 'value'])
                writer.writerows(rows)
            
            size_kb = os.path.getsize(csv_path) / 1024
            print(f"   ✅ {sheet_name}.csv: {len(rows):,} weights ({size_kb:.0f}KB)")
            sheets_created += 1
    
    # Manifest
    sparsity = (1 - total_nonzero/total_params) * 100 if total_params > 0 else 0
    manifest = {
        'model': 'distilgpt2',
        'exported_at': time.time(),
        'num_sheets': sheets_created,
        'total_params': total_params,
        'non_zero_params': total_nonzero,
        'sparsity_pct': round(sparsity, 1),
        'sheets': list(layer_groups.keys()),
        'format': 'row,col,value (sparse)'
    }
    
    with open(os.path.join(export_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Exported {sheets_created} sheets to data/quadratic/")
    print(f"   Non-zero: {total_nonzero:,} / {total_params:,} ({sparsity:.1f}% sparse)")
    print(f"   Quadratic will only load the non-zero weights!")

if __name__ == '__main__':
    export_weights()
