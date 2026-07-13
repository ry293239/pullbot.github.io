"""
Export pruned + quantized model weights to CSV files for Quadratic.
Splits large layers into multiple files to stay under GitHub's 100MB limit.
"""

import os, sys, json, csv, time, numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

from model import PullbotModel

MAX_FILE_SIZE_MB = 45  # Keep well under 100MB limit
MAX_ROWS_PER_FILE = 100000  # Roughly 45MB in CSV

def export_weights():
    print("=" * 50)
    print("📤 EXPORTING WEIGHTS FOR QUADRATIC")
    print("=" * 50)
    
    bot = PullbotModel()
    
    export_dir = os.path.join(REPO_ROOT, "data", "quadratic")
    os.makedirs(export_dir, exist_ok=True)
    
    # Clear old exports
    for old_file in os.listdir(export_dir):
        if old_file.endswith('.csv'):
            os.remove(os.path.join(export_dir, old_file))
    
    # Define layers
    layer_map = {
        'embedding': ['transformer.wte', 'transformer.wpe'],
        'output': ['lm_head'],
    }
    
    for i in range(6):
        layer_map[f'layer_{i}_attn'] = [f'transformer.h.{i}.attn']
        layer_map[f'layer_{i}_ffn'] = [f'transformer.h.{i}.mlp']
    
    all_sheets = []
    total_nonzero = 0
    total_params = 0
    
    for sheet_base_name, layer_patterns in layer_map.items():
        # Collect all weights for this layer group
        all_rows = []
        
        for name, param in bot.model.named_parameters():
            if any(p in name for p in layer_patterns) and param.dim() >= 2:
                weights = param.data.cpu().numpy()
                total_params += weights.size
                
                nonzero_mask = np.abs(weights) > 0.0001
                nonzero_indices = np.where(nonzero_mask)
                nonzero_count = len(nonzero_indices[0])
                total_nonzero += nonzero_count
                
                for r, c in zip(nonzero_indices[0], nonzero_indices[1]):
                    all_rows.append([int(r), int(c), float(weights[r, c])])
        
        if not all_rows:
            continue
        
        # Split into chunks if too big
        num_chunks = max(1, len(all_rows) // MAX_ROWS_PER_FILE + 1)
        rows_per_chunk = len(all_rows) // num_chunks + 1
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * rows_per_chunk
            end = min(start + rows_per_chunk, len(all_rows))
            chunk_rows = all_rows[start:end]
            
            if num_chunks == 1:
                sheet_name = sheet_base_name
            else:
                sheet_name = f"{sheet_base_name}_{chunk_idx}"
            
            csv_path = os.path.join(export_dir, f'{sheet_name}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['row', 'col', 'value'])
                writer.writerows(chunk_rows)
            
            size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            print(f"   ✅ {sheet_name}.csv: {len(chunk_rows):,} weights ({size_mb:.1f}MB)")
            all_sheets.append(sheet_name)
    
    # Manifest
    sparsity = (1 - total_nonzero/total_params) * 100 if total_params > 0 else 0
    manifest = {
        'model': 'distilgpt2',
        'exported_at': time.time(),
        'num_sheets': len(all_sheets),
        'total_params': total_params,
        'non_zero_params': total_nonzero,
        'sparsity_pct': round(sparsity, 1),
        'sheets': all_sheets,
        'format': 'row,col,value (sparse)',
        'max_file_size_mb': MAX_FILE_SIZE_MB
    }
    
    with open(os.path.join(export_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Exported {len(all_sheets)} sheets to data/quadratic/")
    print(f"   Non-zero: {total_nonzero:,} / {total_params:,} ({sparsity:.1f}% sparse)")
    print(f"   Each file < {MAX_FILE_SIZE_MB}MB")

if __name__ == '__main__':
    export_weights()
