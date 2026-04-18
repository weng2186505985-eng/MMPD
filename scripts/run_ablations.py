import os
import sys
import json
import subprocess
import shutil

def run_experiment(name, config_updates):
    print(f"\n{'='*80}")
    print(f"Running Experiment: {name}")
    print(f"{'='*80}")
    
    # Load base config
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    # Update config
    for k, v in config_updates.items():
        config[k] = v
        
    # Save temp config
    with open('config_temp.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    # Run training (train.py must be modified to accept config_temp.json if we want, 
    # but the easiest way is to overwrite config.json and restore it later)
    shutil.copy('config.json', 'config_backup.json')
    shutil.copy('config_temp.json', 'config.json')
    
    try:
        # 1. Train
        print(f"--- Training {name} ---")
        subprocess.run([sys.executable, 'scripts/train.py'], check=True)
        
        # 2. Inference
        print(f"--- Evaluating {name} ---")
        subprocess.run([sys.executable, 'scripts/inference.py'], check=True)
        
        # 3. Save results
        os.makedirs('ablation_results', exist_ok=True)
        if os.path.exists('results/summary.txt'):
            shutil.copy('results/summary.txt', f'ablation_results/summary_{name}.txt')
        if os.path.exists('results/train_history.json'):
            shutil.copy('results/train_history.json', f'ablation_results/train_history_{name}.json')
            
    except subprocess.CalledProcessError as e:
        print(f"Experiment {name} failed: {e}")
    finally:
        # Restore config
        shutil.copy('config_backup.json', 'config.json')
        os.remove('config_backup.json')
        os.remove('config_temp.json')

if __name__ == "__main__":
    experiments = [
        {
            "name": "Full_Model",
            "updates": {
                "ablation_no_tc": False,
                "ablation_no_uw": False,
                "ablation_no_transformer": False
            }
        },
        {
            "name": "w_o_TC_Embedding",
            "updates": {
                "ablation_no_tc": True,
                "ablation_no_uw": False,
                "ablation_no_transformer": False
            }
        },
        {
            "name": "w_o_Uncertainty_Weighting",
            "updates": {
                "ablation_no_tc": False,
                "ablation_no_uw": True,
                "ablation_no_transformer": False
            }
        },
        {
            "name": "w_o_Transformer_Denoiser",
            "updates": {
                "ablation_no_tc": False,
                "ablation_no_uw": False,
                "ablation_no_transformer": True
            }
        }
    ]
    
    for exp in experiments:
        run_experiment(exp['name'], exp['updates'])
    
    print("\nAll experiments finished. Results saved in 'ablation_results/' directory.")
