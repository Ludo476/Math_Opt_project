import os
import json
import psutil
import pandas as pd
import time
import gc
import itertools
import traceback
import sys


# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.heuristic.ALNS_2 import ALNSSolver 

BASE_INSTANCES = os.path.join(PROJECT_ROOT, "instances_scalability")
BASE_RESULTS = os.path.join(PROJECT_ROOT, "results_scalability")
OUTPUT_FILE = os.path.join(BASE_RESULTS, "tuning_results_prova.csv")


TUNING_CLASSES = [0]      #I try on this class, but to be robust we should decide the best config for each class
TUNING_FAMILIES = [1] 

# parameters I think reasonable
GRID_COOLING_RATES = [0.75, 0.80, 0.90, 0.975]
GRID_TIME_LIMITS = [60, 300, 600] # 1min, 5min, 10min

# we can use more seeds to more deep results
SEEDS = [42, 123, 456, 101, 24, 999] 

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def safe_float(value):
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None

def save_single_row(row_dict, filepath):
    df = pd.DataFrame([row_dict])
    header = not os.path.exists(filepath)
    try:
        df.to_csv(filepath, mode='a', header=header, index=False)
    except PermissionError:
        time.sleep(1) # Riprova se il file è bloccato
        df.to_csv(filepath, mode='a', header=header, index=False)
    except Exception as e:
        print(f"error saving CSV: {e}")

# =============================================================================
# MAIN TUNING LOOP
# =============================================================================
def run_tuning():
    os.makedirs(BASE_RESULTS, exist_ok=True)
    
    processed_configs = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df_exist = pd.read_csv(OUTPUT_FILE)
            for _, r in df_exist.iterrows():
                key = f"{r['instance']}_{r['cooling_rate']}_{r['time_limit_config']}_{r['seed']}"
                processed_configs.add(key)
        except: pass

    #instances list
    instances_files = []
    for c_id in TUNING_CLASSES:
        class_path = os.path.join(BASE_INSTANCES, f"class_{c_id}")
        if not os.path.exists(class_path): continue
        for f_id in TUNING_FAMILIES:
        
            files = [
                os.path.join(class_path, f) 
                for f in os.listdir(class_path) 
                if f"fam_{f_id}_" in f and f.endswith(".json")
            ]
            instances_files.extend(files)

    #compute runs
    total_runs = len(instances_files) * len(GRID_COOLING_RATES) * len(GRID_TIME_LIMITS) * len(SEEDS)
    current_run = 0

    print(f"Start TUNING ({total_runs} configurations) -> {OUTPUT_FILE}")
    print(f"   Instances: {len(instances_files)}")
    print(f"   Cooling Rates: {GRID_COOLING_RATES}")
    print(f"   Time Limits: {GRID_TIME_LIMITS}")

    # Loop 
    for inst_file in sorted(instances_files):
        fname = os.path.basename(inst_file)
        
        try:
            with open(inst_file) as f:
                instance_data = json.load(f)
        except Exception as e:
            print(f"error loading {fname}: {e}")
            continue

        try:
            parts = fname.replace(".json", "").split('_')
            class_id = int(parts[1])
            family_id = int(parts[3])
        except:
            class_id, family_id = -1, -1

        # Iterations
        for cr, tl, seed in itertools.product(GRID_COOLING_RATES, GRID_TIME_LIMITS, SEEDS):
            current_run += 1
            
            # Skip if processed
            config_key = f"{fname}_{cr}_{tl}_{seed}"
            if config_key in processed_configs:
                # print(f" {config_key} skipped.")
                continue

            print(f"[{current_run}/{total_runs}] {fname} | CR={cr} TL={tl} | Seed {seed}...", end=" ", flush=True)
            
            mem_before = get_memory_usage()
            start_t = time.time()
            alns = None

            try:
                alns = ALNSSolver(
                    instance_data, 
                    time_limit=tl, 
                    seed=seed, 
                    cooling_rate=cr
                )
                best_sol, best_cost = alns.run(cooling_rate=cr)
                
                runtime = time.time() - start_t
                mem_after = get_memory_usage()

                #metrics
                init_cost = best_sol.get('initial_cost', getattr(alns, 'initial_cost', 0))
                imp_pct = ((init_cost - best_cost) / init_cost * 100) if init_cost > 0 else 0.0

                row = {
                    "instance": fname,
                    "class": class_id,
                    "family": family_id,
                    "solver": "ALNS",
                    "seed": seed,
                    
                    # Tuning parameters
                    "cooling_rate": cr,
                    "time_limit_config": tl,
                    
                    # Results
                    "objective": safe_float(best_cost),
                    "initial_cost": safe_float(init_cost),
                    "improvement_pct": safe_float(imp_pct),
                    "time_s": runtime,
                    "time_to_best_s": safe_float(best_sol.get('time_to_best_s')),
                    
                    # Diagnostics
                    "iterations": best_sol.get('total_iterations', 0),
                    "reheats": best_sol.get('reheats', 0),
                    "iter_to_best": best_sol.get('iter_to_best', 0),
                    "mem_gb": mem_after - mem_before
                }
                
                save_single_row(row, OUTPUT_FILE)
                print(f"Obj: {best_cost:.2f}")

            except Exception as e:
                print(f" failed: {e}")
                traceback.print_exc()
                err_row = {
                    "instance": fname, 
                    "cooling_rate": cr, 
                    "time_limit_config": tl, 
                    "status": "Error",
                    "error_msg": str(e)
                }
                save_single_row(err_row, OUTPUT_FILE)

            finally:
                if alns: del alns
                gc.collect()

    print("\n Tuning finished.")

if __name__ == "__main__":
    run_tuning()