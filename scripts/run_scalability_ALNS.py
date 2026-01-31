import os
import json
import psutil
import pandas as pd
import time
import gc
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.heuristic.ALNS_2 import ALNSSolver 

BASE_INSTANCES = os.path.join(PROJECT_ROOT, "instances_scalability_new")
BASE_RESULTS = os.path.join(PROJECT_ROOT, "results_scalability_prova")
OUTPUT_FILE = os.path.join(BASE_RESULTS, "global_scalability_results_alns_prova.csv")

SELECTED_CLASSES = [0, 1, 2] #the instances tested
SELECTED_FAMILIES = [0, 1, 2]  #24,36,44 stations
NUM_INSTANCES = 4

ALNS_TIMELIMIT = 600 #I find this is the best tradeoff between quality and speed
COOLING_RATE = 0.95 #0.75 
SEEDS = [42, 123, 456, 101, 999] #I think these are enough but we can add to have a more robust analysis

def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 ** 3)

def safe_float(x):
    return float(x) if x is not None else float('nan')

def save_single_row(row_dict, filepath):
    df = pd.DataFrame([row_dict])
    file_exists = os.path.isfile(filepath)
    try:
        df.to_csv(filepath, mode='a', index=False, header=not file_exists)
    except Exception as e:
        print(f"error saving CSV: {e}")

# =============================================================================
# MAIN LOOP
# =============================================================================

def run_experiment():
    os.makedirs(BASE_RESULTS, exist_ok=True)

    processed_runs = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            for _, r in existing.iterrows():
                processed_runs.add(f"{r['instance']}_{r['seed']}")
        except: pass
    
    total_runs = len(SELECTED_CLASSES) * len(SELECTED_FAMILIES) * NUM_INSTANCES * len(SEEDS)
    current_run_count = 0

    print(f"Starting ALNS ({total_runs} total runs) -> {OUTPUT_FILE}")

    for class_id in SELECTED_CLASSES:
        class_path = os.path.join(BASE_INSTANCES, f"class_{class_id}")
        if not os.path.exists(class_path): continue

        for family_id in SELECTED_FAMILIES:
            for inst_id in range(NUM_INSTANCES):

                fname = f"class_{class_id}_fam_{family_id}_inst_{inst_id}.json"
                fpath = os.path.join(class_path, fname)
                if not os.path.exists(fpath): continue

                with open(fpath) as f:
                    instance_data = json.load(f)

                for seed in SEEDS:
                    current_run_count += 1

                    if f"{fname}_{seed}" in processed_runs:
                        print(f"[{current_run_count}/{total_runs}] {fname} Seed {seed} already done.")
                        continue
                    print(f"[{current_run_count}/{total_runs}] {fname} | Seed {seed}...", end=" ", flush=True)

                    mem_before = get_memory_usage()
                    start_wall = time.time()
                    alns = None

                    try:
                        alns = ALNSSolver(instance_data, time_limit=ALNS_TIMELIMIT, seed=seed)
                        best_sol, best_cost = alns.run(cooling_rate=COOLING_RATE)
                        
                        runtime = time.time() - start_wall
                        mem_after = get_memory_usage()

                        # improving rate %
                        initial_cost = best_sol.get('initial_cost', alns.initial_cost)
                        improv_pct = 0.0
                        if initial_cost and initial_cost > 0:
                            improv_pct = ((initial_cost - best_cost) / initial_cost) * 100

                        row = {
                            "instance": fname,
                            "class": class_id,
                            "family": family_id,
                            "solver": "ALNS",
                            "seed": seed,
                            # Quality
                            "objective": safe_float(best_cost),
                            "initial_cost": safe_float(initial_cost),
                            "improvement_pct": safe_float(improv_pct),
                            # Time & Performance
                            "time_s": runtime,
                            "time_to_best_s": safe_float(best_sol.get('time_to_best_s')),
                            "iterations": best_sol.get('total_iterations', 0),
                            "iter_to_best": best_sol.get('iter_to_best', 0),
                            "reheats": best_sol.get('reheats', 0),
                            "improving_sols": best_sol.get('n_improving_sols', 0),
                            "memory_delta_gb": mem_after - mem_before,
                            # Solution Structure
                            "percent_cs_delivery": safe_float(best_sol.get('percent_cs_delivery', 0)),
                            "num_delivered_parcels": len(best_sol.get('parcels_assigned', {})),
                            # Instance Size
                            "num_parcels": len(instance_data["demand"]["parcels"]),
                            "num_cs": len(instance_data["network"]["crowdshippers"]),
                            # For convergence plots: I'm saving the history as a string, but it's long so we can decomment only if needed
                            # "history": json.dumps(best_sol.get("history", [])) 
                            "history": best_sol.get("history", [])
                        }
                        
                        save_single_row(row, OUTPUT_FILE)
                        print(f"Obj: {best_cost:.2f}")

                    except Exception as e:
                        print(f"{e}")
                        err_row = {"instance": fname, "solver": "ALNS", "status": "Error", "seed": seed}
                        save_single_row(err_row, OUTPUT_FILE)

                    finally:
                        if alns: del alns
                        gc.collect()

    print("\n ALNS analysis completated")

if __name__ == "__main__":
    run_experiment()