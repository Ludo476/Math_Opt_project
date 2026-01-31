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

from src.milp_model.milp_model_simple import solve_milp_crowdshipping_simple
from src.milp_model.milp_model_cuts import solve_milp_crowdshipping
from src.milp_model.milp_model_callback import solve_milp_crowdshipping_callback

BASE_INSTANCES = os.path.join(PROJECT_ROOT, "instances_scalability_new") #for larger classes instance_scalability + class 2-5
BASE_RESULTS =os.path.join(PROJECT_ROOT,"results_scalability_prova")

# Configurazione del Run
SELECTED_CLASSES = [0, 1, 2]     # classes 0,1,2,6 are okay, the others too big (also 2 requires a lof of time tu run)
SELECTED_FAMILIES = [0, 1, 2]  #These are the families tested (indicate 24, 36, 44 stations)
NUM_INSTANCES = 4               #instances sharing exactly the same dim 

MILP_TIMELIMIT = 3600

# Solver list
SOLVERS = [
    ("MILP_Base",     solve_milp_crowdshipping_simple,   "global_scalability_results_milp_base_prova.csv"),
    ("MILP_Cuts",     solve_milp_crowdshipping,          "global_scalability_results_milp_cuts_prova.csv"),
    ("MILP_Callback", solve_milp_crowdshipping_callback, "global_scalability_results_milp_callback_prova.csv")
]

# =============================================================================
# HELPERS
# =============================================================================
def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 ** 3) # GB

def safe_float(x):
    return float(x) if x is not None else float('nan')

def save_single_row(row_dict, filepath):
    df = pd.DataFrame([row_dict])
    file_exists = os.path.isfile(filepath)
    try:
        df.to_csv(filepath, mode='a', index=False, header=not file_exists)
    except Exception as e:
        print(f"error saving CSV: {e}")

def run_experiment():
    os.makedirs(BASE_RESULTS, exist_ok=True)

    for s_name, s_func, csv_filename in SOLVERS:
        
        output_path = os.path.join(BASE_RESULTS, csv_filename)
        print(f"\n Starting analysis: {s_name} -> saving results in {csv_filename}")
        
        processed_keys = set()
        if os.path.exists(output_path):
            try:
                existing = pd.read_csv(output_path)
                processed_keys = set(existing['instance'].unique())
            except: pass

        for class_id in SELECTED_CLASSES:
            class_path = os.path.join(BASE_INSTANCES, f"class_{class_id}")
            if not os.path.exists(class_path): continue

            for family_id in SELECTED_FAMILIES:
                for inst_id in range(NUM_INSTANCES):

                    fname = f"class_{class_id}_fam_{family_id}_inst_{inst_id}.json"
                    if fname in processed_keys:
                        print(f"{fname} already done. Skipping.")
                        continue

                    fpath = os.path.join(class_path, fname)
                    if not os.path.exists(fpath): continue

                    print(f" {s_name} on {fname}...", end=" ", flush=True)

                    with open(fpath) as f:
                        instance_data = json.load(f)

                    mem_before = get_memory_usage()
                    start_wall = time.time()
                    res = None
                    
                    try:
                        # execution
                        res = s_func(instance_data, timelimit=MILP_TIMELIMIT)
                        
                        runtime = time.time() - start_wall
                        mem_after = get_memory_usage()
                        
                        row = {
                            "instance": fname,
                            "class": class_id,
                            "family": family_id,
                            "solver": s_name,
                            "status": res.get("status"),
                            #Quality
                            "objective": safe_float(res.get("objective")),
                            "gap": safe_float(res.get("mip_gap")),
                            "root_bound_after_cuts": safe_float(res.get("root_bound_after_cuts")),
                            "integrality_gap_root": safe_float(res.get("integrality_gap_root")),
                            # Performance
                            "time_s": runtime,
                            "time_to_best_s": res.get("time_to_best_s"),
                            "nodes": res.get("nodes_explored", 0),
                            "presolve_time": res.get("presolve_time"),
                            "memory_delta_gb": mem_after - mem_before,
                            # Model Size
                            "num_vars": res.get("num_vars"),
                            "num_constrs": res.get("num_constrs"),
                            # Solution Details
                            "percent_cs_delivery": res.get("percent_cs_delivery"),
                            "num_delivered_parcels": res.get("num_delivered_parcels"),
                            "num_parcels": len(instance_data["demand"]["parcels"]),
                            "num_cs": len(instance_data["network"]["crowdshippers"]),
                            # HISTORY 
                            "history": json.dumps(res.get("history", []))   
                        }

                        #  CUTS (only callback) 
                        if s_name == "MILP_Callback":
                            cuts_stats = res.get("cuts_stats", {})
                            row["cuts_added"] = res.get("cuts_added", 0)
                            
                            # columns for cuts (VI1, VI2...)
                            for i in range(1, 10):
                                vi_key = f"VI{i}"
                                row[f"cuts_{vi_key}"] = cuts_stats.get(vi_key, 0)

                        save_single_row(row, output_path)
                        print(f"{res.get('status')} ({runtime:.1f}s)")

                    except Exception as e:
                        print(f"ERROR: {e}")
                        err_row = {"instance": fname, "solver": s_name, "status": "Error"}
                        save_single_row(err_row, output_path)
                    
                    finally:
                        if res: del res
                        gc.collect()


    print("\n MILP analysis completed.")

if __name__ == "__main__":
    run_experiment()