import os
import sys
import json
import pandas as pd
from gurobipy import GRB

# ==========================================
# 1. SETUP PATHS
# ==========================================

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_generation.data_generator import DataGenerator, generate_realistic_txt_dataset
from src.data_generation.data_generator import preprocess_instance_from_txt
from src.milp_model.milp_model_cuts import solve_milp_crowdshipping
from src.milp_model.milp_model_callback import solve_milp_crowdshipping_callback
from src.milp_model.milp_model_simple import solve_milp_crowdshipping_simple
from src.heuristic.ALNS_2 import ALNSSolver

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def make_json_safe(data):
    """to convert data in a compatible JSON format."""
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, (list, tuple, set)):
        return [make_json_safe(item) for item in data]
    elif isinstance(data, dict):
        return {str(k): make_json_safe(v) for k, v in data.items()}
    return str(data)



def setup_instance(params, filename_base="instance_test_small"): #I'm using this instance I generated, but who tests can rigenerate 
    """Generate or load the instance"""
    instances_dir = os.path.join(project_root, "instances") 
    #here is the folder where I put images for small tests
    os.makedirs(instances_dir, exist_ok=True)

    txt_path = os.path.join(instances_dir, f"{filename_base}.txt")
    json_path = os.path.join(instances_dir, f"{filename_base}.json")

    if os.path.exists(json_path):
        print(f" JSON instance found. Loading from: {json_path}")
        with open(json_path, "r") as f:
            return json.load(f)
    
    print(" Instance not found. Generating new dataset...")
    generator = DataGenerator()
    dataset_txt = generate_realistic_txt_dataset(generator=generator, **params)
    
    with open(txt_path, "w") as f:
        f.write(dataset_txt)
    #I generate it first in the txt format, to make it more clear
    print(f"TXT saved in: {txt_path}") 

    instance_data = preprocess_instance_from_txt(txt_path)

    with open(json_path, "w") as f:
        json.dump(make_json_safe(instance_data), f, indent=4)
    # My solvers work with JSON format instances
    print(f"JSON saved in: {json_path}")
    
    return instance_data

def run_milp_solver(solver_func, name, data, time_limit):
    """Wrapper to execute the different MILP solvers and manage their outputs."""
    print(f"\n--- Start {name} (Time Limit: {time_limit}s) ---")
    results = solver_func(data, timelimit=time_limit)
    
    cost = float('inf')
    status = "Not Found"
    
    if results:
        # To manage Gurobi states
        if results.get('status') == GRB.OPTIMAL:
            status = "Optimal"
            cost = results['objective']
        elif results.get('status') == GRB.TIME_LIMIT:
            status = "Time Limit"
            cost = results.get('objective', float('inf'))
        
        print(f"   -> Status: {status}")
        print(f"   -> Objective: {cost:.2f}")
    else:
        print("   -> No results returned.")
        
    return cost

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    print("Starting small instance test script")
    
    # 1. Configuration parameters instance (I find these are the right parameters to run small but not trivial tests)
    params = {
        "num_lines": 6, #number of network lines
        "num_stations": 36, #number of network stations
        "num_exchange_stations": 6, #number of exchange stations: stations shared among different lines
        "num_cs": 55, #number of crowdshippers
        "num_parcels": 35, #number of parcels to deliver
        "step_size": 5, #time discretization (keep it fixed)
        "step_count": 85
    }
    
    MILP_TIME_LIMIT = 3600
    ALNS_TIME_LIMIT = 300 #Since this is a quick test 5 minutes are okay, otherwise better with 10 minutes (600 s)

    # 2. Loading data
    instance_data = setup_instance(params)

    # 3. Execution MILP Models
    # MILP Base (integrated cuts) 
    cost_cuts = run_milp_solver(solve_milp_crowdshipping, "MILP (Cuts)", instance_data, MILP_TIME_LIMIT)
    
    # MILP Base (no valid inequalities)
    cost_simple = run_milp_solver(solve_milp_crowdshipping_simple, "MILP (Simple)", instance_data, MILP_TIME_LIMIT)
    
    # MILP Callback (dynamically added cuts)
    cost_callback = run_milp_solver(solve_milp_crowdshipping_callback, "MILP (Callback)", instance_data, MILP_TIME_LIMIT)

    # 4. Execution ALNS
    print(f"\n--- Start ALNS Heuristic (Time Limit: {ALNS_TIME_LIMIT}s) ---")
    alns_solver = ALNSSolver(instance_data, time_limit=ALNS_TIME_LIMIT)
    best_solution_alns, cost_alns = alns_solver.run()

    if best_solution_alns:
        print(f"   -> Solution Found!")
        print(f"   -> ALNS Cost: {cost_alns:.2f}")
        # Pretty print 
        print("\n   [ALNS Solution Details]")
        alns_solver.pretty_print_solution(best_solution_alns, cost_alns, instance_data)
    else:
        print("   -> ALNS did not find a valid solution.")
        cost_alns = float('inf')

    # 5. Creation comparison table
    print("\n" + "="*50)
    print(" FINAL COMPARISON")
    print("="*50)

    # helper  function to compute the gap (with respect to the baseline- MILP without cuts)
    def calc_gap(val, baseline):
        if baseline == 0 or baseline == float('inf') or val == float('inf'):
            return "N/A"
        return ((val - baseline) / baseline) * 100

    comparison_data = {
        "Method": [
            "MILP (Simple - Baseline)", 
            "MILP (Cuts)", 
            "MILP (Callback)", 
            "ALNS Heuristic"
        ],
        "Total Cost": [
            cost_simple,
            cost_cuts, 
            cost_callback, 
            cost_alns
        ],
        "Gap (%)": [
            0.0, 
            calc_gap(cost_cuts, cost_simple),
            calc_gap(cost_callback, cost_simple),
            calc_gap(cost_alns, cost_simple)
        ],
        "Time Limit (s)": [
            MILP_TIME_LIMIT, 
            MILP_TIME_LIMIT, 
            MILP_TIME_LIMIT, 
            ALNS_TIME_LIMIT
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    
    # Formatting output
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df_comparison.to_string(index=False))
    print("="*50)

    # 6. Final considerations
    if cost_simple != float('inf') and cost_alns != float('inf'):
        if cost_alns <= cost_simple * 1.05:
            print("\n EXCELLENT: The heuristic is within 5% of the MILP optimum.")
        elif cost_alns <= cost_simple * 1.15:
            print("\n GOOD: The heuristic is within 15% of the MILP optimum.")
        else:
            print("\n IMPROVEMENT NEEDED: The gap is significant.")
    else:
        print("\nℹ Cannot compare gaps due to invalid/infinite solutions.")

if __name__ == "__main__":
    main()