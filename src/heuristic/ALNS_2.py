import time
import random
import math
import copy
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List
from tqdm import tqdm

from .helpers import Helpers
from .operators.destroy_operators import DestroyOperators
from .operators.repair_operators import RepairOperators

class ALNSSolver:
    """
    ALNS Solver for the Parcel Transportation Problem with Crowdshippers (PTCP).
    Implements the Inhomogeneous Simulated Annealing (ISA) framework.
    """
    def __init__(self, instance_data: Dict[str, Any], time_limit=60, t_min=1e-4, reheat_delta=5.0, cooling_rate=0.9, seed=42):
        print("[DEBUG] ALNS.__init__ Start")
        sys.stdout.flush()

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.instance_data = instance_data
        self.time_limit = time_limit
        self.t_min = t_min      
        self.alpha = cooling_rate 
        self.reheat_delta = reheat_delta

        # ID mapping
        crowdshippers = self.instance_data['network'].get('crowdshippers', [])
        for idx, cs in enumerate(crowdshippers):
            if 'id' not in cs: cs['id'] = idx

        # --- INIT HELPERS ---
        print("[DEBUG] Initializing Helpers...")
        self.helpers = Helpers(instance_data)
        self.parcels_map = self.helpers.parcels_map
        self.stations_map = self.helpers.stations_map
        
        # --- Precompute Graph for Repair ---
        print("[DEBUG] Building st_adj in ALNS...")
        self.st_adj = defaultdict(list)
        net = self.instance_data['network']
        
        # Safe parser for Kit_map tuples
        def parse_tuple(x):
            if isinstance(x, tuple): return x
            #to deal with "(1, 2)" o "1, 2"
            return tuple(map(int, x.strip("()[] ").split(",")))

        self.Kit_map = {parse_tuple(k): v for k, v in net.get('Kit_map', {}).items()}
              
        for (u, v), k_map in self.helpers.arc_time_map.items():
            for k, (t_start, t_end) in k_map.items():
                if k in self.Kit_map.get((u, t_start), []):
                    self.st_adj[(u, t_start)].append((v, t_end, k))
        
        # Cost parameters 
        self.fixed_costs = {int(k): v for k, v in self.instance_data['costs']['fixed_source_delivery_cost'].items()}
        self.rho = self.instance_data['costs']['crowdshipper_reward'] 

        # --- OPERATORS INIT ---
        self.destroy_op = DestroyOperators(instance_data, self.helpers)
        self.repair_op = RepairOperators(instance_data, self.helpers, self.parcels_map, self.st_adj, self)

        # --- SOLUTION STATE ---
        self.current_solution = None
        self.best_solution = None

        # --- OPERATOR MAPPING ---
        self.destroy_operator_funcs = {
            "path_elimination": lambda sol, **kwargs: 
                self.destroy_op.destroy_by_parcel_path_elimination(sol, metric_name="random", **kwargs),

            "capacity_reduction": lambda sol, **kwargs: 
                self.destroy_op.destroy_by_capacity_reduction(sol, metric_name="random", **kwargs),
        }

        self.repair_operator_funcs = {
            "repair_R1": lambda sol, **kwargs: self.repair_op.repair_R1(sol, **kwargs),
            "repair_R2": lambda sol, **kwargs: self.repair_op.repair_R2(sol, **kwargs),
            "repair_R3": lambda sol, **kwargs: self.repair_op.repair_R3(sol, **kwargs),
            "repair_R4": lambda sol, **kwargs: self.repair_op.repair_R4(sol, **kwargs),
        }
        
        # ALNS Weights & Scores
        self.destroy_weights = {op: 1.0 for op in self.destroy_operator_funcs}
        self.repair_weights = {op: 1.0 for op in self.repair_operator_funcs}
        self.destroy_scores = {op: 0.0 for op in self.destroy_operator_funcs}
        self.repair_scores = {op: 0.0 for op in self.repair_operator_funcs}
        self.destroy_counts = {op: 0 for op in self.destroy_operator_funcs}
        self.repair_counts = {op: 0 for op in self.repair_operator_funcs}
   
        print("[DEBUG] ALNS.__init__ Done")

    # ====================================================================
    #  PRETTY PRINT
    # ====================================================================
    def pretty_print_solution(self, solution, final_cost, instance_data):
        """
        Outputs the ALNS solution in a readable format.
        """
        if not solution:
            print("\n [WARN] Invalid or Empty Solution passed to pretty_print")
            return
        
        print("\n" + "="*40)
        print("       ALNS SOLUTION REPORT")
        print("="*40)

        station_data = {s['id']: s for s in instance_data['network']['stations']}
        parcel_data = {p['id']: p for p in instance_data['demand']['parcels']}
        fixed_source_delivery_cost_data = instance_data['costs']['fixed_source_delivery_cost']
        rho = instance_data.get('costs', {}).get('crowdshipper_reward', 1)

        delivered_parcels = solution.get('parcels_assigned', {})
        backup_parcels_ids = solution.get('backup_parcels', set())
      
        # Costs breakdown
        total_backup_cost = sum(parcel_data[p_id].get('backup_cost', 0.0) for p_id in backup_parcels_ids)
        
        total_loading_cost = 0.0
        for p_id, data in delivered_parcels.items():
            s_id = data['origin_station']
            total_loading_cost += station_data.get(s_id, {}).get('loading_cost', 0.0)

        used_source_stations = set(data['origin_station'] for data in delivered_parcels.values())
        total_fixed_delivery_cost = sum(fixed_source_delivery_cost_data.get(str(s_id), 0.0) for s_id in used_source_stations)

        crowdshippers_used = set()
        for assignment in delivered_parcels.values():
            for segment in assignment.get('path', []):
                if len(segment) >= 3 and segment[2] is not None:
                    crowdshippers_used.add(segment[2])
        total_cs_remuneration = rho * len(crowdshippers_used)

        # Summary Statistics
        print(f"Objective Value: {final_cost:.3f}")
        print(f"  - Crowdshipper Reward: {total_cs_remuneration:.2f}")
        print(f"  - Fixed Setup Cost:    {total_fixed_delivery_cost:.2f}")
        print(f"  - Loading Cost:        {total_loading_cost:.2f}")
        print(f"  - Backup Service Cost: {total_backup_cost:.2f}")
        print("-" * 40)
        print(f"Parcels Delivered: {len(delivered_parcels)} | Backup: {len(backup_parcels_ids)}")
        print(f"Active Crowdshippers: {len(crowdshippers_used)}")
        print("="*40 + "\n")

        #route details
        print("Assigned Parcels:")
        print("─" * 20)
        for parcel_id, data in delivered_parcels.items():
            path_segments = [
                f"{u} -> {v} (cs={cs})" if cs is not None else f"{u} -> {v}"
                for u, v, cs, *_ in data['path']
            ]
            print(f"  - Parcel {parcel_id}: {data['origin_station']} -> {data['destination']}")
            print(f"    Path: {', '.join(path_segments)}\n")
            
            
        print(f"Backup parcels: {backup_parcels_ids}\n")

        #details crowdshippers
        print("Crowdshippers used:")
        print("─" * 27)
        used_cs_data = {cs_id: arcs for cs_id, arcs in solution.get('crowdshipper_assignments', {}).items() if arcs}
        for cs_id in sorted(used_cs_data.keys()):
            print(f"  - Crowdshipper {cs_id}:")
            for (parcel_id, from_node, to_node, ts, te) in used_cs_data[cs_id]:
                print(f"    - Parcel {parcel_id}: {from_node} -> {to_node} (Time {ts} to {te})")
        
        print("\n" + "─" * 27)


    # ====================================================================
    #  CORE METHODS
    # ====================================================================

    def destroy_solution(self, solution, op_name, **kwargs):
        destroy_func = self.destroy_operator_funcs.get(op_name)
        valid_args = {"q", "num_stations_to_select", "reduction_amount"}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
        return destroy_func(solution, **filtered_kwargs)

    def repair_solution(self, solution, op_name, **kwargs):
        repair_func = self.repair_operator_funcs.get(op_name)
        return repair_func(solution, **kwargs)
    
    def cih_constructive_heuristic(self, instance_data):
        initial_solution = {
            'parcels_assigned': {},
            'crowdshipper_assignments': {cs['id']: [] for cs in self.instance_data['network']['crowdshippers']},
            'backup_parcels': {parcel['id'] for parcel in instance_data['demand']['parcels']},
            'solution_cost': 0
        }
        return self.repair_op.repair_R1(initial_solution, order_strategy="As-Is")
    
    def _initialize_temperature(self, sample_size=40, accept_prob=0.8):
        current_cost = self._calculate_cost(self.current_solution)
        deltas = []
        for _ in range(sample_size):
            d_op = random.choice(list(self.destroy_operator_funcs.keys()))
            r_op = random.choice(list(self.repair_operator_funcs.keys()))
            kw_dest = {"q": 2, "num_stations_to_select": 2, "reduction_amount": 2}
            try:
                tmp_sol = self.destroy_solution(copy.deepcopy(self.current_solution), d_op, **kw_dest)
                tmp_sol = self.repair_solution(tmp_sol, r_op, order_strategy="Random")
                delta = self._calculate_cost(tmp_sol) - current_cost
                if delta > 0: deltas.append(delta)
            except: continue

        avg_delta = sum(deltas) / len(deltas) if deltas else max(10.0, current_cost * 0.05)
        T0 = -avg_delta / math.log(accept_prob) if accept_prob > 0 and accept_prob < 1 else max(1.0, avg_delta)
        return max(T0, 0.1)

    def run(self, q_D1=25, q_D2=5, max_reheats=1000, update_interval=50, no_improve_limit=500,
        cooling_rate=0.9975, time_limit=None, sample_size=30, accept_prob=0.7,stop_if_optimal=True, stagnation_limit=None):
        
        self.q_D1 = q_D1 
        self.q_D2 = q_D2
        if time_limit is not None: 
            self.time_limit = time_limit
        if cooling_rate: self.alpha = cooling_rate

        self.history = [] 
        start_time = time.time()

        print(f"\n[INIT] Generating Initial Solution (CIH)...")
        self.current_solution = self.cih_constructive_heuristic(self.instance_data)
        self.current_solution['solution_cost'] = self._calculate_cost(self.current_solution) 
        self.initial_cost = self.current_solution['solution_cost']
        
        # print(f"[INIT] CIH Cost: {self.initial_cost:.2f}")
        self.history.append((0.0, self.initial_cost))
        self.temperature = self._initialize_temperature(sample_size=sample_size, accept_prob=accept_prob)
        # print(f"[INIT] Start Temp: {self.temperature:.4f}")
    
        self.best_solution = copy.deepcopy(self.current_solution)
        self.best_solution['time_to_best_s'] = 0.0
        
        iteration_counter = 0
        reheat_counter = 0
        last_improvement_iter = 0   
        self.n_improving_sols = 0
        self.iter_to_best = 0
        global_stagnation_counter = 0

        pbar = tqdm(total=self.time_limit, desc="ALNS", unit="s")
        
        while (time.time() - start_time) < self.time_limit:

            if stop_if_optimal and self.best_solution['solution_cost'] <= 1e-6:
                print(" Optimal solution found (Cost 0). Stopping.")
                break
            if stagnation_limit and (iteration_counter - self.iter_to_best) > stagnation_limit:
                print(f"  Stagnation limit reached ({stagnation_limit} iters without global improvement). Stopping.")
                break
            
            d_name = self._select_operator("destroy")
            r_name = self._select_operator("repair")
            if random.random() < 0.1:
                d_name = random.choice(list(self.destroy_weights.keys()))
                r_name = random.choice(list(self.repair_weights.keys()))

            kwargs_destroy = {}
            if d_name == "path_elimination":
                act_q = random.choice([12, 15, 25])
                kwargs_destroy = {"q": act_q}
            elif d_name == "capacity_reduction":
                act_q = random.randint(1, 5)
                kwargs_destroy = {"num_stations_to_select": act_q, "reduction_amount": act_q}
            
            strategy_order = random.choice(["As-Is", "Random", "Max Backup", "Min Backup"])

            try:
                destroyed = self.destroy_solution(self.current_solution, d_name, **kwargs_destroy)
                new_sol = self.repair_solution(destroyed, r_name, order_strategy=strategy_order)
                if 'solution_cost' not in new_sol:
                    new_sol['solution_cost'] = self._calculate_cost(new_sol)
            except Exception as e:
                continue

            cost_new = new_sol['solution_cost']
            cost_curr = self.current_solution['solution_cost']
            cost_best = self.best_solution['solution_cost']
            
            delta = cost_new - cost_curr
            outcome = "rejected"

            if delta < 0 or random.random() < math.exp(-delta / self.temperature):
                self.current_solution = new_sol
                outcome = "accepted"

                if delta < 0:
                    self.n_improving_sols += 1  
                    outcome = "better_than_current"

                if cost_new < cost_best:
                    self.best_solution = copy.deepcopy(new_sol)

                    current_time = time.time() - start_time
                    self.best_solution['time_to_best_s'] = current_time
                    self.history.append((current_time, cost_new))

                    last_improvement_iter = iteration_counter
                    self.iter_to_best = iteration_counter
                    outcome = "best_found"

                elif delta < 0:
                    outcome = "better_than_current"

            self._update_scores(d_name, r_name, outcome)
            
            if iteration_counter % update_interval == 0:
                self._update_weights()

            self.temperature *= self.alpha
            iteration_counter += 1

            if (iteration_counter - last_improvement_iter) > no_improve_limit or self.temperature < self.t_min:
                self.current_solution = copy.deepcopy(self.best_solution)
                t_reinit = self._initialize_temperature(sample_size=min(10, sample_size), accept_prob=0.5)
                self.temperature = max(self.temperature * 2.5, t_reinit, 0.1)
                try:
                    # Diversification step
                    d_short = random.choice(list(self.destroy_operator_funcs.keys()))
                    r_short = random.choice(list(self.repair_operator_funcs.keys()))
                    small_q = max(2, min(8, self.q_D1 // 2))
                    destroyed_short = self.destroy_solution(copy.deepcopy(self.current_solution), d_short, q=small_q)
                    cand = self.repair_solution(destroyed_short, r_short, order_strategy="Random")
                    if 'solution_cost' not in cand:
                        cand['solution_cost'] = self._calculate_cost(cand)
                        self.current_solution = cand
                except Exception: pass

                reheat_counter += 1

            elapsed = time.time() - start_time
            pbar.n = min(int(elapsed), self.time_limit)
            pbar.set_postfix({"Best": f"{self.best_solution['solution_cost']:.1f}", 
                              "Curr": f"{cost_new:.1f}",
                              "T": f"{self.temperature:.4f}"})
            pbar.refresh()

        pbar.close()

        num_p = len(self.instance_data['demand']['parcels'])
        num_del = len(self.best_solution.get('parcels_assigned', {}))
        self.best_solution['percent_cs_delivery'] = (num_del / num_p) * 100 if num_p > 0 else 0
        self.best_solution['iter_to_best'] = self.iter_to_best
        self.best_solution['total_iterations'] = iteration_counter
        self.best_solution['reheats'] = reheat_counter
        self.best_solution['n_improving_sols'] = self.n_improving_sols
        self.best_solution['initial_cost'] = self.initial_cost 
        self.best_solution['history'] = self.history
        
        return self.best_solution, self.best_solution['solution_cost']

    def _calculate_cost(self, solution):
        if not solution: return float('inf')
        total_cost = 0.0
        
        for data in solution.get('parcels_assigned', {}).values():
            s_id = data['origin_station']
            total_cost += self.stations_map[s_id].get('loading_cost', 0.0)
        
        used_sources = {d['origin_station'] for d in solution.get('parcels_assigned', {}).values()}
        for s_id in used_sources:
            total_cost += self.fixed_costs.get(s_id, 0.0)
            
        used_cs = set()
        for data in solution.get('parcels_assigned', {}).values():
            for seg in data['path']:
                if len(seg) >= 3 and seg[2] is not None:
                    used_cs.add(seg[2])
        total_cost += len(used_cs) * self.rho
        
        for p_id in solution.get('backup_parcels', []):
            total_cost += self.parcels_map[p_id].get('backup_cost', 0.0)
            
        return total_cost

    def _select_operator(self, op_type="destroy"):
        if op_type == "destroy": weights = self.destroy_weights
        else: weights = self.repair_weights
        
        ops = list(weights.keys())
        w_vals = list(weights.values())
        return random.choices(ops, weights=w_vals, k=1)[0]

    def _update_scores(self, d_op, r_op, outcome):
        rewards = {"best_found": 10, "better_than_current": 5, "accepted": 2, "rejected": 0}
        score = rewards.get(outcome, 0)
        self.destroy_scores[d_op] += score
        self.repair_scores[r_op] += score
        self.destroy_counts[d_op] += 1
        self.repair_counts[r_op] += 1

    def _update_weights(self, reaction=0.1):
        for op in self.destroy_weights:
            if self.destroy_counts[op] > 0:
                avg = self.destroy_scores[op] / self.destroy_counts[op]
                self.destroy_weights[op] = self.destroy_weights[op] * (1-reaction) + avg * reaction
                
        for op in self.repair_weights:
            if self.repair_counts[op] > 0:
                avg = self.repair_scores[op] / self.repair_counts[op]
                self.repair_weights[op] = self.repair_weights[op] * (1-reaction) + avg * reaction
        
        self.destroy_scores = {k: 0.0 for k in self.destroy_scores}
        self.repair_scores = {k: 0.0 for k in self.repair_scores}
        self.destroy_counts = {k: 0 for k in self.destroy_counts}
        self.repair_counts = {k: 0 for k in self.repair_counts}