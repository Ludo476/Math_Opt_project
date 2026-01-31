import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import ast
import json
import os, traceback
import time



def pretty_print_milp(model, results, instance_data, vars_dict):
    """
    Uniforms the MILP output to match the ALNS pretty_print style.
    vars_dict should contain: 'z_k', 'q_p', 's_ip', 'x_ij_kp'
    """
    if model.SolCount == 0:
        print("\n No MILP Solution found.")
        return

    print("\n" + "="*30)
    print(" MILP Execution Completed.")
    print("="*30)
    
    # Header e Statistiche Summary
    print(f"Objective Value: {results['objective']:.3f}")
    print(f"  - Total Crowdshipper Remuneration: {results['total_crowdshipper_reward']:.2f}")
    print(f"  - Total Fixed Setup Cost: {results['total_fixed_cost']:.2f}")
    print(f"  - Total Loading Cost: {results['total_loading_cost']:.2f}")
    print(f"  - Total Backup Service Cost: {results['total_backup_cost']:.2f}")
    
    num_delivered = results['num_delivered_parcels']
    num_backup = len(instance_data['demand']['parcels']) - num_delivered
    print(f"Parcels Delivered: {num_delivered} | Backup: {num_backup}")
    print(f"Unique Crowdshippers Active: {results['num_crowdshippers_used']}\n")

    # detail paths
    print("Assigned Parcels (Optimal Routes):")
    print("─" * 20)
    
    x = vars_dict['x_ij_kp']
    s_ip = vars_dict['s_ip']
    
    # Raggruppiamo gli archi per pacco per ricostruire il path
    parcel_paths = {p['id']: [] for p in instance_data['demand']['parcels'] if results['objective'] < 1e9} # logic for q_p < 0.5
    
    for (i, j, k, p), var in x.items():
        if var.X > 0.5:
            parcel_paths[p].append(f"{i} -> {j} (cs={k})")

    for p_id, path_segments in parcel_paths.items():
        if path_segments:
            # Trova l'origine (s_ip)
            origin = "Unknown"
            for i in [s['id'] for s in instance_data['network']['stations'] if s['is_source']]:
                if s_ip[i, p_id].X > 0.5:
                    origin = i
                    break
            print(f"  - Parcel {p_id}: origin {origin}")
            print(f"    Path: {', '.join(path_segments)}\n")

    # Dettaglio Backup
    backup_ids = [p['id'] for p in instance_data['demand']['parcels'] if vars_dict['q_p'][p['id']].X > 0.5]
    print(f"Backup parcels: {set(backup_ids)}\n")

    # Dettaglio Crowdshippers
    print("Crowdshippers used:")
    print("─" * 27)
    z = vars_dict['z_k']
    for k in sorted(z.keys()):
        if z[k].X > 0.5:
            print(f"  - Crowdshipper {k} is active.")
            # Archi percorsi da questo CS
            for (i, j, k_idx, p), var in x.items():
                if k_idx == k and var.X > 0.5:
                    print(f"    - Parcel {p}: {i} -> {j}")
    print("\n" + "─" * 27)

def ptcp_callback(model, where):
    # --- 1. TIME TO BEST (TTB) ---
    if where == GRB.Callback.MIPSOL:
        # value of the new solution found
        new_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        current_best = model._best_obj
        
        # if it is the first solution or better than the previous
        if current_best is None or new_obj < current_best - 1e-6:
            current_time = time.time() - model._start_time

            model._best_obj = new_obj
            model._ttb = current_time

            if hasattr(model, '_solution_history'):
                model._solution_history.append((current_time, new_obj))

    # --- 2. ROOT RELAXATION (Pure LP vs Root Bound) ---
    if where == GRB.Callback.MIPNODE:
        # verify you're in the root node
        if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
            # actual relaxation
            node_status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if node_status == GRB.OPTIMAL:
                if model._lp_relaxation is None:
                    model._lp_relaxation = model.cbGet(GRB.Callback.MIPNODE_OBJBND)

                # 2. Catch the Root Bound 
                model._root_bound_after_cuts = model.cbGet(GRB.Callback.MIPNODE_OBJBND)