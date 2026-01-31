import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import ast
import time
import json
import os, traceback

from .helpers_milp import pretty_print_milp

def solve_milp_crowdshipping_callback(instance_data, timelimit=None, nodelimit=None, N_nodes_interval=50):
    """
    MILP with callback to dynamically add valid inequalities.
    """
    try:
        model = gp.Model("PTCP_Crowdshipping_callback")
        if timelimit: model.Params.TimeLimit = timelimit

        # 1. DATA AND SETS  
        P = [p['id'] for p in instance_data['demand']['parcels']]
        N = instance_data["network"]["global_graph"]["N"]
        S = instance_data["network"]["global_graph"]["S"]
        D = instance_data["network"]["global_graph"]["D"]
        T = list(range(instance_data["time_windows"]["end"]))
        cs_list = instance_data["network"]["crowdshippers"]
        for idx, cs in enumerate(cs_list):
            if 'id' not in cs:
                cs['id'] = idx
        K = [c['id'] for c in cs_list]
        arc_time_k = {ast.literal_eval(k): v for k, v in instance_data["network"]["arc_time_k"].items()} 
        Kit_map = {ast.literal_eval(k): v for k, v in instance_data["network"]["Kit_map"].items()}
        K_ij_map = defaultdict(list)
        for (i,j,k) in arc_time_k.keys():
            K_ij_map[(i,j)].append(k)   
        D_p = {p['id']: p['destination'] for p in instance_data['demand']['parcels']}
        
        # adjacency maps
        arcs_in = defaultdict(list)
        arcs_out = defaultdict(list)
        for (u, v, k) in arc_time_k.keys():
            arcs_out[u].append((u, v, k))
            arcs_in[v].append((u, v, k))
        # Costs
        rho = instance_data['costs']['crowdshipper_reward']
        vi = {int(i): float(c) for i, c in instance_data['costs']['fixed_source_delivery_cost'].items()}
        sigma = {s["id"]: s.get("loading_cost", 0.0) for s in instance_data["network"]["stations"]}
        gamma = {p['id']: p.get('backup_cost', 0.0) for p in instance_data['demand']['parcels']}
        Cap = {s['id']: s.get('capacity', 0) for s in instance_data["network"]["stations"]}

        # Variables
        valid_keys = []
        for (i,j), ks in K_ij_map.items():
            for k in ks:
                for p in P:
                    valid_keys.append((i,j,k,p))
        #to reduce the dim of the model we add variables only for feasible arcs
        x_ij_kp = model.addVars(valid_keys, vtype=GRB.BINARY, name="x") 
        #binary variable taking value 1 if crowdshipper k transfers parcel p from station i to station j
        y_kp = model.addVars(K, P, vtype=GRB.BINARY, name="y")
        #binary variable taking value 1 if parcel p is assigned to crowdshipper k
        z_k  = model.addVars(K, vtype=GRB.BINARY, name="z")
        #binary variable taking value 1 if crowdshipper k is selected
        s_ip = model.addVars(N, P, vtype=GRB.BINARY, name="s")
        #binary variable taking value 1 if parcel p starts its route at source station i
        d_ip = model.addVars(N, P, vtype=GRB.BINARY, name="d")
        #binary variable taking value 1 if parcel p ebds its route at destination station i
        e_i  = model.addVars(S, vtype=GRB.BINARY, name="e")
        #binary variable taking value 1 if source station i is selected
        q_p  = model.addVars(P, vtype=GRB.BINARY, name="q")
        #binary variable taking value 1 if parcel p is delivered by the backup service
        l_ipt = model.addVars(N, P, T, vtype=GRB.BINARY, name="l")
        #binary variable taking value 1 if parcel p leaves station i at time slot t
        a_ipt = model.addVars(N, P, T, vtype=GRB.BINARY, name="a")
        #binary variable taking value 1 if parcel p arrives at station i at time slot t
        o_ipt = model.addVars(N, P, T, vtype=GRB.BINARY, name="o")
        #binary variable taking value 1 if parcel p is at station i at time slot t

        # save for callback
        model._x_ij_kp = x_ij_kp
        model._y_kp = y_kp
        model._z_k = z_k
        model._s_ip = s_ip
        model._d_ip = d_ip
        model._o_ipt = o_ipt
        model._a_ipt = a_ipt
        model._l_ipt = l_ipt
        model._e_i = e_i
        model._q_p = q_p
        model._N = N
        model._S = S
        model._D = D
        model._P = P
        model._K = K
        model._T = T
        model._vi = vi
        model._sigma = sigma
        model._gamma = gamma
        model._C = Cap
        model._D_p = D_p

        # OBJECTIVE FUNCTION (1)
        objective_expr = gp.LinExpr()
        objective_expr += z_k.sum() * rho #cost for using crowdshippers
        objective_expr += gp.quicksum(e_i[i] * vi.get(i,0) for i in S) #cost for activating source stations
        objective_expr += gp.quicksum(s_ip[i, p] * sigma.get(i,0) for i in S for p in P) #cost for loading parcels in APL stations
        objective_expr += gp.quicksum(q_p[p] * gamma.get(p,0) for p in P) #backup cost for using the LSP vehicles
        model.setObjective(objective_expr, GRB.MINIMIZE)

        # CONSTRAINTS
        #(2) if the LSP does not deliver parcel p through the backup service the parcel must have exactly one source station
        model.addConstrs((gp.quicksum(s_ip[i, p] for i in S) == 1 - q_p[p] for p in P), name="V2_source")
        #(3) if the LSP does not deliver parcel p through the backup service the parcel must have exactly one destination station
        model.addConstrs((gp.quicksum(d_ip[i, p] for i in D_p.get(p, []) if i in D) == 1 - q_p[p] for p in P), name="V3_dest")
        #(4) set e_i equal to 1 if station i is used as a source station by at least one parcel
        model.addConstrs((gp.quicksum(s_ip[i, p] for p in P) <= Cap[i]*e_i[i] for i in S if Cap[i] != float('inf')), name="V4_capacity")
        #(5) flow conservation
        for p in P:
            for i in N:
                outbound = gp.quicksum(
                    x_ij_kp[i,j,k,p] for j in N for k in K_ij_map.get((i,j), [])
                )
                inbound = gp.quicksum(
                    x_ij_kp[j,i,k,p] for j in N for k in K_ij_map.get((j,i), [])
                )               
                model.addConstr(outbound - inbound == s_ip.get((i,p),0) - d_ip.get((i,p),0), f"V5_Flow_{i}_{p}")
        
        #(6) crowdshipper k can transfer parcel p from i to j only if it has been assigned to them 
        model.addConstrs((x_ij_kp[i,j,k,p] <= y_kp[k,p] for (i,j,k) in arc_time_k for p in P), name="V6_xy")
        #(7) each crowdshipper can deliver at most one parcel
        model.addConstrs((gp.quicksum(y_kp[k,p] for p in P) <= z_k[k] for k in K), name="V7_1cs-1pc")
        #(8) these constraints prohibit a parcel from leaving a station more than once. 
        # Similarly constraints (9) excludes the parcel from entering a station more than once at time t. 
        l_map = defaultdict(list) 
        a_map = defaultdict(list) 
        #precompute maps to accelerate the search
        for (i, j, k), (t_i, t_j) in arc_time_k.items():
            if t_i in T:
                l_map[(i, t_i)].append((i, j, k)) #map (leaving, time) -> arc
            if t_j in T:
                a_map[(j, t_j)].append((i, j, k)) #map (arriving, time) -> arc
        for i in N:
            for p in P:
                for t in T:
                    # parcel p leaving station i at time t
                    terms_leave = [x_ij_kp[i_arc, j_arc, k_arc, p] 
                                for (i_arc, j_arc, k_arc) in l_map.get((i,t), [])
                                if (i_arc, j_arc, k_arc, p) in x_ij_kp]
                    model.addConstr(l_ipt[i, p, t] == gp.quicksum(terms_leave), name=f"V8_l_{i}_{p}_{t}")
                    # parcel p arriving at station i at time t
                    terms_arrive = [x_ij_kp[i_arc, j_arc, k_arc, p] 
                                for (i_arc, j_arc, k_arc) in a_map.get((i,t), [])
                                if (i_arc, j_arc, k_arc, p) in x_ij_kp]
                    model.addConstr(a_ipt[i, p, t] == gp.quicksum(terms_arrive), name=f"V9_a_{i}_{p}_{t}")
                    
        #(10) parcel p is located at station i at time t+1, if the parcel arrived at this station at time t
        #(11) forces the location of parcel p at station i at time t+1 if the parcel has not been moved from station i at time t and it was already located there 
        #(12) force variable o_ipt to 0 if parcel p leaves station i at time t
        for i in N:
            for p in P:
                for t in range(len(T)-1):
                    model.addConstr(o_ipt[i, p, t+1] >= a_ipt[i, p, t], f"V10_{i}_{p}_{t}")
                    model.addConstr(o_ipt[i, p, t+1] >= o_ipt[i, p, t] - l_ipt[i, p, t], f"V11_{i}_{p}_{t}")
                    model.addConstr(o_ipt[i, p, t+1] <= 1 - l_ipt[i, p, t], f"V12_{i}_{p}_{t}")
        #(13) each parcel can be located in at most one station for each time slot
        model.addConstrs((gp.quicksum(o_ipt[i,p,t] for i in N) <= 1 - q_p[p] for p in P for t in T), name="V13")
        #(14) the number of parcels located at station i plus those arriving at the same station at time t has to be satisfy the station capacity
        model.addConstrs((gp.quicksum(o_ipt[i,p,t] for p in P) + gp.quicksum(a_ipt[i,p,t] for p in P) <= Cap[i] for i in N for t in T if Cap[i] != float('inf')), name="V14_capacity")
        #(15) sets the parcels at time 1 at their source stations
        model.addConstrs((o_ipt[i,p,0] == s_ip[i,p] for i in N for p in P), name="V15_init")

        # --- 5. CALLBACK SETUP ---
        model._start_time = time.time()
        model._ttb = None
        model._lp_relaxation = None
        model._best_obj = None
        model._root_bound_after_cuts = None
        model._solution_history = []
        model._cuts_added = 0
        model._cuts_added_per_node = []
        
        # Dizionario per tracciare QUALE tipo di taglio viene aggiunto
        model._cuts_stats = defaultdict(int)

        ########### CALLBACK: valid inequalities (inner function)
        def valid_inequalities_callback(model, where):
            TOL = 1e-6
            try:
                # --- TIME TO BEST ---
                if where == GRB.Callback.MIPSOL:
                    new_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    if model._best_obj is None or new_obj < model._best_obj - 1e-6:
                        current_time = time.time() - model._start_time
                
                        model._best_obj = new_obj
                        model._ttb = current_time
                
                #Save history for the graph
                        if hasattr(model, '_solution_history'):
                            model._solution_history.append((current_time, new_obj))
                # --- VALID INEQUALITIES ---
                elif where == GRB.Callback.MIPNODE:
                    node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)

                    # if node_count != 0 and node_count % N_nodes_interval != 0:
                    #     return  # only root or each N nodes
                    
                    status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
                    if status != GRB.OPTIMAL:
                        return
                    
                    y_val = {k: model.cbGetNodeRel(v) for k, v in model._y_kp.items()}
                    e_val = {i: model.cbGetNodeRel(v) for i, v in model._e_i.items()}
                    s_val = {k: model.cbGetNodeRel(v) for k, v in model._s_ip.items()}
                    d_val = {k: model.cbGetNodeRel(v) for k, v in model._d_ip.items()}
                    l_val = {k: model.cbGetNodeRel(v) for k, v in model._l_ipt.items()}
                    a_val = {k: model.cbGetNodeRel(v) for k, v in model._a_ipt.items()}
                    o_val = {k: model.cbGetNodeRel(v) for k, v in model._o_ipt.items()}
                    q_val = {p: model.cbGetNodeRel(v) for p, v in model._q_p.items()}
                    
                    cuts_in_pass = 0

                    #add cuts (VI1-VI9)
                    # VI1(24) if the source station i is selected at least one parcel must start its path at it
                    for i in model._S:
                        lhs = sum(s_val.get((i, p), 0.0) for p in model._P)
                        if lhs < e_val.get(i, 0.0) - TOL:
                            model.cbCut(gp.quicksum(model._s_ip[i, p] for p in model._P) >= model._e_i[i])
                            model._cuts_added += 1
                            model._cuts_stats["VI1"] += 1
                            cuts_in_pass += 1
                    for p in model._P:
                        # VI2 (25) destination station can't be selected for parcel p if the parcel has not been assigned to a crowdshipper
                        lhs_d = sum(d_val.get((i, p), 0.0) for i in model._D if (i, p) in model._d_ip)
                        rhs_y = sum(y_val.get((k, p), 0.0) for k in model._K)
                        if lhs_d > rhs_y + TOL:
                            model.cbCut(gp.quicksum(model._d_ip[i, p] for i in model._D if (i, p) in model._d_ip) <= gp.quicksum(model._y_kp[k, p] for k in model._K))
                            model._cuts_added += 1
                            model._cuts_stats["VI2"] += 1
                            cuts_in_pass += 1
                        # VI3 (26) a source station can't be selected for parcel p if the parcel has not been assigned to a crowdshipper
                        lhs_s = sum(s_val.get((i, p), 0.0) for i in model._S if (i, p) in model._s_ip)
                        if lhs_s > rhs_y + TOL:
                            model.cbCut(gp.quicksum(model._s_ip[i, p] for i in model._S if (i, p) in model._s_ip) <= gp.quicksum(model._y_kp[k, p] for k in model._K))
                            model._cuts_added += 1
                            model._cuts_stats["VI3"] += 1
                            cuts_in_pass += 1
                        # VI4 (27) parcel p can leave a station only if a crowdshipper handles this parcel
                        lhs_l = sum(l_val.get((i, p, t), 0.0) for i in model._N for t in model._T if (i, p, t) in model._l_ipt)
                        if lhs_l > rhs_y + TOL:
                            model.cbCut(gp.quicksum(model._l_ipt[i, p, t] for i in model._N for t in model._T if (i, p, t) in model._l_ipt) <= gp.quicksum(model._y_kp[k, p] for k in model._K))
                            model._cuts_added += 1
                            model._cuts_stats["VI4"] += 1
                            cuts_in_pass += 1
                        # VI5 (28) parcel p can arrive at a station only if a crowdshipper handles this parcel
                        lhs_a = sum(a_val.get((i, p, t), 0.0) for i in model._N for t in model._T if (i, p, t) in model._a_ipt)
                        if lhs_a > rhs_y + TOL:
                            model.cbCut(gp.quicksum(model._a_ipt[i, p, t] for i in model._N for t in model._T if (i, p, t) in model._a_ipt) <= gp.quicksum(model._y_kp[k, p] for k in model._K))
                            model._cuts_added += 1
                            model._cuts_stats["VI5"] += 1
                            cuts_in_pass += 1
                        # VI6 (29) a parcel cannot be assigned to a crowdshipper if it has already been allocated to the backup service 
                        for k in model._K:
                            if y_val.get((k, p), 0.0) > 1 - q_val.get(p, 0.0) + TOL:
                                model.cbCut(model._y_kp[k, p] <= 1 - model._q_p[p])
                                model._cuts_added += 1
                                model._cuts_stats["VI6"] += 1
                                cuts_in_pass += 1

                        # VI7 (30) for a parcel p source and destination variables cannot be both equal to 1
                        for i in model._N:
                            if s_val.get((i, p), 0.0) + d_val.get((i, p), 0.0) > 1 + TOL:
                                if (i, p) in model._s_ip and (i, p) in model._d_ip:
                                    model.cbCut(model._s_ip[i, p] + model._d_ip[i, p] <= 1)
                                    model._cuts_added += 1
                                    model._cuts_stats["VI7"] += 1
                                    cuts_in_pass += 1
                        
                        # VI8 and VI9 (31)-(32) the location variables for parcel p at time slot t can only be active at a station if both a starting and a destination stations have been assigned to this parcel
                        for t in model._T:
                            lhs_o = sum(o_val.get((i, p, t), 0.0) for i in model._N if (i, p, t) in model._o_ipt)
                            rhs_s = sum(s_val.get((i, p), 0.0) for i in model._S if (i, p) in model._s_ip)
                            if lhs_o > rhs_s + TOL:
                                model.cbCut(gp.quicksum(model._o_ipt[i, p, t] for i in model._N if (i, p, t) in model._o_ipt) <= gp.quicksum(model._s_ip[i, p] for i in model._S if (i, p) in model._s_ip))
                                model._cuts_added += 1
                                model._cuts_stats["VI8"] += 1
                                cuts_in_pass += 1
                            rhs_d = sum(d_val.get((i, p), 0.0) for i in model._D if (i, p) in model._d_ip)
                            if lhs_o > rhs_d + TOL:
                                model.cbCut(gp.quicksum(model._o_ipt[i, p, t] for i in model._N if (i, p, t) in model._o_ipt) <= gp.quicksum(model._d_ip[i, p] for i in model._D if (i, p) in model._d_ip))
                                model._cuts_added += 1
                                model._cuts_stats["VI9"] += 1
                                cuts_in_pass += 1
                                            
                    model._cuts_added_per_node.append(cuts_in_pass)

                    # Root Bound Tracking
                    if node_count == 0:
                        model._root_bound_after_cuts = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            except Exception:
                return

        
        model.Params.LazyConstraints = 0
        model.setParam(GRB.Param.Seed, 123)
        model.setParam(GRB.Param.Presolve, 1)
        model.setParam(GRB.Param.Method, -1)
        model.setParam(GRB.Param.Heuristics, 0.05)
        model.setParam(GRB.Param.Cuts, -1)
        model.setParam(GRB.Param.Threads, 2)
        model.setParam(GRB.Param.OutputFlag, 1)
        
        model.setParam(GRB.Param.LogFile, "gurobi_callback.log")
        model.update()

        model._num_vars_pre = model.NumVars
        model._num_constrs_pre = model.NumConstrs
        vars_dict = {
            'z_k': z_k,
            'q_p': q_p,
            's_ip': s_ip,
            'x_ij_kp': x_ij_kp
            }
        #OPTIMIZE
        model.optimize(valid_inequalities_callback)


        has_solution = model.SolCount > 0
        presolve_time = None
        logfile = "gurobi_callback.log"
        
        try:
            with open(logfile, "r") as f:
                for line in f:
                    if "Presolve time:" in line:
                        presolve_time = float(line.split(":")[1].replace("s", "").strip())
        except:
            presolve_time = None


        rb_cuts = getattr(model, "_root_bound_after_cuts", None)

        root_gap = None
        if has_solution:
            total_crowdshipper_reward = sum(z_k[k].x for k in K) * rho
            total_fixed_cost = sum(e_i[i].x * vi[i] for i in S)
            total_loading_cost = sum(s_ip[i, p].x * sigma[i] for i in S for p in P)
            total_backup_cost = sum(q_p[p].x * gamma[p] for p in P)
            if model.SolCount > 0 and rb_cuts is not None and abs(model.ObjVal) > 1e-9:
                root_gap = abs(model.ObjVal - rb_cuts) /abs(model.ObjVal)  

            # print(f"  - Total cost crowdshippers reward: {total_crowdshipper_reward:.2f}")
            # print(f"  - Total fixed delivery cost: {total_fixed_cost:.2f}")
            # print(f"  - Total loading cost: {total_loading_cost:.2f}")
            # print(f"  - Total backup cost: {total_backup_cost:.2f}")

            num_delivered_parcels = sum(1 for p in P if q_p[p].X < 0.5)
            num_backup_parcels = sum(1 for p in P if q_p[p].X > 0.5)
            num_crowdshippers_used = sum(1 for k in K if z_k[k].X > 0.5)
            percent_cs_delivery = (num_delivered_parcels / len(P)) * 100 if len(P) > 0 else 0.0

        else:
            total_crowdshipper_reward = None
            total_fixed_cost = None
            total_loading_cost = None
            total_backup_cost = None
            num_delivered_parcels = None
            num_crowdshippers_used = None
            percent_cs_delivery = None

        #RESULTS
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.INTERRUPTED:
            print("\n Optimal solution found")
            print(f"Objective value: {model.ObjVal:.2f}")
            
            # Gestione del Best Bound e Gap
            obj_val = getattr(model, "ObjVal", None)
            obj_bound = getattr(model, "ObjBound", None)
            gap = getattr(model, "MIPGap", None)
   
            # Risultati
            results = {
                "status": model.status,
                "objective": model.ObjVal,
                "best_bound": model.ObjBound if has_solution else None,
                "resolved_to_optimality": model.status == GRB.OPTIMAL,
                "nodes_explored": int(model.NodeCount),
                "integrality_gap_root": root_gap,
                
                "mip_gap": gap if model.SolCount > 0 else None, 
                "num_vars": model._num_vars_pre,
                "num_constrs": model._num_constrs_pre,
                "num_delivered_parcels": num_delivered_parcels,
                "num_crowdshippers_used": num_crowdshippers_used,
                "percent_cs_delivery": percent_cs_delivery,
                "total_backup_cost": total_backup_cost,
                "total_loading_cost": total_loading_cost,
                "total_fixed_cost": total_fixed_cost,
                "total_crowdshipper_reward": total_crowdshipper_reward,
               
                "presolve_time": presolve_time,
                "execution_time_s": model.Runtime,
                "nodes_explored": model.NodeCount,
                "simplex_iterations": model.IterCount,
                "time_to_best_s": model._ttb,
                "cuts_added": getattr(model, "_cuts_added", 0),  
                "root_bound_after_cuts": getattr(model, "_root_bound_after_cuts", None),
                "cuts_added_per_node": getattr(model, "_cuts_added_per_node", []),
                "cuts_stats": dict(model._cuts_stats),
            }

            model.write("debug.lp")
                         
            pretty_print_milp(model, results, instance_data, vars_dict)
            return results
            
        elif model.status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT]:
            print("\n Time/Node limit reached")
            results = {
                "objective": model.ObjVal if model.SolCount > 0 else None,
                "best_bound": model.ObjBound,
                "mip_gap": model.MIPGap if model.SolCount > 0 else None,
                "execution_time_s": model.Runtime,
                "nodes_explored": model.NodeCount,
                "integrality_gap_root": root_gap,
                "resolved_to_optimality": False,
                "status": model.status,
                "num_vars":model._num_vars_pre,
                "num_constrs": model._num_constrs_pre, 
                "num_delivered_parcels": num_delivered_parcels,
                "num_crowdshippers_used": num_crowdshippers_used,
                "percent_cs_delivery": percent_cs_delivery,
                "total_backup_cost": total_backup_cost,
                "total_loading_cost": total_loading_cost,
                "total_fixed_cost": total_fixed_cost,
                "total_crowdshipper_reward": total_crowdshipper_reward,
                "simplex_iterations": model.IterCount,
                "time_to_best_s": model._ttb,
                "cuts_added": getattr(model, "_cuts_added", 0),  
                "root_bound_after_cuts": getattr(model, "_root_bound_after_cuts", None),
                "presolve_time": presolve_time,
                "cuts_added_per_node": getattr(model, "_cuts_added_per_node", []),
                "cuts_stats": dict(model._cuts_stats),

            }
                         
            pretty_print_milp(model, results, instance_data, vars_dict)
            return results

    except gp.GurobiError as e:
        if "Out of memory" in str(e) or e.errno == 10001:
            print(f"\n OOM on instance! Extract partial data...")

            try:
                gap_val = model.MIPGap
            except:
                gap_val = None
                
            try:
                obj_val = model.ObjVal
            except:
                obj_val = None
                
            try:
                bound_val = model.ObjBound
            except:
                bound_val = None
            
            results = {
                "status": "OutOfMemory",
                "objective": obj_val,
                "best_bound": bound_val,
                "integrality_gap_root": None,
                "nodes_explored": getattr(model, "NodeCount", None),
                "root_bound_after_cuts": getattr(model, "_root_bound_after_cuts", None),
                "mip_gap": gap_val,
                "execution_time_s": getattr(model, "Runtime", timelimit),
                "simplex_iterations": getattr(model, "IterCount", 0),
                
                "memory_used_MB": getattr(model, "MemoryUsage", 0),
                
                "num_vars": getattr(model, "_num_vars_pre", None),
                "num_constrs": getattr(model, "_num_constrs_pre", None),

                "num_delivered_parcels": None,
                "num_crowdshippers_used": None,
                "percent_cs_delivery": None,
                "total_backup_cost": None,
                "total_loading_cost": None,
                "total_fixed_cost": None,
                "total_crowdshipper_reward": None,

                "presolve_time": None,
                "time_to_best_s": None,

                "cuts_added": getattr(model, "_cuts_added", 0),
                "cuts_added_per_node": getattr(model, "_cuts_added_per_node", []),
                "cuts_stats": dict(model._cuts_stats),

            }
                         
            
            return results
        else:
            print(f"Gurobi Error: {e}")
            return {"status": "Error", "message": str(e)}
    except Exception as e:
        print(f"*** GENERAL ERROR: {e} ***")
        print(traceback.format_exc())
        return {"status": "Error", "message": str(e), "traceback": traceback.format_exc()}




# instance_filepath = os.path.join("instances", "instance_test_notebook")
# # Caricamento JSON
# with open(instance_filepath, "r") as f:
#     instance_data = json.load(f)  # json.load converte il JSON in dict/list

# results = \
#     solve_milp_crowdshipping_callback(instance_data, timelimit=1000)
