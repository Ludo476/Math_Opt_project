import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict
import ast
import json
import os, traceback
import time

from .helpers_milp import pretty_print_milp, ptcp_callback


def solve_milp_crowdshipping_simple(instance_data, timelimit=None, nodelimit=None):
    """
    MILP without integrated cuts
    """
    try:
        model = gp.Model("PTCP_Crowdshipping_Final_simple")
        if timelimit: model.Params.TimeLimit = timelimit

        # 1. DATA AND SETS  
        P = [p['id'] for p in instance_data['demand']['parcels']]
        N = instance_data["network"]["global_graph"]["N"]
        S = instance_data["network"]["global_graph"]["S"]
        D = instance_data["network"]["global_graph"]["D"]
        S = list(map(int, S))
        D = list(map(int, D))
        N = list(map(int, N))
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

        model.update()  # fondamentale
        model._num_vars_pre = model.NumVars
        model._num_constrs_pre = model.NumConstrs


        #OPTIMIZE        
              
        model.setParam(GRB.Param.Seed, 123)
        model.setParam(GRB.Param.Presolve, 1)
        model.setParam(GRB.Param.Method, -1)
        model.setParam(GRB.Param.Heuristics, 0.05)
        model.Params.MemLimit = 12
        model.setParam(GRB.Param.Cuts, -1)
        model.setParam(GRB.Param.Threads, 2)
        model.setParam(GRB.Param.OutputFlag, 1)
        # model.setParam(GRB.Param.LogToConsole, 1)
        # model.Params.NodeFileStart = 0.5
        # model.Params.NodefileDir = "."

        #========================DEBUG==========================#      
        # print("== SUMMARY PRE-OPT ==")
        # print("len(P), len(N), len(S), len(D), len(T):", len(P), len(N), len(S), len(D), len(T))
        # print("len(K):", len(K))
        # print("len(arc_time_k):", len(arc_time_k))
        # print("len(K_ij_map):", len(K_ij_map))
        # print("len(valid_keys) for x:", len(valid_keys))
        # print("example valid_keys (first 20):", list(valid_keys)[:20])

        # vk = set(valid_keys)
        # with open("valid_keys_versionX.json","w") as f:
        #     json.dump(sorted(list(vk)), f)
        # print("min/max valid_keys:", min(vk), max(vk))

        # print("Sample arc_time_k items (first 20):", list(arc_time_k.items())[:20])
        # print("Sample l_map entries keys:", list(l_map.keys())[:10])
        # print("Sample a_map entries keys:", list(a_map.keys())[:10])
        # # controllo tipi:
        # for key in list(arc_time_k.keys())[:20]:
        #     print(type(key), key)
        # for val in list(arc_time_k.values())[:10]:
        #     print(type(val), val)

        # print("Num Vars:", model.NumIntVars)
        # print("Num Constrs:", model.NumConstrs)
        # # opzionale: conta vincoli con nomi particolari
        # print("Constraints names sample (first 30):", [c.ConstrName for c in model.getConstrs()[:30]])

        model._start_time = time.time()
        model._ttb = None
        model._best_obj = None
        model._lp_relaxation = None
        model._solution_history = []
        model._root_bound_after_cuts = None
        logfile = "gurobi_log_tmp.log"
        model.setParam(GRB.Param.LogFile, logfile)
        model._cuts_added = 0
        model._solution_history = []
        
        model.optimize(ptcp_callback)
        vars_dict = {
            'z_k': z_k,
            'q_p': q_p,
            's_ip': s_ip,
            'x_ij_kp': x_ij_kp
            }
        #History
        if not hasattr(model, '_solution_history'):
            model._solution_history = []
        if model.SolCount > 0 and not model._solution_history:
            model._solution_history.append((model.Runtime, model.ObjVal))

        #Presolve time
        presolve_time = None
        try:
            with open(logfile, "r") as f:
                for line in f:
                    if "Presolve time:" in line:
                        presolve_time = float(line.split(":")[1].replace("s", "").strip())
        except:
            presolve_time = None

        
        has_solution = model.SolCount > 0

        
        nodes = model.NodeCount
        runtime = model.Runtime
        root_gap = None


        if has_solution:
            total_crowdshipper_reward = sum(z_k[k].x for k in K) * rho
            total_fixed_cost = sum(e_i[i].x * vi[i] for i in S)
            total_loading_cost = sum(s_ip[i, p].x * sigma[i] for i in S for p in P)
            total_backup_cost = sum(q_p[p].x * gamma[p] for p in P)

            num_delivered_parcels = sum(1 for p in P if q_p[p].X < 0.5)
            num_backup_parcels = sum(1 for p in P if q_p[p].X > 0.5)
            num_crowdshippers_used = sum(1 for k in K if z_k[k].X > 0.5)
            percent_cs_delivery = (num_delivered_parcels / len(P)) * 100 if len(P) > 0 else 0.0


            rb_cuts = getattr(model, "_root_bound_after_cuts", None)
            if rb_cuts is not None and abs(model.ObjVal) > 1e-9:
                root_gap = abs(model.ObjVal - rb_cuts) /abs(model.ObjVal)  

            if model._lp_relaxation is not None and abs(model.ObjVal) > 1e-9:
                integrality_gap_root = (model.ObjVal - model._lp_relaxation) / model.ObjVal * 100
            else:
                integrality_gap_root = None
            
        else:
            total_crowdshipper_reward = None
            total_fixed_cost = None
            total_loading_cost = None
            total_backup_cost = None
            num_delivered_parcels = None
            num_crowdshippers_used = None
            percent_cs_delivery = None

        mip_gap = model.MIPGap if has_solution else None

        if model.status == GRB.OPTIMAL:
            print("\n Optimal solution found")
            print(f"Objective value: {model.ObjVal:.2f}")

            gap = getattr(model, "MIPGap", None)
   
            # results
            results = {
                "status": model.status,
                "resolved_to_optimality":model.status == GRB.OPTIMAL,
                "objective": model.ObjVal if has_solution else None,
                "best_bound": getattr(model, "ObjBound", None),
                "mip_gap": mip_gap,
                # Performance
                "execution_time_s": model.Runtime,
                "nodes_explored": int(model.NodeCount),
                "simplex_iterations": model.IterCount,
                "time_to_best_s": model._ttb,
                "presolve_time": presolve_time,
                "history": model._solution_history,
                # data of the model
                "num_vars": getattr(model, "_num_vars_pre", model.NumVars),
                "num_constrs": getattr(model, "_num_constrs_pre", model.NumConstrs),
                "root_bound_after_cuts": getattr(model, "_root_bound_after_cuts", None),
                "integrality_gap_root": root_gap,

                "num_delivered_parcels": num_delivered_parcels,
                "num_crowdshippers_used": num_crowdshippers_used,
                "percent_cs_delivery": percent_cs_delivery,
                "total_backup_cost": total_backup_cost,
                "total_loading_cost": total_loading_cost,
                "total_fixed_cost": total_fixed_cost,
                "total_crowdshipper_reward": total_crowdshipper_reward,
            }

            # model.write("debug.lp")
            pretty_print_milp(model, results, instance_data, vars_dict)
            return results
    
            
        elif model.status in [GRB.TIME_LIMIT, GRB.NODE_LIMIT]:
            print("\n Time/Node limit reached")
            results = {
                "objective": model.ObjVal if model.SolCount > 0 else None,
                "best_bound": model.ObjBound if hasattr(model, "ObjBound") else None,
                "root_bound_after_cuts": model._root_bound_after_cuts if model._root_bound_after_cuts is not None else None,
                "mip_gap": model.MIPGap if model.SolCount > 0 else None,
                "execution_time_s": model.Runtime,
                "nodes_explored": model.NodeCount,
                "resolved_to_optimality": False,
                "simplex_iterations": model.IterCount,
                "status": model.status,
                "integrality_gap_root": root_gap,

                "time_to_best_s": model._ttb,
                "presolve_time": presolve_time,
                "history": model._solution_history,

                "num_vars": model._num_vars_pre,
                "num_constrs": model._num_constrs_pre, 
                "num_delivered_parcels": num_delivered_parcels,
                "num_crowdshippers_used": num_crowdshippers_used,
                "percent_cs_delivery": percent_cs_delivery,
                "total_backup_cost": total_backup_cost,
                "total_loading_cost": total_loading_cost,
                "total_fixed_cost": total_fixed_cost,
                "total_crowdshipper_reward": total_crowdshipper_reward,
                
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
                "objective": obj_val,
                "best_bound": bound_val,
                "mip_gap": gap_val,
                "integrality_gap_root": None,
                "root_bound_after_cuts": getattr(model, "_root_bound_after_cuts", None),
                
                "execution_time_s": getattr(model, "Runtime", timelimit),
                "nodes_explored": getattr(model, "NodeCount", None),
                "simplex_iterations": getattr(model, "IterCount", 0),
                "presolve_time": None,  
                "time_to_best_s": None,

                "status": "OutOfMemory",
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
            
            }
            
            return results
        else:
            print(f"Gurobi Error: {e}")
            return {"status": "Error", "message": str(e)}
    except Exception as e:
        print(f"*** GENERAL ERROR: {e} ***")
        print(traceback.format_exc())
        return {"status": "Error", "message": str(e), "traceback": traceback.format_exc()}
    


# instance_filepath = os.path.join("instances_scalability/class_5", "class_5_fam_0_inst_0.json")
# instance_filepath = os.path.join("instances", "test_giocattolo.json")

# with open(instance_filepath, "r") as f:
#     instance_data = json.load(f) 

# results = \
#     solve_milp_crowdshipping_simple(instance_data, timelimit=1000)
# if results:
#     print("Model correctly created and optimized!")
#     print("results:{results}")
# else:
#     print("something went wrong.")
