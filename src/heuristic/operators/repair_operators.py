import random
import math
import copy
import time
import numpy as np 
import heapq
import networkx as nx

from ..helpers import Helpers

class RepairOperators:
    """
    Implements the Repair stage of the ALNS algorithm for the PTCP.
    """
    def __init__(self, instance_data, helpers, parcels_map, st_adj, solver):
        self.instance_data = instance_data
        self.helpers = helpers
        self.solver = solver
        self.st_adj = st_adj
        self.parcels_map = parcels_map
        self.station_capacities = {s['id']: s['capacity'] for s in instance_data['network']['stations']}
        self.use_global_best=True
        self.G_global = nx.DiGraph()
        for u, neighbors in self.helpers.global_graph_adj.items():
            for v in neighbors:
                self.G_global.add_edge(u, v)
        
        # compute distances here
        source_stations = [s['id'] for s in instance_data['network']['stations'] if s.get('is_source')]
        self.cached_source_distances = {}
        for s_id in source_stations:
            try:
                self.cached_source_distances[s_id] = nx.shortest_path_length(self.G_global, source=s_id)
            except nx.NetworkXNoPath:
                self.cached_source_distances[s_id] = {}
    # ================================================================
    # REPAIR R1 → Parcel Distribution + Shortest Path
    # ================================================================
    def repair_R1(self, solution,**kwargs):
        if self.use_global_best:
            noise = kwargs.get('noise_level', 0.1)
            return self._run_repair(solution, self._assign_stations_R1, strategy="min_crowdshippers",noise_level=noise)
        else:
            return self._run_repair_global_best(solution, self._assign_stations_R1, strategy="min_crowdshippers", **kwargs)

    # ================================================================
    # REPAIR R2 → Delivery Cost Minimization + Shortest Path
    # ================================================================
    def repair_R2(self, solution, **kwargs):
        if self.use_global_best:
            noise = kwargs.get('noise_level', 0.1)
            return self._run_repair(solution, self._assign_station_R2, strategy="min_crowdshippers",noise_level=noise)
        else:
            return self._run_repair_global_best(solution, self._assign_station_R2, strategy="min_crowdshippers")
    
    # ================================================================
    # REPAIR R3 → Parcel Distribution + Load Balancing
    # ================================================================
    def repair_R3(self, solution, **kwargs):
        if self.use_global_best:
            noise = kwargs.get('noise_level', 0.1)
            return self._run_repair(solution, self._assign_stations_R1, strategy="load_balancing",noise_level=noise)
        else:
            return self._run_repair_global_best(solution, self._assign_stations_R1, strategy="load_balancing")
    # ================================================================
    # REPAIR R4 → Delivery Cost Minimization + Load Balancing
    # ================================================================
    def repair_R4(self, solution, **kwargs):
        if self.use_global_best:
            noise = kwargs.get('noise_level', 0.1)
            return self._run_repair(solution, self._assign_station_R2, strategy="load_balancing",noise_level=noise)
        else:
            return self._run_repair_global_best(solution, self._assign_stations_R1, strategy="load_balancing")
    

    # ================================================================
    # Definitive function to handle with repairs
    # ================================================================ 

    def _run_repair(self, solution, station_assigner, strategy, noise_level=0.15):
        """
        Implement a little different version from paper but finds better solutions 
        """
        new_solution = copy.deepcopy(solution)
        parcels_to_repair = list(new_solution['backup_parcels'])
        
        # Setup
        used_crowdshippers_in_repair = {
            cs_id for cs_id, a in new_solution.get('crowdshipper_assignments', {}).items() if a
        }
        station_load = self.helpers._get_num_parcels_by_station(new_solution)
        current_load_matrix, _ = self.helpers._build_load_matrix_from_solution(new_solution)
        
        # Caching
        stats = self.instance_data['network']['stations']
        fixed_costs = {s['id']: self.instance_data['costs']['fixed_source_delivery_cost'].get(str(s['id']), 0.0) 
                    for s in stats}
        loading_costs = {s['id']: s['loading_cost'] for s in stats}
        capacities = self.station_capacities
        crowd_reward = float(self.instance_data.get('costs', {}).get('crowdshipper_reward', 1.0))
        active_sources = {s_id for s_id, load in station_load.items() if load > 0}

        repaired = set()
        
        while parcels_to_repair:
            regret_candidates = []

            for parcel_id in parcels_to_repair:
                parcel_data = self.parcels_map[parcel_id]
                backup_cost = float(parcel_data.get('backup_cost', 0.0))
                
                # Candidate stations
                candidates_map = station_assigner(new_solution, [parcel_id])
                suggested = candidates_map.get(parcel_id, {}).get('origin_station')
                all_sources = [s['id'] for s in stats if s.get('is_source')]
                candidate_stations = ([suggested] + [s for s in all_sources if s != suggested] 
                                    if suggested in all_sources else all_sources)
                
                valid_moves = []

                for origin in candidate_stations:
                    # Capacity check
                    if station_load.get(origin, 0) >= capacities.get(origin, float('inf')): 
                        continue
                    
                    # Path finding
                    path = self.helpers._label_setting_algorithm(
                        parcel_data, origin, parcel_data['destination'],
                        new_solution, strategy, used_crowdshippers_in_repair,
                        load_matrix=current_load_matrix
                    )
                    
                    if not path: continue

                    # Cost calculation
                    f_cost = 0.0 if origin in active_sources else fixed_costs.get(origin, 0.0)
                    l_cost = loading_costs.get(origin, 0.0)
                    n_cs = sum(1 for arc in path if len(arc) >= 3 and arc[2] is not None)
                    base_cost = f_cost + l_cost + (n_cs * crowd_reward)
                    
                    # FIX: Apply noise first, then check
                    if noise_level > 0:
                        random_factor = 1.0 + random.uniform(-noise_level, noise_level)
                        cost = base_cost * random_factor
                    else:
                        cost = base_cost
                    
                    # Valid only if beats backup (with noise already applied)
                    if cost < backup_cost * 1.05:  # small tolerance (5%) for noise
                        valid_moves.append((cost, path, origin))

                # Regret calculation
                if not valid_moves:
                    continue
                
                valid_moves.sort(key=lambda x: x[0])
                best_move = valid_moves[0]
                
                if len(valid_moves) >= 2:
                    second_best = valid_moves[1]
                    regret = second_best[0] - best_move[0]
                else:
                    regret = backup_cost - best_move[0]
                
                regret_candidates.append((regret, parcel_id, best_move))

            # Selection
            if not regret_candidates:
                break
            
            regret_candidates.sort(key=lambda x: x[0], reverse=True)
            
            chosen_regret, chosen_pid, move_info = regret_candidates[0]
            _, chosen_path, chosen_origin = move_info

            # Commit
            self.helpers._update_solution_with_new_path(new_solution, chosen_pid, chosen_path, chosen_origin)
            self.helpers._update_matrix_inplace(current_load_matrix, chosen_path, chosen_origin, chosen_pid)
            
            repaired.add(chosen_pid)
            parcels_to_repair.remove(chosen_pid)
            
            station_load[chosen_origin] = station_load.get(chosen_origin, 0) + 1
            active_sources.add(chosen_origin)
            for arc in chosen_path:
                if len(arc) >= 3 and arc[2] is not None:
                    used_crowdshippers_in_repair.add(arc[2])

        # Finalize
        new_solution['backup_parcels'] = set(new_solution['backup_parcels']) - repaired
        new_solution['load_matrix'] = current_load_matrix
        new_solution['solution_cost'] = self.solver._calculate_cost(new_solution)
        
        return new_solution
    
    #========================== I didn't use this version, very slow, so not good efficiency-effectiveness trade-off

    def _run_repair_global_best(self, solution, station_assigner, strategy, order_strategy="Random"):
        """
        GLOBAL BEST INSERTION REPAIR
        Iteratively selects the (Parcel, Station) pair that yields the highest global saving.
        Slower than sequential (O(N^2)), but produces significantly higher quality solutions.
        """
        # [DEBUG] Start
        print(f"\n[DEBUG][REPAIR] Start {strategy}. Backup size: {len(solution['backup_parcels'])}")
        t_start = time.time()

        new_solution = copy.deepcopy(solution)
        parcels_to_repair = list(new_solution['backup_parcels'])
        
        # 1. Setup 
        used_crowdshippers_in_repair = {
            cs_id for cs_id, assignments in new_solution.get('crowdshipper_assignments', {}).items() 
            if len(assignments) > 0
        }
        
        station_load = self.helpers._get_num_parcels_by_station(new_solution)
        active_sources_in_repair = {s_id for s_id, load in station_load.items() if load > 0}
        
        
        current_load_matrix, _ = self.helpers._build_load_matrix_from_solution(new_solution)

        # Caching Costs
        fixed_costs = {s['id']: self.instance_data['costs']['fixed_source_delivery_cost'].get(str(s['id']), 0.0) for s in self.instance_data['network']['stations']}
        loading_costs = {s['id']: s['loading_cost'] for s in self.instance_data['network']['stations']}
        crowd_reward = float(self.instance_data.get('costs', {}).get('crowdshipper_reward', 1.0))
        station_capacities = self.station_capacities

        repaired = set()
        remaining = set(parcels_to_repair)

        
        def compute_assignment_cost_for_path(path, origin_station):
            if not path: return float('inf')
            # Setup cost added only if new source
            f_cost = 0.0 if origin_station in active_sources_in_repair else fixed_costs.get(origin_station, 0.0)
            l_cost = loading_costs.get(origin_station, 0.0) 
            n_cs = sum(1 for arc in path if len(arc) >= 3 and arc[2] is not None)
            return f_cost + l_cost + (n_cs * crowd_reward)

        # -----------------------------------------------------------
        # MAIN LOOP (Global Best)
        # -----------------------------------------------------------
        iter_id = 0
        while remaining:
            iter_id += 1
            candidates = []  # (delta, parcel_id, origin_station, path, assign_cost)
            
            # Order the parcell to process first the more expensive
            ordered_parcels = self.helpers._get_parcels_in_order(remaining, order_strategy)

            # Analyze remaining parcels
            for parcel_id in ordered_parcels:
                parcel_data = self.parcels_map[parcel_id]
                backup_cost = float(parcel_data.get('backup_cost', 0.0))
                
                # Obtain candidate stations
                candidates_map = station_assigner(new_solution, [parcel_id])
                suggested_origin = candidates_map.get(parcel_id, {}).get('origin_station')
                
                all_sources = [s['id'] for s in self.instance_data['network']['stations'] if s.get('is_source')]
                if suggested_origin in all_sources:
                    all_sources.remove(suggested_origin)
                    candidate_stations = [suggested_origin] + all_sources
                else:
                    candidate_stations = all_sources
                
                # Find the best station for this parcel
                for origin_station in candidate_stations:
                    
                    # 1. Capacity Check 
                    if station_load.get(origin_station, 0) >= station_capacities.get(origin_station, float('inf')):
                        continue
                    
                    # 2. Cost Pruning: if activating the station is more expensive than backup, skip
                    if origin_station not in active_sources_in_repair:
                        if fixed_costs.get(origin_station, 0) > backup_cost:
                            continue

                    # 3. Label Setting 
                    path = self.helpers._label_setting_algorithm(
                        parcel_data,
                        origin_station,
                        parcel_data['destination'],
                        new_solution,
                        strategy,
                        used_crowdshippers_in_repair,
                        load_matrix=current_load_matrix, 
                    )
                    
                    if not path: continue

                    assign_cost = compute_assignment_cost_for_path(path, origin_station)
                    delta = assign_cost - backup_cost 

                    if delta < -1e-4: # only if improves
                        candidates.append((delta, parcel_id, origin_station, path, assign_cost))
                        
                        if origin_station in active_sources_in_repair and delta < -0.5 * backup_cost:
                            break 

            # if no valid candidates for any parcels, finish
            if not candidates: 
                # print(f"[DEBUG][REPAIR] No valid candidates found for remaining {len(remaining)} parcels. Stopping.")
                break

            candidates.sort(key=lambda x: x[0]) # Order using delta (more negative is better)
            best_delta, best_pid, best_origin, best_path, _ = candidates[0]

            # [DEBUG] Commit 
            # print(f"   > Commit: P{best_pid} -> St{best_origin} (Delta: {best_delta:.2f})")

            self.helpers._update_solution_with_new_path(new_solution, best_pid, best_path, best_origin)
            self.helpers._update_matrix_inplace(current_load_matrix, best_path, best_origin, best_pid)

            repaired.add(best_pid)
            remaining.remove(best_pid)

            station_load[best_origin] = station_load.get(best_origin, 0) + 1
            active_sources_in_repair.add(best_origin)

            for arc in best_path:
                if len(arc) >= 3 and arc[2] is not None:
                    used_crowdshippers_in_repair.add(arc[2])

        # Finalize
        new_solution['backup_parcels'] = set(parcels_to_repair) - repaired
        new_solution['load_matrix'] = current_load_matrix # Salva la matrice aggiornata
        new_solution['solution_cost'] = self.solver._calculate_cost(new_solution)

        # [DEBUG] Summary
        elapsed = time.time() - t_start
        # print(f"[DEBUG][REPAIR] Done in {elapsed:.2f}s. Repaired: {len(repaired)}/{len(parcels_to_repair)}. New Cost: {new_solution['solution_cost']:.2f}\n")
        
        return new_solution



    # --- Station Assigner R1: marginal cost priority queue ---
    def _assign_stations_R1(self, solution, parcels):
        """
        Suggests stations based on the function p_i(n) as described in the paper.
        Priority is given to stations with lower marginal loading costs.
        """
        station_map = {s['id']: s for s in self.instance_data['network']['stations']}

        station_capacities_map = {s['id']: s['capacity'] for s in self.instance_data['network']['stations']}
        loading_costs_map = {s['id']: s['loading_cost'] for s in self.instance_data['network']['stations']}
        fixed_delivery_costs_map = {s['id']: self.instance_data['costs']['fixed_source_delivery_cost'].get(str(s['id']), 0.0) for s in self.instance_data['network']['stations']}
        
        num_parcels_by_station = self.helpers._get_num_parcels_by_station(solution)
        pq = []        
        for station_id, station in station_map.items():
            if not station.get('is_source', False): continue                
            current_load = num_parcels_by_station.get(station_id, 0)
            capacity = station_capacities_map.get(station_id, 0)
            
            if current_load < capacity:
                if current_load == 0:
                    cost_to_add = fixed_delivery_costs_map[station_id] + loading_costs_map[station_id]
                else:
                    cost_to_add = loading_costs_map[station_id] #*(current_load+ 1)                    
                heapq.heappush(pq, (cost_to_add, station_id, current_load))
                
        assignments = {}        
        for parcel_id in parcels:
            if not pq: break                
            cost, station_id, current_load = heapq.heappop(pq)
            assignments[parcel_id] = {'origin_station': station_id}
            
            current_load += 1    
            if current_load < self.station_capacities[station_id]:
                new_cost = loading_costs_map[station_id]
                heapq.heappush(pq, (new_cost, station_id, current_load))                
        return assignments

    def _assign_station_R2(self, solution, parcels_to_repair):
        """
        fills stations in order of setup cost by matching them with the nearest undelivered parcels.
        """

        fixed_costs = {s['id']: self.instance_data['costs']['fixed_source_delivery_cost'].get(str(s['id']), 0.0)for s in self.instance_data['network']['stations']}
        loading_costs = {s['id']: s['loading_cost'] for s in self.instance_data['network']['stations']}
        capacities = self.station_capacities
        source_stations = [s['id'] for s in self.instance_data['network']['stations'] if s.get('is_source', False)]

        station_load = self.helpers._get_num_parcels_by_station(solution)
        

        # Build the graph for distance calculations using the Global Graph A
        # G = nx.DiGraph()
        # for u, neighbors in self.helpers.global_graph_adj.items():
        #     for v in neighbors:
        #         G.add_edge(u, v)
        # #check
        # if G.number_of_edges() == 0:
        #     print("Global graph is empty! Pathfinding will fail. Check preprocessing.")
        #     return {}
        
        # # we pre-calculate All-Pairs Shortest Paths from source stations to all nodes.
        # try:
        #     source_distances = {s_id: nx.shortest_path_length(G, source=s_id) for s_id in source_stations}
        # except nx.NetworkXNoPath:
        #     source_distances = {}
        #     for s_id in source_stations:
        #         try:
        #             source_distances[s_id] = nx.shortest_path_length(G, source=s_id)
        #         except nx.NetworkXNoPath:
        #             source_distances[s_id] = {} 

        unassigned_parcel_ids = set(parcels_to_repair)
        assignments = {}

        # calculate p_i(0): the cost to activate and load the first parcel at station i
        station_base_costs = {s_id: fixed_costs.get(s_id, 0) + loading_costs.get(s_id, 0) for s_id in source_stations}
        sorted_stations = sorted(source_stations, key=lambda s_id: station_base_costs.get(s_id, float('inf')))

        # main loop        
        for station_id in sorted_stations:
            current_load = station_load.get(station_id, 0)

            if current_load >= capacities.get(station_id, 0): continue
            if not unassigned_parcel_ids: break

            dist_map = self.cached_source_distances.get(station_id, {})
            
            while current_load < capacities.get(station_id, 0) and unassigned_parcel_ids:
                
                best_parcel_id = None
                best_destination = None
                min_distance = float('inf')

                # Find the parcel whose destination is closest to this station
                for parcel_id in unassigned_parcel_ids:
                    parcel_data = self.parcels_map[parcel_id]
                    destinations = parcel_data['destination']
                    # Ensure destinations are iterable (handles single ID or list)
                    dest_list = destinations if isinstance(destinations, (list, set)) else [destinations]
                    
                    for dest_id in dest_list:
                        # Retrieve pre-computed hop distance
                        distance = dist_map.get(dest_id, float('inf'))
                        if distance < min_distance:
                            min_distance = distance
                            best_parcel_id = parcel_id
                            best_destination = dest_id
                
                if best_parcel_id is not None:
                    assignments[best_parcel_id] = {
                        "origin_station": station_id,
                        "destination_station": best_destination }
                    unassigned_parcel_ids.remove(best_parcel_id)
                    current_load += 1
                else: break
            
            if not unassigned_parcel_ids: break  # Optimization: stop if the backup set is empty               
        return assignments