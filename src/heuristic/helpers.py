import heapq
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional
import numpy as np
import networkx as nx

class Helpers:
    """
    Utility class for the PTCP ALNS Solver.
    Manages data preprocessing, graph structures and the label-setting algorithm 
    """
    def __init__(self, instance_data: Dict[str, Any]):
        t_start = time.time()
        print(f"[DEBUG] Helpers Init Start...")
        self.instance_data = instance_data
        # mapping physical stations to matrix indices for numerical operation
        self.station_to_idx = {s['id']: i for i, s in enumerate(instance_data['network']['stations'])}
        self.station_capacities = {s['id']: s.get('capacity', float('inf')) for s in instance_data['network']['stations']}
        self.stations_map = {station['id']: station for station in instance_data['network']['stations']}
        self.parcels_map = {parcel['id']: parcel for parcel in instance_data['demand']['parcels']}

        self.cs_id_to_idx = {cs['id']: i for i, cs in enumerate(instance_data['network']['crowdshippers'])}
        # print(f"  > [Time] Basic Maps: {time.time() - t_start:.4f}s")

        #Graph and path data preprocessing
        t0 = time.time()
        raw_graphs = instance_data.get("network", {}).get("delivery_graphs", instance_data.get("delivery_graphs", {}))
        self.delivery_graphs = self._preprocess_delivery_graphs(raw_graphs)
        # print(f"  > [Time] Delivery Graphs: {time.time() - t0:.4f}s")

        # preprocess arc time K
        t0 = time.time()
        raw_arcs = instance_data.get("network", {}).get("arc_time_k", instance_data.get("arc_time_k", {}))
        self.arc_time_map = self._preprocess_arc_time_k(raw_arcs)
        # print(f"  > [Time] Arc time map (fast parse): {time.time() - t0:.4f}s")
        
        # global physical graph analysis
        # Build the physical adjacency list to compute lower-bound hop distances
        t0 = time.time()
        self.global_graph_adj = defaultdict(list)
        global_graph_arcs = self.instance_data['network']['global_graph']['A']        
        for arc_str in global_graph_arcs:
            try:
                if isinstance(arc_str, str):
                    u, v = self._safe_parse_tuple(arc_str)
                else:
                    u, v = arc_str[0], arc_str[1]
                self.global_graph_adj[u].append(v)
            except Exception: pass
        # print(f"  > [Time] Global Graph Parsing: {time.time() - t0:.4f}s")      

        # pre-compute all-pairs shortest path lengths (hop counts) for heuristic guidance
        t0 = time.time()
        self._build_hop_distance()
        # print(f"  > [Time] hop distance: {time.time() - t0:.4f}s")

        #transforms the flat arc list into a time-expanded adjacency representation
        t0 = time.time()
        self.fast_adj = defaultdict(list)
        edge_count = 0
        for (u, v), k_map in self.arc_time_map.items():
            for k, (t_dep, t_arr) in k_map.items():
                self.fast_adj[u].append({
                    'to': v,'cs_id': k,
                    't_dep': int(t_dep),'t_arr': int(t_arr)
                })
                edge_count += 1

            for u in list(self.fast_adj.keys()):
                self.fast_adj[u].sort(key=lambda e: e['t_dep'])
            # print(f"  > [Time] Fast adj ({edge_count} edges): {time.time() - t0:.4f}s")

        # print(f"[DEBUG] helpers init complete. Total time: {time.time() - t_start:.2f}s")

        

    # --------------------------------------------------------------------------
    # CORE ALGORITHM: Label-Setting 
    # --------------------------------------------------------------------------    
    def _label_setting_algorithm(
        self,
        parcel: Dict[str, Any],
        origin_station: int,
        destination_stations,
        current_solution: Dict[str, Any],
        cost_function: str = "min_crowdshippers",
        used_crowdshippers: Optional[Set[int]] = None,
        max_expansions: int = 1000000,
        load_matrix: Optional[np.ndarray] = None,
        max_labels_per_node: int = 50  # mantenuto per compatibilità, non usato
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        An extension of Dijkstra's algorithm to solve the resource constrained shortest path problem on a time-expanded network.    
        a label L1 dominates L2 if: Cost(L1) <= Cost(L2) AND Time(L1) <= Time(L2) AND UsedCS(L1) subset UsedCS(L2).
        """

        destination_set = (set(destination_stations) if isinstance(destination_stations, (list, set, tuple)) else {destination_stations})
        parcel_id = parcel["id"]
        start_slot = int(self.parcels_map.get(parcel_id, {}).get("creation_time_slot", 0))
        num_time_slots = int(self.instance_data["time_windows"]["end"])
        
        # setup initial bitmask 
        used_mask = 0
        if used_crowdshippers:
            for k in used_crowdshippers:
                if k in self.cs_id_to_idx:
                    used_mask |= (1 << self.cs_id_to_idx[k])

        # current station occupancy matrix and load matrix lazy loading
        if load_matrix is None:
            load_matrix, _ = self._build_load_matrix_from_solution(current_solution)

        cs_usage = current_solution.get("crowdshipper_usage", {})
        # cache functions
        get_idx = self.station_to_idx.get
        get_cap = self.station_capacities.get
        get_cs_idx = self.cs_id_to_idx.get

        def get_arc_cost(k_id: int) -> float:
            if cost_function == "min_crowdshippers":
                return 1.0
            if cost_function == "load_balancing":
                return 1.0 + float(cs_usage.get(k_id, 0))
            return 1.0
        
        # priority queue: (cost, time, current_node, used_cs_frozenset)
        pq = [(0.0, start_slot, origin_station, used_mask)]
        predecessors = {}
        #best cost for (node, time)
        best_costs = {}

        best_final_cost = float("inf")
        best_target_state = None
        expansions = 0

        while pq:

            c_cost, c_time, i, c_mask = heapq.heappop(pq)
            expansions += 1

            if c_cost >= best_final_cost: continue
            # State dominance check 
            state_key = (i, c_time)
            if best_costs.get(state_key, float("inf")) <= c_cost: continue
            best_costs[state_key] = c_cost

            #  Hop Distance Pruning
            if i in self.hop_distance:
                # minimun distance towards the destinations' set
                min_hops = min((self.hop_distance[i].get(d, float('inf')) for d in destination_set), default=float('inf'))
                if c_time + min_hops >= num_time_slots: continue
                # #if with also the shortest path we overcome the cost we split
                # if c_cost + min_hops >= best_final_cost: continue

            # check if current node is a valid destination
            if i in destination_set:
                best_final_cost = c_cost
                best_target_state = (i, c_time, c_mask)
                continue

            for edge in self.fast_adj.get(i, []):
                v, k, t_dep, t_arr = (
                    edge["to"], edge["cs_id"],
                    edge["t_dep"], edge["t_arr"])
                min_dep_time = c_time if i == origin_station else c_time + 1

                if t_dep < min_dep_time: continue
                if t_arr >= num_time_slots: continue

                k_idx = get_cs_idx(k)
                if k_idx is not None and (c_mask & (1 << k_idx)): continue

                # capacity checks
                # idx_i = self.station_to_idx.get(i)
                # idx_v = self.station_to_idx.get(v)

                can_pass = True
                #check occupancy at origin during the waiting interval [arrival, departure)
                if t_dep > c_time:
                    cap_i = get_cap(i, float('inf'))
                    if cap_i != float('inf'):
                        idx_i = get_idx(i)
                        if idx_i is not None:
                            if t_dep > c_time:
                                if np.any(load_matrix[idx_i, c_time : t_dep] + 1 > cap_i):
                                    can_pass = False
                        
                            # Deve comunque esserci spazio nell'istante c_time
                            elif t_dep == c_time:
                                if load_matrix[idx_i, c_time] + 1 > cap_i:
                                    can_pass = False
                # check occupancy at arrival node
                if can_pass:
                    cap_v = get_cap(v, float('inf'))
                    if cap_v != float('inf'):
                        idx_v = get_idx(v)
                        if idx_v is not None and load_matrix[idx_v, t_arr] + 1 > cap_v:
                            can_pass = False

                if not can_pass: continue
                # label update and relaxation
                new_cost = c_cost + get_arc_cost(k)
                if new_cost >= best_final_cost: continue

                new_mask = c_mask | (1 << k_idx) if k_idx is not None else c_mask

                if (v, t_arr, new_mask) not in best_costs or best_costs[(v, t_arr, new_mask)] > new_cost:
                    predecessors[(v, t_arr, new_mask)] = (i, c_time, k, t_dep, t_arr, c_mask)
                    heapq.heappush(pq, (new_cost, t_arr, v, new_mask))

        return (self._reconstruct_path_spacetime(predecessors, origin_station, start_slot, best_target_state) if best_target_state else [])

    # --------------------------------------------------------------------------
    # PARCEL ORDERING STRATEGIES 
    # --------------------------------------------------------------------------
    def _get_parcels_in_order(self, parcels_set, order_strategy) -> List[int]: 
        """
        Determines the sequence in which parcels are processed by repair operators.
        """
        parcels_list = list(parcels_set)

        if order_strategy == "As-Is":
            return sorted(parcels_list) 
        
        elif order_strategy == "Random":
            random.shuffle(parcels_list)
            return parcels_list

        elif order_strategy == "Max backup":
            return sorted(parcels_list, key=lambda pid: self.parcels_map[pid].get('backup_cost', 0), reverse=True)
        
        elif order_strategy == "Min Backup":
            return sorted(parcels_list, key=lambda pid: self.parcels_map[pid].get('backup_cost', 0))
    
        return parcels_list
    
    # --------------------------------------------------------------------------
    # MATRIX
    # --------------------------------------------------------------------------
    def _build_load_matrix_from_solution(self, solution) -> Tuple[np.ndarray, bool]:
        """
        Constructs the load matrix H. h_it denotes the number of parcels at station i at time t.
        A parcel occupies a locker from its creation until it is picked up by a crowdshipper,
        and between subsequent legs of a multi-segment path.
        """
        num_stations = len(self.instance_data['network']['stations'])
        num_time_slots = int(self.instance_data['time_windows']['end'])
        H = np.zeros((num_stations, num_time_slots), dtype=int)
        parcels_map = self.parcels_map

        for parcel_id, assignment in solution.get('parcels_assigned', {}).items():
            parcel_info = parcels_map.get(parcel_id)
            if not parcel_info: continue

            origin_station = assignment.get('origin_station')
            path = assignment.get('path', [])
            creation_slot = int(parcel_info.get('creation_time_slot', 0))
            pickup_slot = num_time_slots
            if path:
                # path[0] = (from, to, k, t_start, t_end)
                if path[0][0] == origin_station: 
                    pickup_slot = int(path[0][3])

            if origin_station in self.station_to_idx:
                idx = self.station_to_idx[origin_station]
                end_slot = min(num_time_slots, max(pickup_slot, creation_slot + 1))
                if creation_slot < end_slot:
                    H[idx, creation_slot:end_slot] += 1

            for i in range(len(path)): #intermediate stations
                _from, to_station, _k, _t_start, t_end = path[i]
                next_pickup_slot = num_time_slots
                if i + 1 < len(path):
                    next_pickup_slot = int(path[i+1][3]) # t_start of subsequent segment

                if to_station in self.station_to_idx:
                    idx = self.station_to_idx[to_station]
                    # occupancy includes arrival moment. Starts at t_end
                    start_wait = min(num_time_slots, int(t_end))
                    end_wait = min(num_time_slots, next_pickup_slot)

                    if start_wait < end_wait:
                        H[idx, start_wait:end_wait] += 1

        is_feasible = True # feasibility check against physical capacity limits
        for station_id, capacity in self.station_capacities.items():
            if station_id in self.station_to_idx and capacity != float('inf'):
                idx = self.station_to_idx[station_id]
                if np.any(H[idx, :] > capacity):
                    is_feasible = False
                    break

        return H, is_feasible
    

    def _update_matrix_inplace(self, H, path, origin_station, parcel_id):
        """
        Helper method to update the load matrix incrementally (+1) for a newly assigned path.
        Avoids the expensive full reconstruction.
        """
        parcel_info = self.parcels_map.get(parcel_id)
        if not parcel_info: return

        creation_slot = int(parcel_info.get('creation_time_slot', 0))
        num_time_slots = H.shape[1]
        
        # 1. Update Origin Station (Waiting time)
        pickup_slot = num_time_slots
        if path:
            # path[0] = (from, to, k, t_start, t_end)
            pickup_slot = int(path[0][3]) # t_start of first leg
        
        idx_origin = self.station_to_idx.get(origin_station)
        if idx_origin is not None:
            end_slot = min(num_time_slots, max(pickup_slot, creation_slot + 1))
            if creation_slot < end_slot:
                H[idx_origin, creation_slot:end_slot] += 1

        # 2. Update Intermediate Stations (Transshipment Wait)
        for i in range(len(path)):
            _from, to_station, _k, _t_start, t_end = path[i]
            
            # Determine next pickup time (or end of horizon if last leg)
            next_pickup_slot = num_time_slots
            if i + 1 < len(path):
                next_pickup_slot = int(path[i+1][3])
            
            if to_station in self.station_to_idx:
                idx = self.station_to_idx[to_station]

                start_wait = min(num_time_slots, int(t_end))
                end_wait = min(num_time_slots, next_pickup_slot)
                
                # if we have to wait brtween the arrive and departure occupy the slot
                if start_wait < end_wait:
                    H[idx, start_wait:end_wait] += 1
    
    # --------------------------------------------------------------------------
    # DATA PARSING & GEOMETRY
    # --------------------------------------------------------------------------
        
    def _build_hop_distance(self):
        """Pre-calculates the physical hop distance between all station pairs."""
        G = nx.DiGraph()
        for u, neighbors in self.global_graph_adj.items():
            for v in neighbors:
                G.add_edge(u, v)
        self.hop_distance = dict(nx.all_pairs_shortest_path_length(G))
    
    def _safe_parse_tuple(self, s):
        if isinstance(s, (list, tuple)):
            return tuple(map(int, s))
        if isinstance(s, str):
            cleaned = s.strip().strip("()[]")
            if not cleaned:
                raise ValueError("Empty tuple string")
            parts = [p.strip() for p in cleaned.split(",") if p.strip() != ""]
            return tuple(int(p) for p in parts)
        raise TypeError(f"Unsupported tuple format: {s!r}")


    def _preprocess_delivery_graphs(self, raw_graphs) -> Dict[int, Dict[str, Any]]:
        processed_graphs = {}
        for parcel_id_str, data in raw_graphs.items():
            try:
                parcel_id = int(parcel_id_str)
                edges_raw = data.get('edges', [])
                edges = []
                for e in edges_raw:
                    if isinstance(e, str): tup = self._parse_str_tuple(e)
                    else: tup = tuple(e)
                    if len(tup) >= 2:
                        edges.append((tup[0], tup[1]))  
                processed_graphs[parcel_id] = { "nodes": list(data.get('nodes', [])), "edges": edges }
            except (ValueError, TypeError): continue
        return processed_graphs
    
    def _preprocess_arc_time_k(self, raw_dict):
        arc_map = defaultdict(dict)
        for key_raw, val_raw in raw_dict.items():    
            try:
                key = self._safe_parse_tuple(key_raw)
                t_range = self._safe_parse_tuple(val_raw)
            except Exception:
                continue
            
            if len(key) != 3 or len(t_range) < 2:
                continue
            i, j, k = key
            
            arc_map[(i, j)][k] = (int(t_range[0]), int(t_range[1]))
        return {k: dict(v) for k, v in arc_map.items()}
    
    
    # --------------------------------------------------------------------------
    # PATH CONSTRUCTION
    # --------------------------------------------------------------------------
    def _reconstruct_path_spacetime(self, predecessors, start_node, start_slot, target_state):
        """Trace back the predecessors from the target state to reconstruct the delivery path."""
        path = []
        curr = target_state # (node, time, cs_set)
        while curr[0] != start_node or curr[1] != start_slot:
            pred = predecessors.get(curr)
            if not pred: break
            # pred = (pred_node, pred_time, k, t_s, t_e, pred_cs_set)
            path.append((pred[0], curr[0], pred[2], pred[3], pred[4]))
            curr = (pred[0], pred[1], pred[5])
        path.reverse()
        return path


    # --------------------------------------------------------------------------
    # SOLUTION MANAGEMENT
    # --------------------------------------------------------------------------
    def _update_solution_with_new_path(self, solution: Dict[str, Any],parcel_id: int,path: List[Tuple[int, int, int, int, int]], origin_station: int,):
        """
        Updates the current solution with a new path for a parcel.
        1. Removing the parcel from the backup set.
        2. Clearing any previous assignments (parcel and crowdshipper levels).
        3. Committing the new path to both the parcels_assigned and crowdshipper_assignments dictionaries.
        """
        # Clean old parcel data and assignements
        solution['backup_parcels'].discard(parcel_id)
        solution['parcels_assigned'].pop(parcel_id, None)
        # remove previous instances of this parcel from all crowdshipper schedules
        for cs_id in list(solution['crowdshipper_assignments'].keys()): solution['crowdshipper_assignments'][cs_id] = [
                arc for arc in solution['crowdshipper_assignments'][cs_id] if arc[0] != parcel_id]

        # if no path is found revert to backup
        if not path:
            solution['backup_parcels'].add(parcel_id)
            return

        # Aggiungi il nuovo percorso alla soluzione
        destination_station = path[-1][1]
        # new assignements
        solution['parcels_assigned'][parcel_id] = {
            'origin_station': origin_station,
            'destination': destination_station,
            'path': path
            }
        #--------------DEBUG---------------
        # parcel_data = next((p for p in self.instance_data['demand']['parcels'] if p['id'] == parcel_id), None)
        # if parcel_data:
        #     valid_destinations = parcel_data.get("destination", [])
        #     if not isinstance(valid_destinations, list): valid_destinations = [valid_destinations]
        #     if destination_station not in valid_destinations:
        #         print(f"[WARNING] Parcel {parcel_id} assigned to {destination_station}, " f"which is not in valid destinations {valid_destinations}")

        for from_node, to_node, cs_id, t_start, t_end in path:
            if cs_id not in solution['crowdshipper_assignments']:
                solution['crowdshipper_assignments'][cs_id] = []
            solution['crowdshipper_assignments'][cs_id].append((parcel_id, from_node, to_node, t_start, t_end))

    # --------------------------------------------------------------------------
    # HEURISTIC 
    # --------------------------------------------------------------------------
    def _get_incident_parcels(self, solution, station_id) -> List[int]:
        """
        Retrieves IDs of all parcels whose current delivery paths pass through the specified station (origin, destination, or intermediate stop).
        Used by the capacity reduction operator.
        """
        incident_parcels = set()
        for parcel_id, assignment in solution.get('parcels_assigned', {}).items():
            path = assignment.get('path', [])
            stations_in_path = {assignment.get('origin_station')}
            for arc in path:
                stations_in_path.add(arc[0])
                stations_in_path.add(arc[1])
            
            if station_id in stations_in_path:
                incident_parcels.add(parcel_id)
                
        return list(incident_parcels)
                
    def _get_num_parcels_by_station(self, solution)-> Dict[int, int]:
        """
        Aggregates the total number of assigned parcels per source station.
        Used to track station load and respect locker capacity constraints.
        """
        station_counts = {s['id']: 0 for s in self.instance_data['network']['stations']}
        for assignment in solution.get('parcels_assigned', {}).values():
            station_id = assignment['origin_station']
            station_counts[station_id] += 1
        return station_counts
    
    # --------------------------------------------------------------------------
    # NOT USED IN THIS VERSION BUT USEFUL FOR EVENTUAL CHANGES
    # --------------------------------------------------------------------------
    def _calculate_distance(self, station_id_1, station_id_2):
        """
        Calculates the Euclidean distance between stations.
        If station_id_2 is a collection of nodes, returns the minimum distance to that set.
        """
        s1 = self.stations_map[station_id_1]        
        if isinstance(station_id_2, (list, set)):
            min_distance = float('inf')
            for dest_id in station_id_2:
                s2 = self.stations_map[dest_id]
                dx = s1['x'] - s2['x']
                dy = s1['y'] - s2['y']
                distance = (dx**2 + dy**2) ** 0.5
                if distance < min_distance: min_distance = distance
            return min_distance
        else:
            s2 = self.stations_map[station_id_2]
            dx = s1['x'] - s2['x']
            dy = s1['y'] - s2['y']
            return (dx**2 + dy**2) ** 0.5

    def _is_crowdshipper_feasible(self, k_id, from_station_id, to_station_id):
        """
        if crowdshipper k_id is scheduled to cover the arc (from_id -> to_id)
        """
        arc_dict = self.arc_time_map.get((from_station_id, to_station_id))
        return k_id in arc_dict if arc_dict else False

    def _calculate_arc_cost(self, i, j, k, parcel, cost_function, current_solution):
        """
        Calculates the edge weight for the label-setting algorithm.
        
        - min_crowdshippers: Returns unit cost to prioritize shorter paths.
        - load_balancing: Penalizes overused crowdshippers based on historical ALNS usage.
        """
        if cost_function == "min_crowdshippers":
            return 1.0
        elif cost_function == "load_balancing":
            num_uses = current_solution.get('crowdshipper_usage', {}).get(k, 0)
            return float(num_uses) + 1.0
        return float('inf')
    
    # --------------------------------------------------------------------------
    # STATIC METHOD
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_metrics(load_matrix, metric, C):
        """Calculates station load metrics (Absolute, Relative, Average, Random) for Destroy Operators."""
        N, T = load_matrix.shape
        if metric == "absolute load":
            return np.max(load_matrix, axis=-1)
        elif metric == "relative load":
            a = np.max(load_matrix, axis=-1)
            C_array = np.array(C)
            return  a / C_array
        elif metric == "average load":
            # m_i = sum_{t in T} h_it / |T|
            return np.sum(load_matrix, axis=-1) / T
        elif metric == "random":
            return np.random.rand(N)
        else:
            raise ValueError(f"not supported: {metric}")