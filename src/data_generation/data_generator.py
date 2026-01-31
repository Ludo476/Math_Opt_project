import json
import os
import networkx as nx
import math
import random
import math
import heapq
import json
from collections import defaultdict
from typing import Dict, Any, Optional, Set, List, Tuple
from shapely.geometry import LineString, Point

# ======================================================================
# CLASS STATION
# ======================================================================

class Station:
    def __init__(self, id, name, coordinates, capacity_apl, loading_cost,
                 is_source=False, is_destination=False, is_exchange_station=False):
        self.id = id
        self.name = name
        self.coordinates = coordinates
        self.capacity_apl = capacity_apl
        self.loading_cost = loading_cost
        self.is_source = is_source
        self.is_destination = is_destination
        self.is_exchange_station = is_exchange_station
        self.lines = set() # Set of PT lines passing through this station


# ======================================================================
# 2. PT NETWORK GENERATOR
# ======================================================================
class DataGenerator:
    """
    To generate realistic PT topologies. It utilizes curve interpolation to simulate transit lines and manages the physical distance calculations between stations.
    """
    def __init__(self):
        self.apl_stations = []
        self.station_map = {}

    def _calculate_distance(self, id1, id2):
        c1 = self.station_map[id1].coordinates
        c2 = self.station_map[id2].coordinates
        dx = c1['x'] - c2['x']
        dy = c1['y'] - c2['y']
        return (dx**2 + dy**2)**0.5

    def _calculate_travel_minutes_between_stations(self, id1, id2, speed: float=1.0):
        """returns the travel times in minutes"""
        dist = self._calculate_distance(id1, id2)
        return dist / max(1e-9, speed)
       
    def _generate_pt_network_from_curves(
        self,
        num_lines: int,
        num_stations: int,
        num_exchange_stations: int,
        map_size=(100, 100),
        min_apl_capacity=10,
        max_apl_capacity=50,
        min_station_distance=0.05,
        rng=random):
        """
        Algorithm Flow:
        1. Generate Exchange Stations randomly within the map.
        2. Assign at least two lines to each hub to ensure network connectivity.
        3. Build the geometry of PT lines using LineString interpolation through hubs.
        4. Distribute standard stations along the segments of these lines.
        """
        num_exchange_stations = min(num_exchange_stations, num_stations)
        if num_exchange_stations > 0 and num_lines < 2:
            raise ValueError("Exchange stations require at least 2 lines to facilitate transfers")

        def round_coord(pt, prec=6):
            return (round(pt[0], prec), round(pt[1], prec))

        # 1. Generate exchange stations
        exchange_coords = []
        max_attempts = 500
        attempts = 0
        while len(exchange_coords) < num_exchange_stations and attempts < max_attempts:
            new = (rng.uniform(0, map_size[0]), rng.uniform(0, map_size[1]))
            if all(math.dist(new, c) >= min_station_distance for c in exchange_coords):
                exchange_coords.append(new)
            attempts += 1

        # 2. Assign lines to the exchange stations
        exchange_line_map = {round_coord(c): set() for c in exchange_coords}
        exch_list = list(exchange_line_map.keys())
        for idx, coord in enumerate(exch_list):
            l1 = idx % num_lines
            l2 = (idx + 1) % num_lines
            exchange_line_map[coord].update({f"line_{l1}", f"line_{l2}"})  
        line_has_exchange = {f"line_{i}": False for i in range(num_lines)}
        for coord, lines in exchange_line_map.items():
            for l in lines: line_has_exchange[l] = True
        for ltag in [l for l, has in line_has_exchange.items() if not has]:
            if exch_list: exchange_line_map[rng.choice(exch_list)].add(ltag)

        # 3. Build the geometry of the lines 
        lines_geometry = []
        lines_exchanges = defaultdict(list)
        for coord, lines in exchange_line_map.items():
            for ltag in lines: lines_exchanges[int(ltag.split("_")[1])].append(coord)
        for lid in range(num_lines):
            start = (rng.uniform(0, map_size[0]), rng.uniform(0, map_size[1]))
            end = (rng.uniform(0, map_size[0]), rng.uniform(0, map_size[1]))
            assigned = lines_exchanges.get(lid, [])
            
            base_vec = (end[0] - start[0], end[1] - start[1])
            base_len2 = base_vec[0]**2 + base_vec[1]**2
            def proj_t(p):
                if base_len2 == 0: return 0.5
                v = (p[0] - start[0], p[1] - start[1])
                return (v[0]*base_vec[0] + v[1]*base_vec[1]) / base_len2
            
            sorted_assigned = sorted(assigned, key=proj_t)
            coords = [start] + sorted_assigned + [end]
            lines_geometry.append(LineString(coords))
        
        # 4. Add non-exchange stations to the built lines
        final_stations_with_lines = defaultdict(set, exchange_line_map)
        remaining = max(0, num_stations - len(exchange_coords))
        per_line = remaining // num_lines if num_lines > 0 else remaining
        
        for l_id, line in enumerate(lines_geometry):
            num_to_add = per_line + (1 if l_id < (remaining % num_lines) else 0)
            t_values = [(i + 1) / (num_to_add + 1) for i in range(num_to_add)]
            for t in t_values:
                for _ in range(20): 
                    tt = max(0.001, min(0.999, t + rng.uniform(-0.05, 0.05)))
                    pt = line.interpolate(tt, normalized=True)
                    coords = round_coord(pt.coords[0])
                    if all(math.dist(coords, existing) >= min_station_distance for existing in final_stations_with_lines):
                        final_stations_with_lines[coords].add(f"line_{l_id}")
                        break

        # 5. Create object Station
        self.apl_stations = []
        self.station_map = {} 
        sorted_coords = sorted(final_stations_with_lines.keys(), key=lambda c: len(final_stations_with_lines[c]), reverse=True)

        for station_id_counter, coords in enumerate(sorted_coords):
            lines = final_stations_with_lines[coords]
            is_exchange = len(lines) >= 2
            capacity_apl = rng.randint(min_apl_capacity, max_apl_capacity)
            station = Station(
                id=station_id_counter, name=f"Station {station_id_counter}",
                coordinates={"x": float(coords[0]), "y": float(coords[1])},
                capacity_apl=capacity_apl, loading_cost=round(rng.uniform(0.5, 4), 3),
                is_exchange_station=is_exchange
            )
            station.lines = set(lines)
            self.apl_stations.append(station)
            self.station_map[station_id_counter] = station

        # 6. Generate physical arcs for the transport network
        edge_set = set()
        transport_lines_data = []
        for l_id, line in enumerate(lines_geometry):
            line_tag = f"line_{l_id}"
            line_stations = [s for s in self.apl_stations if line_tag in s.lines]
            if len(line_stations) > 1:
                line_stations.sort(key=lambda s: line.project(Point(s.coordinates['x'], s.coordinates['y'])))
                for idx in range(len(line_stations) - 1):
                    f_stat, t_stat = line_stations[idx], line_stations[idx + 1]
                    
                    if (f_stat.id, t_stat.id) not in edge_set:
                        edge_set.add((f_stat.id, t_stat.id))
                        edge_set.add((t_stat.id, f_stat.id)) # Add the inverse edges
                        transport_lines_data.append({
                            'from': f_stat.id, 'to': t_stat.id, 'line_id': line_tag,})
        
        return transport_lines_data, self.apl_stations, self.station_map


# ======================================================================
# 3. MAIN FUNCTION FOR THE GENERATION
# ======================================================================

def generate_realistic_txt_dataset(
    generator: DataGenerator,
    pt_network_data=None,
    num_lines=None,
    num_stations=None,
    num_exchange_stations=None,
    step_size: int = 5,
    step_count: int = 85,
    cs_cost: float = 1.0,
    num_cs: int = 20,
    num_parcels: int = 15,
    backup_cost_range: tuple = (8, 12),
    rng=random):
    """
    Generates the dataset in the txt format following the generation of a realistic PT network
    """
    # 1. Create PT network
    print("1. Generation realistic PT network...")
    if pt_network_data is None:
        if num_lines is None or num_stations is None or num_exchange_stations is None:
            raise ValueError(
                "If pt_network_data is None, num_lines, num_stations and "
                "num_exchange_stations must be provided."
            )

        transport_lines_data, apl_stations, station_map = \
            generator._generate_pt_network_from_curves(
                num_lines=num_lines,
                num_stations=num_stations,
                num_exchange_stations=num_exchange_stations,
            )
    else:
        transport_lines_data, apl_stations, station_map = pt_network_data

    num_nodes = len(apl_stations)
    print(f"Physical network with {num_nodes} stations created.")
    
    # 2. Crowdshippers 
    cs_data = []
    station_ids = list(range(num_nodes))
    for _ in range(num_cs):
        origin, dest = rng.sample(station_ids, 2)
        departure_slot = rng.randint(0, step_count // 2)
        cs_data.append((origin, dest, departure_slot))

    # 3. Stop Info 
    stop_info = []
    sorted_stations = sorted(apl_stations, key=lambda s: s.id) 
    #we choose as depts the exchange stations and 10% of the others
    source_station_ids = {s.id for s in sorted_stations if s.is_exchange_station}
    num_other_sources = int(0.1 * (num_nodes - len(source_station_ids)))
    other_stations = [s.id for s in sorted_stations if not s.is_exchange_station]
    if other_stations and num_other_sources > 0:
        source_station_ids.update(rng.sample(other_stations, min(num_other_sources, len(other_stations))))
    # we want destination and origin stations to belong to disjoint sets   
    potential_destination_ids = [s_id for s_id in station_ids if s_id not in source_station_ids]
    if not potential_destination_ids:
        raise ValueError("Error: all the stations are source stations! there are no eligible destinations.")
    for station in sorted_stations:
        if station.id in source_station_ids:
             fixed_cost = rng.randint(2, 5)
             variable_cost = round(rng.uniform(0.5, 2.0), 2)
        else:
             fixed_cost = -1
             variable_cost = -1
        is_exchange_flag = 1 if station.is_exchange_station else 0
        stop_info.append((station.capacity_apl, fixed_cost, variable_cost, is_exchange_flag))

     # 3. Parcels
    parcels = []
    for _ in range(num_parcels):
        k = rng.randint(1, min(3, len(potential_destination_ids)))
        # only from non-source stations
        destinations = rng.sample(potential_destination_ids, k)
        parcels.append(destinations)

    # 4. Backup Costs
    backup_costs = [rng.randint(*backup_cost_range) for _ in range(num_parcels)]

    # 5. Writing the dataset 
    lines = [
        f"STEP-SIZE {step_size}",
        f"STEP-COUNT {step_count}",
        f"CROWD-SHIPPER-COST 1.0",
        "",
        f"CROWD-SHIPPERS {len(cs_data)}" ]
    for o, d, s in cs_data: lines.append(f"{o} {d} {s}")
    lines.append("END\n")
    #to map the arcs on the corresponding line
    lines.append(f"TRANSPORT-LINES {len(transport_lines_data)}")
    for arc in transport_lines_data:
        # Formato: from to line_id
        lines.append(f"{arc['from']} {arc['to']} {arc['line_id']}")
    lines.append("END\n")
    # physical graph
    graph_arcs = []
    for arc in transport_lines_data:
        minutes = generator._calculate_travel_minutes_between_stations(arc['from'], arc['to'])
    
    # Save in the txt as integer (minutes). 
        m_val = int(math.ceil(minutes))
        graph_arcs.append((arc['from'], arc['to'], m_val))
        graph_arcs.append((arc['to'], arc['from'], m_val))

    lines.append(f"GRAPH {len(graph_arcs)}")
    lines.append(f"{len(apl_stations)}")
    for u, v, t in graph_arcs: lines.append(f"{u} {v} {t}")
    lines.append("END\n")   

    lines.append(f"PARCELS {len(parcels)}")
    for dests in parcels: lines.append(" ".join(map(str, dests)))
    lines.append("END\n")

    lines.append(f"BACKUP-COST {len(backup_costs)}")
    for c in backup_costs: lines.append(f"{c}")
    lines.append("END\n")

    lines.append(f"STOP-INFO {len(stop_info)}")
    lines.append("# capacity fixed_cost variable_cost is_exchange (1=yes, 0=no)")
    for cap, fix, var, is_ex in stop_info:
        lines.append(f"{int(cap)} {int(fix)} {var} {is_ex}")
    lines.append("END\n")

    return "\n".join(lines)

# ================================================================
# 1. PARSER OF FILE .TXT
# ================================================================

def parse_paper_instance(path):
    
    instance = {}
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        # STEP
        if line.startswith("STEP-SIZE"):
            instance["STEP-SIZE"] = int(line.split()[1])
        elif line.startswith("STEP-COUNT"):
            instance["STEP-COUNT"] = int(line.split()[1])
        elif line.startswith("CROWD-SHIPPER-COST"):
            instance["CROWD-SHIPPER-COST"] = float(line.split()[1])

        # CROWD-SHIPPERS
        elif line.startswith("CROWD-SHIPPERS"):
            n_cs = int(line.split()[1])
            cs_list = []
            i += 1
            while not lines[i].startswith("END"):
                o, d, dep =map(float, lines[i].split())
                cs_list.append({"origin": int(o), "destination": int(d), "departure_time": int(dep)})
                i += 1

            if len(cs_list) != n_cs:
                raise ValueError(f"CROWD-SHIPPERS count mismatch: expected {n_cs}, got {len(cs_list)}")
            instance["CROWD-SHIPPERS"] = cs_list

        # TRANSPORT-LINES
        if line.startswith("TRANSPORT-LINES"):
            n_tlines = int(line.split()[1])
            tlines_list = []
            i += 1
            while not lines[i].startswith("END"):
                parts = lines[i].split()
                tlines_list.append({
                    "from": int(parts[0]),
                    "to": int(parts[1]),
                    "line_id": parts[2]
                })
                i += 1
            instance["TRANSPORT-LINES"] = tlines_list

        # GRAPH
        elif line.startswith("GRAPH"):
            n_arcs = int(line.split()[1])
            n_nodes = int(lines[i + 1])
            arcs = []
            i += 2
            while not lines[i].startswith("END"):
                u, v, t = map(float, lines[i].split())
                arcs.append((int(u), int(v), int(t)))
                i += 1
            if len(arcs) != n_arcs: raise ValueError(f"GRAPH arc count mismatch: expected {n_arcs}, got {len(arcs)}")
            instance["GRAPH"] = {"n": int(n_nodes), "arcs": arcs}

        # STOP-INFO
        elif line.startswith("STOP-INFO"):
            n_stops = int(line.split()[1])
            stop_info = []
            i += 2  # salta il commento
            while not lines[i].startswith("END"):
                cap, fix, var, is_ex = lines[i].split()
                stop_info.append((int(cap), float(fix), float(var), bool(int(is_ex))))
                i += 1
            if len(stop_info) != n_stops:
                raise ValueError(f"STOP-INFO count mismatch: expected {n_stops}, got {len(stop_info)}")            
            instance["STOP-INFO"] = stop_info

        elif line.startswith("PARCELS"):
            n_p = int(line.split()[1])
            parcels = []
            i += 1
            while not lines[i].startswith("END"):
                parcels.append(list(map(int, lines[i].split())))
                i += 1
            if len(parcels) != n_p:
                raise ValueError(f"PARCELS count mismatch: expected {n_p}, got {len(parcels)}")
            instance["PARCELS"] = parcels

        elif line.startswith("BACKUP-COST"):
            n_b = int(line.split()[1])
            costs = []
            i += 1
            while not lines[i].startswith("END"):
                costs.append(float(lines[i]))
                i += 1
            if len(costs) != n_b:
                raise ValueError(f"BACKUP-COST count mismatch: expected {n_b}, got {len(costs)}")
            instance["BACKUP-COST"] = costs
        i += 1

    # --------------------------------------------------
    # SANITY CHECK 
    # --------------------------------------------------
    required_keys = [
        "STEP-SIZE", "STEP-COUNT","CROWD-SHIPPERS","GRAPH","STOP-INFO","PARCELS","BACKUP-COST"]
    for k in required_keys:
        if k not in instance: raise ValueError(f"Missing block in instance file: {k}")
    return instance


# ================================================================
# BUILD PHYSICAL GRAPH
# ================================================================

def build_consistent_sets(instance):
    """
    This method computes:
    - cs_schedules: Precise arrival time slots for each crowdshipper at each station.
    - Ak: The set of delivery arcs (i, j) where a crowdshipper can carry a parcel.
    - Kit_map: The synchronization map identifying which crowdshippers are at station i at time t.
    """
    # edge weights in time slots
    G_phys = nx.DiGraph()
    sz = instance['STEP-SIZE']
    for u, v, t_min in instance['GRAPH']['arcs']:
        t_slots = max(1, math.ceil(t_min / sz)) if t_min > 0 else 0
        G_phys.add_edge(u, v, weight=t_slots)

    arc_to_line = {}
    if 'TRANSPORT-LINES' in instance:
        for entry in instance['TRANSPORT-LINES']:
            arc_to_line[(entry['from'], entry['to'])] = entry['line_id']
            arc_to_line[(entry['to'], entry['from'])] = entry['line_id']

    exchange_stations = {i for i, info in enumerate(instance["STOP-INFO"]) if info[3] == 1}
    cs_schedules = {}  
    Nk = {}          

    for k, cs in enumerate(instance["CROWD-SHIPPERS"]):
        o, d, dep_slot = cs["origin"], cs["destination"], int(cs["departure_time"])
        try:
            # assume crowdshippers follow the shortest path in the physical network
            path = nx.shortest_path(G_phys, source=o, target=d, weight='weight')
            
            schedule = {path[0]: dep_slot}
            curr_slot = dep_slot
            for idx in range(len(path) - 1):
                curr_slot += G_phys[path[idx]][path[idx+1]]['weight']
                schedule[path[idx+1]] = curr_slot
            
            path_arc_lines = []
            for idx in range(len(path) - 1):
                u, v = path[idx], path[idx+1]
                line = arc_to_line.get((u, v), "unknown")
                path_arc_lines.append(line)

           # Nk: set of nodes where the CS can pick up or drop off parcels
           #notice that even if a station is an exchange but the cs does not change line on that station it cannot be a relevant node
            relevant_nodes = [path[0]]
            for i in range(1, len(path) - 1):
                curr_node = path[i]
                if curr_node in exchange_stations:
                    #if previous arc_line is different from the next one
                    line_in = path_arc_lines[i-1]
                    line_out = path_arc_lines[i]
                    if line_in != line_out:
                        relevant_nodes.append(curr_node)

            if path[-1] not in relevant_nodes:
                relevant_nodes.append(path[-1])

            if schedule[path[-1]] < instance['STEP-COUNT']: #ensure crowdshipper completes the trip within the planning horizon
                cs_schedules[k] = schedule
                Nk[k] = relevant_nodes

        except nx.NetworkXNoPath: 
            continue

    # generation of delivery arcs (Ak) and crowdshipper availability (Kij)
    Ak = defaultdict(set)
    Kij = defaultdict(set)
    arc_times = {} # (i, j, k) -> (t_start, t_end)
    for k, nodes in Nk.items():
        if len(nodes) < 2: continue
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)): #respect the order of the path
                i, j = nodes[a], nodes[b]
                t_start, t_end = cs_schedules[k][i], cs_schedules[k][j]
                if t_end <= t_start: t_end = t_start + 1 #(departure < arrival)                
                Ak[k].add((i, j))
                Kij[(i, j)].add(k)
                arc_times[(i, j, k)] = (t_start, t_end)

    # Kit_map construction (temporal presence at each station)
    Kit_map = defaultdict(set)
    for k, nodes in Nk.items():
        for node in nodes:
            t_presence = cs_schedules[k][node]
            Kit_map[(node, t_presence)].add(k)

    return {
        "Nk": Nk,
        "Ak": Ak,
        "Kij": Kij,
        "Kit_map": Kit_map,
        "arc_times": arc_times,
        "G_phys": G_phys }


def preprocess_instance_from_txt(pathfile: str) -> dict:
    # 1. Parsing
    instance_sections = parse_paper_instance(pathfile) 
    # 2. Generate sets
    results = build_consistent_sets(instance_sections)
    #convert sets in list   
    # N_global
    nodes_set = {node for path in results["Nk"].values() for node in path}
    N_global = sorted(list(nodes_set))    
    # S_global and D_global
    S_global = [i for i, info in enumerate(instance_sections['STOP-INFO']) if info[1] != -1 and i in nodes_set]    
    D_req = set()
    for p_dests in instance_sections['PARCELS']: 
        D_req.update(p_dests)
    D_global = sorted(list(D_req.intersection(nodes_set)))

    # JSON formatting
    stop_info = instance_sections['STOP-INFO']   
    instance_data = {
        "network": {
            "stations": [
                {"id": i, "capacity": info[0], "loading_cost": info[2] if info[2] != -1 else 0.0, "is_exchange": int(info[3]), "is_source": i in S_global} 
                for i, info in enumerate(stop_info)],
            "delivery_lines": instance_sections.get('TRANSPORT-LINES', []),   
            "crowdshippers": instance_sections['CROWD-SHIPPERS'],
            # A_k: arcs (tuple) in "(i,j)" and sets in lists
            "A_k": {str(k): [str(e) for e in edges] for k, edges in results["Ak"].items()},
            "arc_time_k": {str(key): val for key, val in results["arc_times"].items()},  # (i,j,k) -> strings
            "Kit_map": {str(key): list(val) for key, val in results["Kit_map"].items()}, #cs values in lists           
            "global_graph": {
                "N": N_global,
                "A": [str(a) for a in results["Kij"].keys()],
                "S": S_global,
                "D": D_global}
        },
        "demand": {
            "parcels": [
                {"id": i, "destination": d_list, "backup_cost": instance_sections['BACKUP-COST'][i]} 
                for i, d_list in enumerate(instance_sections['PARCELS'])]
        },
        "time_windows": {"start": 0, "end": instance_sections['STEP-COUNT']},
        "costs": {
            "crowdshipper_reward": instance_sections['CROWD-SHIPPER-COST'],
            "fixed_source_delivery_cost": {str(i): info[1] for i, info in enumerate(stop_info) if info[1] != -1}}
    }
    return instance_data



if __name__ == '__main__':
    my_generator = DataGenerator()
    dataset_txt = generate_realistic_txt_dataset(
        generator=my_generator,
        num_lines=6,
        num_stations=36,
        num_exchange_stations=6,
        num_cs=25,
        num_parcels=15,
        step_size=5,  
        step_count=85   
    )
    # 3. Save
    os.makedirs("instances", exist_ok=True)
    txt_path = "instances/instance_prova_dopo_natale.txt"
    with open(txt_path, "w") as f:
        f.write(dataset_txt)    
    print(f"Dataset saved in: {txt_path}")

    instance_data_for_milp = preprocess_instance_from_txt(txt_path)
    # 5. Save JSON (for debug)
    json_path = "instances/final_instance_dopo_natale_prova.json"
    with open(json_path, "w") as f:
        json.dump(instance_data_for_milp, f, indent=4)
        
    print(f"Instance MILP ready and saved in: {json_path}")


