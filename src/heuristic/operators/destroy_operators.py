import random
import math
import copy
import time
import numpy as np 
import heapq  

from ..helpers import Helpers

class DestroyOperators:
    """
    Implements the Destroy stage of the ALNS algorithm for the PTCP.
    Remove parcel routes to enable search space exploration by prioritizing burdened stations.
    """
    def __init__(self, instance_data, helpers):
        self.instance_data = instance_data
        self.helpers=helpers

    # ================================================================
    # (D1) Parcel Paths Elimination
    # ================================================================
    def destroy_by_parcel_path_elimination(self, solution, metric_name, q=5):
        """
        Removes parcel paths originating from a high-metric station.   
        Args:
            solution: the incumbent solution to destroy.
            metric_name: the metric to prioritize stations (absolute, relative, average, random).
            q: pre-specified number of paths to eliminate from the selected station.
        """
        new_solution = copy.deepcopy(solution)
       #identify active source stations
        active_origins = list(set(a['origin_station'] for a in new_solution['parcels_assigned'].values()))
        if not active_origins:
            # print("[DESTROY-path_elimination] No parcel assigned in the solution. Nothing to destroy")
            return new_solution
        
        # rank stations based on the chosen load metric
        if metric_name == "random":
            selected_station_id = random.choice(active_origins)
            #print(f"[DEBUG-DESTROY] random metric: active station chosen")
        else:
            load_matrix, _ = self.helpers._build_load_matrix_from_solution(new_solution)
            station_index = self._select_station_for_removal(metric_name, load_matrix)
            selected_station_id = self.instance_data['network']['stations'][station_index]['id']
            
        #--------------------DEBUG--------------------
            # print(f"\n[DEBUG-DESTROY] Metric: {metric_name}")
            # print(f"[DEBUG-DESTROY] Index station selected by the metric: {station_index}")
            # if station_index is None:
            #     #print("[DESTROY-path_elimination] No station selected.")
            #     return new_solution
            # # Recupero ID
            # try:
            #     selected_station_id = self.instance_data['network']['stations'][station_index]['id']
            #     print(f"[DEBUG-DESTROY] ID corresponding station: {selected_station_id}")
            # except IndexError:
            #     print(f"[DEBUG-DESTROY] Error: station_index {station_index} out of range")
            #     return new_solution
        
        # DEBUG: to check ID station we print an example
        # if new_solution['parcels_assigned']:
        #     first_pid = next(iter(new_solution['parcels_assigned']))
        #     example_origin = new_solution['parcels_assigned'][first_pid]['origin_station']
        #     print(f"[DEBUG-DESTROY] Example structure: Parcel {first_pid} has origin_station = {example_origin} (type: {type(example_origin)})")
        #     print(f"[DEBUG-DESTROY] ID selected type: {type(selected_station_id)}")
        #--------------------DEBUG--------------------
        
        #find all paths originating from the selected source
        parcels_from_station = [pid for pid, a in new_solution['parcels_assigned'].items() if a['origin_station'] == selected_station_id]
        if not parcels_from_station:
            # all_origins = set(a['origin_station'] for a in new_solution['parcels_assigned'].values())
            # print(f"[DEBUG-DESTROY] Mismatch rilevato! Stazioni origine presenti nella soluzione: {all_origins}")
            # print(f"[DESTROY-path_elimination] Nessun pacco da rimuovere dalla stazione {selected_station_id}.")
            return new_solution

        #sort by non-increasing length (longer paths are more resource-intensive)
        parcels_from_station.sort(key=lambda pid: len(new_solution['parcels_assigned'][pid]['path']), reverse=True)

        # paths to remove: q* = min(actual_paths, q_specified)
        q_star = min(len(parcels_from_station), q)
        parcels_to_remove = parcels_from_station[:q_star]
        removed=[]
        # deallocate parcels and release crowdshipper resources
        for parcel_id in parcels_to_remove:
            path = new_solution['parcels_assigned'][parcel_id].get('path', [])
            for arc in path:
                if len(arc) >= 3:
                    crowdshipper_id = arc[2]
                    if crowdshipper_id is not None and crowdshipper_id in new_solution['crowdshipper_assignments']:
                        # remove the parcel's arc from the crowdshipper's schedule
                        new_solution['crowdshipper_assignments'][crowdshipper_id] = [
                            assignment for assignment in new_solution['crowdshipper_assignments'][crowdshipper_id]
                            if assignment[0] != parcel_id]
            new_solution['parcels_assigned'].pop(parcel_id, None)
            new_solution['backup_parcels'].add(parcel_id)
            removed.append(parcel_id)

        


        # refresh the load matrix for the subsequent Repair phase
        new_solution['load_matrix'], _= self.helpers._build_load_matrix_from_solution(new_solution)
        #print(f"[DESTROY-path_elimination] Rimossi {len(removed)} pacchi dalla stazione {selected_station_id}: {removed}")
        return new_solution

    # ================================================================
    # (D2) Capacity Reduction
    # ================================================================
    def destroy_by_capacity_reduction(self, solution, metric_name, num_stations_to_select, reduction_amount):
        """
        temporarily reduces effective station capacities.
        Forces the removal of the longest paths until the load respects the new constraints.        
        Args:
            num_stations_to_select: 'q' =number of top stations to affect.
            reduction_amount: 'q_tilde' =capacity reduction units.
        """
        new_solution = copy.deepcopy(solution)
        num_stations = len(self.instance_data['network']['stations'])
        num_stations_to_select = min(num_stations_to_select, num_stations)
        self.station_capacities = [s['capacity'] for s in self.instance_data['network']['stations']]

        # select the top q stations based on the metric
        load_matrix, _ = self.helpers._build_load_matrix_from_solution(new_solution)
        metric_values = self.helpers.compute_metrics(load_matrix, metric_name, self.station_capacities)
        sorted_station_indices = np.argsort(metric_values)[::-1]
        selected_station_indices = sorted_station_indices[:num_stations_to_select]

        adjusted_capacities = {}
        for idx in selected_station_indices:
            adjusted_capacities[idx] = max(self.station_capacities[idx] - reduction_amount, 0)

        # path removal loop
        for station_idx in selected_station_indices:
            station_id = self.instance_data['network']['stations'][station_idx]['id']
            incident_parcels = self.helpers._get_incident_parcels(new_solution, station_id)
            incident_parcels.sort(key=lambda pid: len(new_solution['parcels_assigned'][pid]['path']), reverse=True)

            removed = []
            while True:
                # calculate current peak load at the station
                temp_load_matrix, _ = self.helpers._build_load_matrix_from_solution(new_solution)
                station_load = np.max(temp_load_matrix[self.helpers.station_to_idx[station_id]])
                if station_load <= adjusted_capacities[station_idx] or not incident_parcels:
                    break

                parcel_to_remove = incident_parcels.pop(0)
                path = new_solution['parcels_assigned'].get(parcel_to_remove, {}).get('path', [])

                for arc in path:
                    if len(arc) >= 3:
                        crowdshipper_id = arc[2]
                        if crowdshipper_id is not None and crowdshipper_id in new_solution['crowdshipper_assignments']:
                            new_solution['crowdshipper_assignments'][crowdshipper_id] = [
                                a for a in new_solution['crowdshipper_assignments'][crowdshipper_id]
                                if a[0] != parcel_to_remove]

                new_solution['parcels_assigned'].pop(parcel_to_remove, None)
                new_solution['backup_parcels'].add(parcel_to_remove)
                removed.append(parcel_to_remove)

            #--------------DEBUG--------------------------
            # total_removed = {}
            # if removed:
            #     total_removed[station_id] = removed
            #     print(f"[DESTROY-capacity_reduction] Station {station_id}: capacity {self.station_capacities[station_idx]} → {adjusted_capacities[station_idx]}, rimoved {len(removed)} parcel: {removed}")
            # else:
            #     print(f"[DESTROY-capacity_reduction] Station {station_id}:no parcels removed (current load <= new capacity).")
        temp_caps_map = {s['id']: s['capacity'] for s in self.instance_data['network']['stations']}
        
        stations_list = self.instance_data['network']['stations']
        for idx, reduced_cap in adjusted_capacities.items():
            s_id = stations_list[idx]['id']
            temp_caps_map[s_id] = reduced_cap

        new_solution['_temp_active_capacities'] = temp_caps_map

        new_solution['load_matrix'], _ = self.helpers._build_load_matrix_from_solution(new_solution)       
        return new_solution

    

    def _select_station_for_removal(self, metric_name, load_matrix):
        """Helper to rank and select a station index based on load metrics."""

        if not self.instance_data['network']['stations']:
                return None
        original_caps = [s['capacity'] for s in self.instance_data['network']['stations']]
        metrics_values = self.helpers.compute_metrics(load_matrix, metric_name, original_caps)

        if metric_name == "random":
            selected_station_index = np.random.choice(len(metrics_values))
        else:
            selected_station_index = np.argmax(metrics_values)
        return selected_station_index