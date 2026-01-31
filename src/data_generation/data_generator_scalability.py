import os
import json
import copy
import random
import math
from collections import defaultdict
from shapely.geometry import LineString, Point  
import heapq 
import sys
import numpy as np

from .data_generator import DataGenerator, generate_realistic_txt_dataset, preprocess_instance_from_txt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
def make_json_safe(data):
    """
    Versione corretta: 
    - Converte le chiavi (tuple) in stringhe.
    - Converte i valori (tuple) in liste (diventano [] in JSON).
    """
    if isinstance(data, (int, float, str, bool, type(None))):
        return data
    elif isinstance(data, (list, tuple)):
        return [make_json_safe(item) for item in data]
    elif isinstance(data, dict):
        return {str(k): make_json_safe(v) for k, v in data.items()}
    else:
        return str(data)



def generate_full_dataset(generator): 
    """
    Genera il dataset di scalabilità rispettando le specifiche:
    - 4 Network (Famiglie): (4,24,3), (6,36,6), (8,40,9), (8,44,9)
    - 6 Classi di dimensione (Parcels/CS)
    - 6 Istanze per famiglia (totale 144 file)
    - 85 slot temporali da 5 minuti
    """
    base_output_folder = os.path.join(PROJECT_ROOT, "instances_scalability_new_prova")
    instances_per_family = 6
    max_network_retries = 100 
    time_slots = 85
    step_size = 5

    class_configs = {
        0: {"num_parcels": 30, "num_crowdshippers": 50},
        1: {"num_parcels": 50, "num_crowdshippers": 75},
        2: {"num_parcels": 75, "num_crowdshippers": 100}
        # 3: {"num_parcels": 120, "num_crowdshippers": 150},
        # 4: {"num_parcels": 160, "num_crowdshippers": 200},
        # 5: {"num_parcels": 225, "num_crowdshippers": 250},
        # 6: {"num_parcels": 250, "num_crowdshippers": 300},
       
    }

    network_configs = [
        {"num_lines": 4, "num_stations": 24, "num_exchange_stations": 3},
        {"num_lines": 6, "num_stations": 36, "num_exchange_stations": 6},
        # {"num_lines": 8, "num_stations": 40, "num_exchange_stations": 9},
        {"num_lines": 8, "num_stations": 44, "num_exchange_stations": 9},
    ]

    # Loop for instances
    for class_id, class_params in class_configs.items():
        class_folder = os.path.join(base_output_folder, f"class_{class_id}")
        os.makedirs(class_folder, exist_ok=True)
        for family_id, net_config in enumerate(network_configs):
            existing_files = [f for f in os.listdir(class_folder) if f.startswith(f"class_{class_id}_fam_{family_id}_inst_")]
            if len(existing_files) >= instances_per_family:
                print(f" - Family {family_id}: Target reached ({len(existing_files)}/{instances_per_family}). Skipping generation.")
                continue


            print(f"Generating instances for Class {class_id}...")

        
            print(f" - Generating Family {family_id} (Network: {net_config['num_stations']} stations, {net_config['num_lines']} lines)...")
            seed_val = 42 + family_id  
            random.seed(seed_val)
            np.random.seed(seed_val)
            pt_network_data = None 
            for attempt in range(max_network_retries):
                try:
                    pt_network_data = generator._generate_pt_network_from_curves(
                        num_lines=net_config['num_lines'],
                        num_stations=net_config['num_stations'],
                        num_exchange_stations=net_config['num_exchange_stations'],
                        map_size=(100, 100),
                    )
                    break 
                except Exception as e:
                    if attempt == max_network_retries - 1:
                        print(f"Critical Error generating network: {e}")
            
            if pt_network_data is None: continue
            
            # retry loop for physical network generation
            for instance_id in range(instances_per_family):
                filename = f"class_{class_id}_fam_{family_id}_inst_{instance_id}.json"
                output_path = os.path.join(class_folder, filename)

                if os.path.exists(output_path):
                    # print(f"    -> Instance {instance_id} exists. Skipping.")
                    continue

                try:
                    dataset_txt = generate_realistic_txt_dataset(
                        generator=generator,
                        pt_network_data=pt_network_data, 
                        num_cs=class_params['num_crowdshippers'],
                        num_parcels=class_params['num_parcels'],
                        step_size=step_size,
                        step_count=time_slots,
                    )
                    # 2. SAVING TXT.
                    temp_txt_path = f"temp_class{class_id}_fam{family_id}_inst{instance_id}.txt"
                    with open(temp_txt_path, "w") as f:
                        f.write(dataset_txt)

                    # 3. PRE-PROCESSING 
                    # create sets arc_time_k, Kit_map, S_global, D_global, ecc.
                    instance_json_data = preprocess_instance_from_txt(temp_txt_path)

                    # 4. SAVING JSON FINALE
                    filename = f"class_{class_id}_fam_{family_id}_inst_{instance_id}.json"

                    output_path = os.path.join(class_folder, filename)
                    
                    with open(output_path, "w") as f:
                        json.dump(make_json_safe(instance_json_data), f, indent=4)

                    print(f"Instance {instance_id} saved.")
                    
                    if os.path.exists(temp_txt_path):
                        os.remove(temp_txt_path)

                except Exception as e:
                    print(f" Error generating instance {instance_id}: {e}")

    print("\n Scalability dataset generation completed.")
            

