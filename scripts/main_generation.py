import os
import pandas as pd
import json
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_generation.data_generator_scalability import generate_full_dataset
from src.data_generation.data_generator import DataGenerator

#This script is intended to call the generator to build all the instances for the scalability tests.
#I generate the instances based on the following cnfiguration:
#     - 4 networks (families): (4,24,3), (6,36,6), (8,40,9), (8,44,9)
#     - 6/7 classes depending on # of parcels and crowdshippers (Parcels/CS)
#     - 3/6 instances per family 
#     - 85 times slots lasting 5 minutes each 
# (you can change values going to the relative script in the folder data_generation)

if __name__ == '__main__':
    
     generator = DataGenerator()
     generate_full_dataset(generator)
     