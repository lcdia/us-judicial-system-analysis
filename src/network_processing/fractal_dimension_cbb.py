import sys
import os
from tqdm import tqdm
import concurrent.futures
import pickle

# Add tools directory to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import fractals_calculations  # Importing fractals_calculations module

base_path = '../../data/processed/coupling/community/'
partitions = []

# Iterating over directories in the base path
for subdir in os.listdir(base_path):
    subdir_path = os.path.join(base_path, subdir)
    # Check if it is a directory
    if os.path.isdir(subdir_path):
        # Full path to the 'mst.pickle' file in the subdirectory
        file_path = os.path.join(subdir_path, 'mst.pickle')
        # Add the file to the list if it exists
        if os.path.isfile(file_path):
            partitions.append(int(subdir))  # Convert partition name to integer

# Iterating over each partition to perform box counting
for partition_id in tqdm(partitions):
    lst = []
    nb = 2
    tamanho = -1

    # Check if the CBB file already exists, skip if it does
    if not os.path.exists(f'../../data/processed/coupling/community/{partition_id}/CBB.pickle'):
        # Load the MST graph from pickle file
        G = pickle.load(open(f'{base_path}{partition_id}/mst.pickle', 'rb'))
        
        # Iteratively apply the CBB (box counting) method
        while tamanho != 1:
            # Run CBB using a function from the fractals_calculations module
            boxes_subgraphs = fractals_calculations.CBB(G, nb)
            lst.append((partition_id, nb, boxes_subgraphs))
            tamanho = len(boxes_subgraphs)
            nb = nb * 2

        # Save the resulting list to a CBB pickle file
        output_path = f'../../data/processed/coupling/community/{partition_id}/CBB.pickle'
        with open(output_path, 'wb') as arquivo:
            pickle.dump(lst, arquivo)