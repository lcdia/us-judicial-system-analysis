import sys
import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import networkx as nx
import warnings
import scipy.stats as st
import matplotlib.pyplot as plt

import dask.dataframe as dd

# Add tools directory to sys.path to import data_loading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import data_loading

# Load case information using a function from the data_loading module
cases = get_cases()[['id', 'partition', 'frontend_url', 'name_abbreviation', 'jurisdiction_id', 'court_id', 'idx', 'landmark_type', 'landmark_subType']]
cases.columns = ['id', 'partition', 'frontend_url', 'name_abbreviation', 'jurisdiction_id', 'court_id', 'node', 'node2', 'landmark_type', 'landmark_subType']

# Convert columns to integer type
cases['node'] = cases['node'].apply(lambda x: int(x))
cases['id'] = cases['id'].apply(lambda x: int(x))
cases['jurisdiction_id'] = cases['jurisdiction_id'].apply(lambda x: int(x))
cases['court_id'] = cases['court_id'].apply(lambda x: int(x))

# Group cases by partition and sort by size
partitions = cases.groupby(by='partition').size().reset_index(name='quantidade').sort_values(by='quantidade', ascending=False).reset_index(drop=True)

# Iterate over partitions and create subgraphs, find maximum spanning trees (MST), and save them
for _, row in (pbar := tqdm(partitions.iterrows(), total=len(partitions))):
    partition_id = row['partition']

    # Skip if MST file already exists
    if not os.path.exists(f'../../data/processed/coupling/community/{partition_id}/mst.pickle'):
        pbar.set_postfix_str(f'getting nodes {partition_id}')
        
        # Get nodes belonging to the current partition
        nodes = cases[cases['partition'] == partition_id]['node'].tolist()

        pbar.set_postfix_str(f'filtering edges {partition_id}')
        
        # Filter edges belonging to the current partition nodes using Dask
        ddf = dd.read_csv('../../data/processed/coupling/glcc.edgelist.gz', sep=';', compression='gzip', names=['source', 'target', 'weight'], header=None)
        dfSubGraph = ddf[ddf['source'].isin(nodes) & ddf['target'].isin(nodes)]
        dfSubGraph = dfSubGraph.compute()  # Trigger computation to convert Dask DataFrame to Pandas DataFrame

        pbar.set_postfix_str(f'creating graph {partition_id}')
        
        # Create NetworkX graph from filtered edges
        G = nx.from_pandas_edgelist(dfSubGraph, "source", "target", ["weight"])
        pickle.dump(dfSubGraph, open(f'../../data/processed/coupling/community/{partition_id}/subgraph.pickle', 'wb'))
        del dfSubGraph

        pbar.set_postfix_str(f'getting Gcc {partition_id}')
        
        # Get the largest connected component (GCC)
        Glcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(Glcc[0])
        del G
        del Glcc

        pbar.set_postfix_str(f'getting MST {partition_id}')
        
        # Calculate the maximum spanning tree (MST) of the largest connected component
        T = nx.maximum_spanning_tree(G0)
        del G0

        pbar.set_postfix_str(f'saving {partition_id}')
        
        # Save the MST to a pickle file
        os.makedirs(f'../../data/processed/coupling/community/{partition_id}', exist_ok=True)
        pickle.dump(T, open(f'../../data/processed/coupling/community/{partition_id}/mst.pickle', 'wb'))

        # Clean up to free memory
        gc.collect()
