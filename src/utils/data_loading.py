import datetime
import numpy as np
import pandas as pd
import dask.dataframe as dd

# Convert a date string to a datetime.date object
def strToDate(strDate):
    try:
        return datetime.datetime.strptime(strDate, '%Y-%m-%d').date()
    except ValueError:
        # Handle the leap year case: change "02-29" to "02-28"
        if strDate[5:10] == '02-29':
            return datetime.datetime.strptime(strDate[0:5] + '02-28', '%Y-%m-%d').date()

# Load Louvain partition data from CSV
def load_louvain():
    _df = pd.read_csv('../../data/processed/coupling/louvain/glcc_communities.gz', sep=';', encoding='utf-8')
    # Ensure columns are of integer type
    _df = _df.astype({"partition": int, "index": int})
    return _df

# Load node information including metadata and landmark cases
def load_nodes_info():
    # Load index information for nodes
    index_nodes = pd.read_csv('../../data/processed/coupling/index_nodes.csv.gz', sep=';', encoding='utf-8')
    
    # Load case metadata and process decision dates
    metadata = pd.read_csv('../../data/raw/caselaw/metadata.csv.gz', sep=',', compression='gzip', encoding='utf-8')
    metadata['decision_date'] = metadata['decision_date_original'].apply(lambda x: strToDate((x + '-01-01')[0:10]))
    
    # Merge metadata with index nodes
    metadata = pd.merge(metadata, index_nodes, left_on='id', right_on='id_caselaw', how='inner')
    metadata['ano'] = metadata['decision_date'].apply(lambda x: x.strftime('%Y'))

    # Load landmark cases and merge with metadata
    land_wiki = pd.read_csv('../../data/processed/landmark_cases.csv.gz', sep=';', encoding='utf-8')
    metadata = pd.merge(metadata, land_wiki, left_on='id', right_on='id_case', how='outer')

    return metadata

# Load case data with Louvain partitions
def get_cases():
    cases = load_nodes_info()
    louvain = load_louvain()

    # Merge cases with Louvain partition information
    _df = pd.merge(cases, louvain, left_on='index', right_on='index', how='inner')

    # Fill missing landmark information
    _df['landmark_type'].fillna(-1, inplace=True)
    _df['landmark_subType'].fillna(-1, inplace=True)
    _df['landmark_type'] = _df['landmark_type'].apply(lambda x: '' if x == -1 else str(x))
    _df['landmark_subType'] = _df['landmark_subType'].apply(lambda x: '' if x == -1 else str(x))

    return _df

# Load edge list for coupling analysis as Dask DataFrame
def load_coupling():
    _ddf = dd.read_csv('../../data/processed/coupling/glcc.edgelist.gz', sep=' ')
    _ddf.columns = ['src', 'dst', 'weight']
    # Correct the usage of astype
    _ddf = _ddf.astype({'src': np.uint32, 'dst': np.uint32, 'weight': np.uint16})
    return _ddf
