import pandas as pd
import networkit as nk

# Read the edge list using EdgeListReader from Networkit
# The parameter commentPrefix='s' is used to ignore the header line that starts with the column name 'src'
edgeListReader = nk.graphio.EdgeListReader(separator=',', firstNode=0, commentPrefix='s', continuous=False, directed=False)

# Load the graph from the given edge list file
Glcc = edgeListReader.read('../../data/processed/coupling/glcc.edgelist.gz')

# Extract node map and save to DataFrame
dfNodeMap = pd.DataFrame(edgeListReader.getNodeMap().items(), columns=['index', 'index_nk'])
dfNodeMap = dfNodeMap.astype(int)

# Save node map to CSV file
dfNodeMap.to_csv('../../data/processed/coupling/louvain/glcc_nodemap.gz', sep=",", index=False, compression='gzip')

# Detect communities using the PLM algorithm (Louvain method)
plmCommunities = nk.community.detectCommunities(Glcc, algo=nk.community.PLM(Glcc, True))

# Save the detected communities to a DataFrame
dfPartition = pd.DataFrame(plmCommunities.getVector(), columns=['community'], dtype=int)
dfPartition.reset_index(level=0, inplace=True)
dfPartition.columns = ['index_nk', 'community']

# Merge partition data with node map
dfPartition = pd.merge(dfPartition, dfNodeMap, on='index_nk', how='inner')

# Save the resulting community information to a CSV file
dfPartition[['index', 'community']].to_csv('../../data/processed/coupling/louvain/glcc_communities.gz', sep=",", index=False, compression='gzip')