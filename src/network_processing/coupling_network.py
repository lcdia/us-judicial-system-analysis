import networkx as nx
import numpy as np 

# Load the graph from the given adjlist file
G = nx.read_adjlist('../../data/processed/citation/gwcc.adjlist.gz', delimiter=',', nodetype=int, create_using=nx.DiGraph())

# Convert the graph to a Scipy sparse matrix
a = nx.to_scipy_sparse_matrix(G)

# Change the data type of the sparse matrix to uint32
a = a.astype(np.uint32)

# Calculate the product of the sparse matrix and its transpose
a_result = a * a.transpose()

# Set diagonal elements to zero
a_result.setdiag(0)
a_result.eliminate_zeros()

# c = a_result.tocoo()
# df = pd.DataFrame({'source': c.row, 'target': c.col, 'weight': c.data})
# df = df.astype({"source": np.uint32, "target": np.uint32, "weight": np.int16})
# df.to_csv('../../data/processed/coupling/coupling.adjlist.gz', sep=",", index=False, compression='gzip')

# Create an undirected graph from the result sparse matrix
G = nx.from_scipy_sparse_matrix(a_result)
nx.write_weighted_edgelist(G, '../../data/processed/coupling/coupling.edgelist.gz', delimiter=',')

# Find the largest connected component of the graph
largest_lcc = max(nx.connected_components(G), key=len)

# Create a subgraph containing only the largest connected component
Glcc = G.subgraph(largest_lcc)

# # Save the nodes of the largest connected component to a CSV file
df = pd.DataFrame(list(Glcc.nodes()), columns=['id_caselaw'], dtype=int)
df.reset_index(level=0, inplace=True)
df.columns = ['index', 'id_caselaw']
df.to_csv('../../data/processed/coupling/index_nodes.csv.gz', sep=';', compression='gzip', index=False)

# Create a mapping to relabel nodes with a new index
mapping = dict(zip(df['id_caselaw'], df['index']))
Glcc = nx.relabel_nodes(Glcc, mapping)

# Save the largest connected component as an edge list
nx.write_weighted_edgelist(Glcc, '../../data/processed/coupling/glcc.edgelist.gz', delimiter=',')
