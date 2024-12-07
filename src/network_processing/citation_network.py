import networkx as nx
import pandas as pd
import datetime

# Function to convert string to date
def strToDate(strDate):
    try:
        return datetime.datetime.strptime(strDate, '%Y-%m-%d').date()
    except ValueError:
        # Handle leap year scenario when the date is February 29th (convert to February 28th)
        if (strDate[5:10] == '02-29'):
            return datetime.datetime.strptime(strDate[0:5] + '02-28', '%Y-%m-%d').date()

# Load the citation network from the given adjlist file
G = nx.read_adjlist('../../data/raw/caselaw/citation/citations.adjlist.gz', delimiter=',', nodetype=int, create_using=nx.DiGraph())

# Load metadata from CSV
metadata = pd.read_csv('../../data/raw/caselaw/metadata.csv.gz', sep=',', encoding='utf-8', compression='gzip')

# Convert 'decision_date_original' to datetime format and assign to 'decision_date'
metadata['decision_date'] = metadata['decision_date_original'].apply(lambda x: strToDate((x + '-01-01')[0:10]))
metadata = metadata[['id', 'decision_date']]

# Set 'decision_date' attribute for each node in the network
attrs = metadata.set_index('id').to_dict('index')
nx.set_node_attributes(G, attrs)

# Filter out edges that create cycles (i.e., if the citation date of the target is before the source)
lstNodes = []
lstEdges = []
for edge in G.edges:
    if G.nodes[edge[0]]['decision_date'] <= G.nodes[edge[1]]['decision_date']:
        lstEdges.append(edge)
        lstNodes.append(edge[0])
        lstNodes.append(edge[1])

# Remove the filtered edges from the graph
G.remove_edges_from(lstEdges)

# Write the updated graph to a new adjlist file (without cycles)
nx.write_adjlist(G, '../../data/processed/citation/citations_network_without_cycle.adjlist.gz', delimiter=',')

# Find the largest weakly connected component in the network
largest_wcc = max(nx.weakly_connected_components(G), key=len)

# Extract the subgraph containing the largest weakly connected component
Gwcc = G.subgraph(largest_wcc)

# Save the relabeled graph
nx.write_adjlist(Gwcc, '../../data/processed/citation/gwcc.adjlist.gz', delimiter=',')
nx.write_edgelist(Gwcc, '../../data/processed/citation/gwcc.edgelist.gz', data=False)
