import networkx as nx
import pandas as pd
from tqdm import tqdm  # Adding tqdm to monitor progress in loops

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import data_loader

# Load the citation graph
G = nx.read_adjlist("../dataset/citation/adjlist/gcc.adjlist.gz", delimiter=',', nodetype=int, create_using=nx.DiGraph())

# Load case information and sort by partition and year
dfCases = get_cases().sort_values(by=['partition', 'year'], ascending=[False, True])

# Group cases by partition
partitions = dfCases.groupby(by=['partition'])

lst = []
total = len(partitions)

# Iterate over each partition to analyze nodes and citations
for idPartition, partition in tqdm(partitions, total=total):
    # Group cases by year within each partition
    years = partition.groupby(by=['year'])
    for year, yearCases in years:
        listNodes = yearCases['id'].tolist()
        
        # Calculate in-degree for all nodes in the year
        lst_in_degree = list(G.in_degree(listNodes))
        valueInDegree = sum(j for _, j in lst_in_degree)

        # Calculate out-degree for all nodes in the year
        lst_out_degree = list(G.out_degree(listNodes))
        valueOutDegree = sum(j for _, j in lst_out_degree)

        # Append metrics for each partition and year
        lst.append([idPartition, year, len(listNodes), valueInDegree, valueOutDegree])
    total -= 1

# Create DataFrame from the list and save yearly citation metrics
dfYearCitation = pd.DataFrame(lst, columns=['partition', 'year', 'qtd_nodes', 'inDegree', 'outDegree'])

# Group by partition to continue further analysis
partitions = dfYearCitation.groupby(by=['partition'])

# Load judges data and group by partition and judge ID
judges = load_judges().groupby(by=['partition', 'id_people']).agg(total=('partition', 'count'), min_year=('year', min)).reset_index()
# Aggregate judges data by partition to get judge counts and earliest year
judges_agg = judges.groupby(by=['partition']).agg(qtd_judges=('partition', 'count'), year=('min_year', min)).reset_index()

lst = []
total = len(partitions)

# Iterate over partitions to collect data for alometry analysis
for idPartition, partition in tqdm(partitions, total=total):
    # Determine the range of years in the partition
    minYear = partition['year'].min()
    maxYear = partition['year'].max()
    
    # Determine the year of highest in-degree (most citations)
    year = partition[partition['inDegree'] == partition['inDegree'].max()].iloc[0]['year']
    judgesInPartition = judges[judges['partition'] == idPartition]
    hasJudgesInPartition = len(judgesInPartition) > 0

    # Total cases, citations, and judges in the partition
    qtdCasosNaParticao = partition['qtd_nodes'].sum()
    qtdCitacoesNaParticao = partition['inDegree'].sum()
    qtdJudgesNaParticao = judgesInPartition['total'].sum() if hasJudgesInPartition else 0

    # Accumulated metrics up to the year of highest citations
    qtdCasosAteAnoMaiorCitacao = partition[partition['year'] <= year]['qtd_nodes'].sum()
    qtdCitacoesAteAnoMaiorCitacao = partition[partition['year'] <= year]['outDegree'].sum()
    qtdJudgesAteAnoMaiorParticao = judgesInPartition[judgesInPartition['min_year'] <= year]['total'].sum() if hasJudgesInPartition else 0

    # Append metrics for each partition
    lst.append([idPartition, minYear, maxYear, qtdCasosNaParticao, qtdCitacoesNaParticao, qtdJudgesNaParticao, qtdCasosAteAnoMaiorCitacao, qtdCitacoesAteAnoMaiorCitacao, qtdJudgesAteAnoMaiorParticao])
    
    total -= 1

# Create DataFrame and save alometry data for further analysis
dfDataToAlometry = pd.DataFrame(lst, columns=['partition', 'inicio', 'ultimo', 'casosParticao', 'citacoesParticao', 'juizesParticao', 'casosAtePico', 'citacoesAtePico', 'juizesAtePico'])
dfDataToAlometry.to_csv('../../data/processed/coupling/data_to_alometry.csv.gz', sep=";", compression='gzip', index=False)