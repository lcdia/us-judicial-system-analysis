import sys
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import scipy.stats as st
import dask.dataframe as dd
import matplotlib.pyplot as plt

# Add tools directory to sys.path to import data_loading module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import data_loading  # Assumes this module contains the get_cases function

# Load case information
cases = data_loading.get_cases()[[
    'id', 'partition', 'frontend_url', 'name_abbreviation', 'jurisdiction_id', 'court_id', 'idx', 'landmark_type', 'landmark_subType'
]]
cases.columns = ['id', 'partition', 'frontend_url', 'name_abbreviation', 'jurisdiction_id', 'court_id', 'node', 'node2', 'landmark_type', 'landmark_subType']

# Convert columns to integer type
cases['node'] = cases['node'].astype(int)
cases['id'] = cases['id'].astype(int)
cases['jurisdiction_id'] = cases['jurisdiction_id'].astype(int)
cases['court_id'] = cases['court_id'].astype(int)

# Load edges using Dask
ddf = dd.read_csv(
    '../../data/processed/coupling/glcc.edgelist.gz',
    sep=';', compression='gzip', names=['source', 'target', 'weight'], header=None
)
ddf.columns = ['source', 'target', 'weight']
ddf = ddf.astype({'source': np.uint32, 'target': np.uint32, 'weight': np.uint16})

# Compute degree for sources and targets
df_src = ddf.groupby(by='source')['weight'].sum().compute().reset_index(name='total')
df_dst = ddf.groupby(by='target')['weight'].sum().compute().reset_index(name='total')

# Combine source and target degrees
df_src.columns = ['node', 'total']
df_dst.columns = ['node', 'total']
_df = pd.concat([df_src, df_dst])

# Calculate total degree and merge with case data
df_degrees = _df.groupby(by='node')['total'].sum().reset_index(name='degree').sort_values(by='degree', ascending=False).reset_index(drop=True)
df_merge = pd.merge(cases, df_degrees, on='node', how='inner').sort_values(by='degree', ascending=False).reset_index(drop=True)

# Select the top 10 nodes for visualization
nodes = df_merge[:10][[
    'partition', 'name_abbreviation', 'degree', 'node', 'landmark_type', 'landmark_subType'
]].reset_index()
nodes['index'] = nodes['index'] + 1

# Configure plot settings
plt.rcParams.update({'font.size': 12})

# Scatter plot data
x = nodes['index'].tolist()
y = nodes['degree'].tolist()
n = nodes['name_abbreviation'].tolist()

fig, ax = plt.subplots()
ax.scatter(x, y, color='black')

# Annotate specific points on the scatter plot
ax.annotate('Tapia\n    v.\nCity of Albuquerque', (x[0]-0.2, y[0] - 13000), fontsize=8)
ax.annotate(n[1].replace(' v. ', '\nv.\n'), (x[1], y[1] - 13000), fontsize=8, ha='center')
ax.annotate('Casanova\nv.\nCity of\nBrookshire', (x[2], y[2] + 4000), fontsize=8, ha='center')
ax.annotate(n[3].replace(' v. ', '\nv.\n'), (x[3], y[3] - 13000), fontsize=8, ha='center')
ax.annotate(n[4].replace(' v. ', '\nv.\n'), (x[4], y[4] + 4000), fontsize=8, ha='center')
ax.annotate(n[5], (x[5], y[5]-5000), fontsize=8, ha='center')
ax.annotate('Kerns\nv.\nBoard of\nCommissioners', (x[6], y[6] + 4000), fontsize=8, ha='center')
ax.annotate(n[7], (x[7], y[7]-5000), fontsize=8, ha='center')
ax.annotate('Bluitt\nv.\nHouston\nIndependent\nSchool\nDistrict', (x[8], y[8] + 4400), fontsize=8, ha='center')
ax.annotate('Honken\nv.\nU.S.', (x[9], y[9] + 2000), fontsize=8, ha='center')

# Configure axes
ax.set_xticks(range(1, 11))
ax.set_yticks([360000, 380000, 400000, 420000, 440000, 460000])
ax.set_yticklabels(['3.6', '3.8', '4.0', '4.2', '4.4', '4.6'])
plt.xlabel('$Ranking$', fontsize=14)
plt.ylabel('$k/10^5$', fontsize=14)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)

################################################################################
# Inset histogram plot for degree distribution
ins = ax.inset_axes([0.35, 0.35, 0.62, 0.62])

k = df_merge.degree.tolist()

# Set logarithmic scale and histogram bins
start_point = min(k)
end_point = max(k)
base = 2.0

ins.set_xscale('log')
ins.set_yscale('log')

ls = np.logspace(np.log(start_point)/np.log(base), np.log(end_point)/np.log(base), base=base, num=30)
n, bins, patches = ins.hist(k, density=True, bins=ls, fill=False, linewidth=0)

# Scatter points and linear regression
x = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
y = n
ins.scatter(x, y, s=28, color='black')

logx = np.log10(x)
logy = np.log10(y)
mask = (logx >= 2) & (logx <= 5)
logx = logx[mask]
logy = logy[mask]

a, b, rval, pval, err = st.linregress(logx, logy)
ins.plot(10**logx, 2.5 * 10**(a*logx + b), color='black', ls='--', lw=2)
a_str = '{:.2f}'.format(abs(a))
ins.text(0.80, 0.78, r'$\alpha$=' + a_str, transform=ins.transAxes, fontsize=14, va='top', ha='right', color='black')

# Configure inset axes
ins.set_xlabel('$k$')
ins.set_ylabel('$P(k)$')
ins.xaxis.set_tick_params(labelsize=8)
ins.yaxis.set_tick_params(labelsize=8)
ins.set_xticks([1, 10, 100, 1000, 10000, 100000, 1000000])
ins.set_yticks([0.01, 0.0001, 0.000001, 0.00000001])
ins.set_ylim(0.00000001, 0.01)
ins.set_xlim(10, 1000000)
################################################################################
print(f'a={a}\n b={b}\n rval={rval}\n pval={pval}\n err={err}')
fig.savefig('../../figures/hist_coupling.png', dpi=fig.dpi, bbox_inches='tight')
