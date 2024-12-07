import datetime
import pandas as pd
from collections import Counter

import plotly.express as px
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

# Add tools directory to sys.path to import data_loading module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import data_loading  # Assumes this module contains the get_cases function

# Load cases and sort by partition
cases = data_loading.get_cases().sort_values(by=['partition'], ascending=[False])

# Load US map data with state coordinates
us_map = pd.read_csv('../../data/raw/us_geodata/us_states_coordinates.csv', sep=',')
us_map.columns = ['state_abbreviation', 'latitude', 'longitude', 'state_name', 'jurisdiction_id']
us_map.loc[us_map.state_abbreviation == 'US', 'latitude'] = 46.558860
us_map.loc[us_map.state_abbreviation == 'US', 'longitude'] = -77.590579

# Merge cases with state data
cases = pd.merge(cases, us_map, on='jurisdiction_id')

# Group cases by partition and count their sizes
partition_sizes = cases.groupby(by='partition').size().reset_index(name='count').sort_values(by='count', ascending=False).reset_index(drop=True)

# Select the top 10 largest partitions for visualization
top_communities = partition_sizes[:10].reset_index()
top_communities['index'] = top_communities['index'] + 1
top_communities['name'] = top_communities['partition'].apply(lambda x: f'$c_{{{x}}}$')

# Configure plot settings
plt.rcParams.update({'font.size': 12})

# Scatter plot for community sizes
x = top_communities['index'].tolist()
y = top_communities['count'].tolist()
n = top_communities['name'].tolist()

fig, ax = plt.subplots()
ax.scatter(x, y, color='black')

# Annotate points on the scatter plot
for i, name in enumerate(n):
    ax.annotate(name, (x[i], y[i] + (150000 if i != 0 else -170000)), fontsize=12, ha='center')

# Set axis labels and ticks
ax.set_xticks(range(1, 11))
ax.set_yticks([500000, 1000000, 1500000, 2000000, 2500000, 3000000])
ax.set_yticklabels(['0.50', '1.00', '1.50', '2.00', '2.50', '3.00'])
plt.xlabel('$Ranking$', fontsize=14)
plt.ylabel('$s/10^6$', fontsize=14)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)

################################################################################
# Inset plot for size distribution
ins = ax.inset_axes([0.35, 0.35, 0.62, 0.62])

community_sizes = partition_sizes['count'].tolist()

# Configure logarithmic scale and bins
start_point = min(community_sizes)
end_point = max(community_sizes)
base = 2.0

ins.set_xscale('log')
ins.set_yscale('log')

bins = np.logspace(np.log(start_point)/np.log(base), np.log(end_point)/np.log(base), base=base, num=30)
n, bins, patches = ins.hist(community_sizes, density=True, bins=bins, fill=False, linewidth=0)

# Scatter points and linear regression on log-log scale
x = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
y = n
ins.scatter(x, y, s=28, color='black')

# Filter positive values for log scale
mask = (x > 0) & (y > 0)
logx = np.log10(x[mask])
logy = np.log10(y[mask])

# Linear regression
a, b, rval, pval, err = st.linregress(logx, logy)
ins.plot(10**logx, 10**(a * logx + b), color='black', ls='--', lw=2)
a_str = f'{abs(a):.2f}'
ins.text(0.70, 0.56, rf'$\alpha$={a_str}', transform=ins.transAxes, fontsize=14, va='top', ha='right', color='black')

# Configure inset axes
ins.set_xlabel('$s$')
ins.set_ylabel('$P(s)$')
ins.xaxis.set_tick_params(labelsize=8)
ins.yaxis.set_tick_params(labelsize=8)
ins.set_xticks([1, 10, 100, 1000, 10000, 100000, 1000000])
ins.set_yticks([0.1, 0.001, 0.00001, 0.0000001])
ins.set_ylim(0.0000001, 0.1)
ins.set_xlim(10, 1000000)
################################################################################

# Save the scatter plot
print(f'a={a}\n b={b}\n rval={rval}\n pval={pval}\n err={err}')
fig.savefig('../../figures/hist_size_community.png', dpi=fig.dpi, bbox_inches='tight')

# Group cases by partition and jurisdiction for the treemap
gpc = cases.groupby(by=['partition', 'jurisdiction_id', 'state_name']).size().reset_index(name='count').sort_values('count', ascending=False)
gpc['type'] = gpc['jurisdiction_id'].apply(lambda x: 'Federal' if x == 39 else 'Regional')
gpc['community_name'] = gpc['partition'].apply(lambda _id: f'<i>Community</i><sub>{_id}</sub>')
gpc['community_abbreviation'] = gpc['partition'].apply(lambda _id: f'<i>c</i><sub>{_id}</sub>')

# Create a treemap visualization
plt.rcParams.update({'font.size': 8})
fig = px.treemap(
    gpc,
    path=['community_abbreviation', 'state_name'],
    values='count',
    color='type',
    color_discrete_map={'Federal': '#1472af', 'Regional': '#e4b039'},
)

fig.update_traces(root_color="white")
fig.update_layout(
    width=1600,
    height=800,
    margin=dict(t=0, l=0, r=0, b=0)
)

# Adjust font size for treemap labels
fig.data[0]['textfont']['size'] = 18
fig.show()

# Save treemap as an image
fig.write_image('../../figures/treemap_community_by_jurisdiction.png')