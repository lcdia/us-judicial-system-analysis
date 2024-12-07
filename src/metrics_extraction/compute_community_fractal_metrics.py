from tqdm import tqdm
import pandas as pd
import pickle
import networkx as nx
import numpy as np
import scipy.stats as st
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# Base directory for processed community data
base_path = '../../data/processed/coupling/community/'
cbb_files = []

# Iterating over directories in the base path
for subdir in os.listdir(base_path):
    subdir_path = os.path.join(base_path, subdir)
    if os.path.isdir(subdir_path):
        # Look for 'CBB.pickle' files
        file_path = os.path.join(subdir_path, 'CBB.pickle')
        if os.path.isfile(file_path):
            cbb_files.append(file_path)  # Append the full file path

df = pd.DataFrame()  # Initialize a DataFrame to store results

# Load and process each CBB file
for file in cbb_files:
    with open(file, 'rb') as f:
        lst = pickle.load(f)

    _df = pd.DataFrame(lst, columns=['partition', 'rb', 'boxes'])
    _df['len_boxes'] = _df.boxes.apply(lambda b: len(b))
    _df['number_nodes'] = _df.boxes.apply(lambda b: count_nodes(b))
    _df['sum_w_edges'] = _df.boxes.apply(lambda b: sum_w_edges(b))

    df = pd.concat([df, _df], ignore_index=True)

# Calculate metrics for each partition
lst = []
partitions = df.partition.unique()

for partition_id in partitions:
    b = df[df.partition == partition_id].number_nodes.max()
    w = df[df.partition == partition_id].sum_w_edges.max()
    x = df[df.partition == partition_id].rb.to_list()
    y = df[df.partition == partition_id].len_boxes.to_list()
    log_x = np.log(x)
    log_y = np.log(y)
    slope, intercept, r_value, p_value, std_err = st.linregress(log_x, log_y)
    lst.append((partition_id, b, w, intercept, r_value, p_value, std_err, abs(slope)))

df_metrics = pd.DataFrame(lst, columns=['partition', 'vertices', 'edges', 'intercept', 'r_value', 'p_value', 'std_err', 'slope'])
df_metrics['1_log_b'] = df_metrics.vertices.apply(lambda b: 1 / np.log(b))
df_metrics = df_metrics.sort_values(by='vertices', ascending=False).reset_index(drop=True)

df[df.partition.isin(top10.partition.tolist())][['partition', 'rb', 'len_boxes']].to_csv('figures/figure4_2.csv', index=False)

# Plot the results
plt.rcParams.update({'font.size': 12})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Scatter plot for box size vs. number of boxes
for i, row in top10.iterrows():
    partition_id = row.partition
    label = f'$C_{{{int(partition_id)}}}$'

    x = np.array(df[df.partition == partition_id].rb.to_list())
    y = np.array(df[df.partition == partition_id].len_boxes.to_list())

    logx = np.log10(x)
    logy = np.log10(y)

    slope, intercept, r_value, p_value, std_err = st.linregress(logx, logy)
    line = (slope * logx) + intercept

    ax1.scatter(logx, logy, label=label, s=80)
    ax1.plot(logx, line, '--', lw=1.5)

ax1.set_xlabel('$Box\ Size\ l_B$', fontsize=16)
ax1.set_ylabel('$Number\ of\ Boxes\ N_B$', fontsize=16)

# Scatter plot for vertices vs. edges
x = np.array(df_metrics["vertices"].to_list())
y = np.array(df_metrics["edges"].to_list())

logx = np.log10(x)
logy = np.log10(y)

slope, intercept, r_value, p_value, std_err = st.linregress(logx, logy)
line = (slope * logx) + intercept

ax2.scatter(logx, logy, color='black', s=64)
ax2.plot(logx, line, '--', color='black', lw=2)

ax2.set_xlabel('$Number\ of\ Vertices$', fontsize=16)
ax2.set_ylabel('$Total\ Weight\ of\ MST$', fontsize=16)

# Save the final plot
fig.savefig('../../figures/fractal_dimension.png', dpi=300, bbox_inches='tight')
plt.show()
