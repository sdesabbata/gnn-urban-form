# Base libraries
import pandas as pd
# NetworkX
import networkx as nx
import osmnx as ox
from local_directories import *
# Multiprocessing
from multiprocessing import Pool


# Load Leciester's graph
leicester = ox.io.load_graphml(bulk_storage_directory + "/osmnx/raw_excluded/leicester-1864.graphml")

# Leicester-wide node stats

# Calculate closeness centrality
leicester_closeness_centrality = pd.DataFrame.from_dict(
    nx.closeness_centrality(leicester),
    orient='index',
    columns=['closeness_networkwide'])
leicester_closeness_centrality['node_id'] = leicester_closeness_centrality.index

# Calculate betweenness centrality
leicester_betweenness_centrality = pd.DataFrame.from_dict(
    nx.betweenness_centrality(leicester),
    orient='index',
    columns=['betweenness_networkwide'])
leicester_betweenness_centrality['node_id'] = leicester_betweenness_centrality.index

leicester_node_stats_df = leicester_closeness_centrality.merge(
    leicester_betweenness_centrality, on="node_id"
    )

# Save
leicester_node_stats_df.to_csv(this_repo_directory + "/data/leicester-1864_node_stats_dist500.csv", index=False)
