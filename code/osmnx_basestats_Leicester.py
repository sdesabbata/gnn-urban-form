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

leicester_basestats_df = None
neighbourhood_min_nodes = 8
max_distance = 1000


def get_base_stats(node):
    # Create the corresponding ego graph
    node_ego_graph = nx.generators.ego_graph(leicester, node, radius=max_distance, undirected=True, distance="length")
    # Only keep the sampled area if it has a minimum number of nodes
    if len(node_ego_graph.nodes()) > neighbourhood_min_nodes:
        node_base_stats = ox.stats.basic_stats(node_ego_graph)
        node_base_stats["node_id"] = node
        return node_base_stats

if __name__ == '__main__':
    p = Pool(processes=25)
    data = p.map(get_base_stats, [node for node in leicester.nodes])
    p.close()
    p.join()

    # Calculate base stats
    for basestats in data:
        if leicester_basestats_df is None:
            leicester_basestats_df = pd.DataFrame.from_dict([basestats])
        else:
            leicester_basestats_df = pd.concat([leicester_basestats_df, pd.DataFrame.from_dict([basestats])])

    # Save
    leicester_basestats_df.to_csv(this_repo_directory + "/data/leicester-1864_basestats.csv", index=False)
