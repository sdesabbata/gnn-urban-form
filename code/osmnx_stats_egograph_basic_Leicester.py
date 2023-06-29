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

leicester_basic_stats_df = None
neighbourhood_min_nodes = 8
max_distance = 500


def get_basic_stats(node):
    # Create the corresponding ego graph
    node_ego_graph = nx.generators.ego_graph(leicester, node, radius=max_distance, undirected=True, distance="length")
    # Only keep the sampled area if it has a minimum number of nodes
    if len(node_ego_graph.nodes()) > neighbourhood_min_nodes:
        node_basic_stats = ox.stats.basic_stats(node_ego_graph)
        node_basic_stats["node_id"] = node
        return node_basic_stats

if __name__ == '__main__':
    p = Pool(processes=25)
    data = p.map(get_basic_stats, [node for node in leicester.nodes])
    p.close()
    p.join()

    # Calculate base stats
    for basic_stats in data:
        if leicester_basic_stats_df is None:
            leicester_basic_stats_df = pd.DataFrame.from_dict([basic_stats])
        else:
            leicester_basic_stats_df = pd.concat([leicester_basic_stats_df, pd.DataFrame.from_dict([basic_stats])])

    # Save
    leicester_basic_stats_df.to_csv(this_repo_directory + "/data/leicester-1864_stats_egograph_basic_dist500.csv", index=False)
