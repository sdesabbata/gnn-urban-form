# Base libraries
import sys
import pandas as pd
# NetworkX
import networkx as nx
import osmnx as ox
from local_directories import *
# Multiprocessing
from multiprocessing import Pool


# Load Leciester's graph
leicester = ox.io.load_graphml(bulk_storage_directory + "/osmnx/raw_excluded/leicester-1864.graphml")


# Ego-graph node stats
leicester_node_stats_df = None
neighbourhood_min_nodes = 8
max_distance = 500

def get_node_stats(node):
    # Create the corresponding ego graph
    node_ego_graph = nx.generators.ego_graph(leicester, node, radius=max_distance, undirected=True, distance="length")
    # Only keep the sampled area if it has a minimum number of nodes
    if len(node_ego_graph.nodes()) > neighbourhood_min_nodes:
        # Calculate closeness centrality
        node_ego_graph_closeness_centrality = pd.DataFrame.from_dict(
            nx.closeness_centrality(leicester),
            orient='index',
            columns=['closeness_egograph'])
        # Calculate betweenness centrality
        node_ego_graph_betweenness_centrality = pd.DataFrame.from_dict(
            nx.betweenness_centrality(leicester),
            orient='index',
            columns=['betweenness_egograph'])
        # Join and return node's row
        node_stats = node_ego_graph_closeness_centrality.join(node_ego_graph_betweenness_centrality)
        node_stats["node_id"] = node_stats.index
        return node_stats.filter(items=[node], axis=0)
    else:
        return None


if __name__ == '__main__':
    node_id = int(sys.argv[1])
    leicester_node_stats_df = get_node_stats(node_id)
    # Save
    if leicester_node_stats_df is not None:
        leicester_node_stats_df.to_csv(this_repo_directory + "/storage/leicester-1864_node_egograph_stats_dist500/leicester-1864_node_stats_dist500__node_id_"+ str(node_id) +".csv", index=False)
        # to then be combined to leicester-1864_node_egograph_stats_dist500.csvleicester-1864_node_egograph_stats_dist500.csv
        # using
        # awk FNR==1 leicester-1864_node_egograph_stats_dist500/leicester-1864_node_stats_dist500__node_id_10023637.csv > leicester-1864_node_egograph_stats_dist500.csv
        # awk FNR!=1 leicester-1864_node_egograph_stats_dist500/* >> leicester-1864_node_egograph_stats_dist500.csv
    else:
        print("Node was excluded")
