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
# Ego-graph node stats
leicester_node_stats_df = None
neighbourhood_min_nodes = 8
max_distance = 500

def get_node_stats(node):
    this_node_stats_to_return = None
    # Create the corresponding ego graph
    node_ego_graph = nx.generators.ego_graph(leicester, node, radius=max_distance, undirected=True, distance="length")
    # Only keep the sampled area if it has a minimum number of nodes
    if len(node_ego_graph.nodes()) > neighbourhood_min_nodes:
        # Calculate closeness centrality
        node_ego_graph_closeness_centrality = pd.DataFrame.from_dict(
            nx.closeness_centrality(node_ego_graph),
            orient="index",
            columns=["closeness_egograph"])
        # Calculate betweenness centrality
        node_ego_graph_betweenness_centrality = pd.DataFrame.from_dict(
            nx.betweenness_centrality(node_ego_graph),
            orient="index",
            columns=["betweenness_egograph"])
        # Join and return node's row
        this_node_stats = node_ego_graph_closeness_centrality.join(node_ego_graph_betweenness_centrality)
        this_node_stats["node_id"] = this_node_stats.index
        this_node_stats_to_return = this_node_stats.filter(items=[node], axis=0)
    return this_node_stats_to_return


if __name__ == "__main__":

    # Leicester-wide node stats

    # Calculate closeness centrality
    leicester_closeness_centrality = pd.DataFrame.from_dict(
        nx.closeness_centrality(leicester),
        orient="index",
        columns=["closeness_networkwide"])
    leicester_closeness_centrality["node_id"] = leicester_closeness_centrality.index

    # Calculate betweenness centrality
    leicester_betweenness_centrality = pd.DataFrame.from_dict(
        nx.betweenness_centrality(leicester),
        orient="index",
        columns=["betweenness_networkwide"])
    leicester_betweenness_centrality["node_id"] = leicester_betweenness_centrality.index

    # Calculate ego graph stats

    # for node in leicester.nodes:
    #     print(node)
    #     node_stats = get_node_stats(node)
    #     if leicester_node_stats_df is None:
    #         leicester_node_stats_df = node_stats
    #     else:
    #         leicester_node_stats_df = pd.concat([leicester_node_stats_df, node_stats])

    p = Pool(processes=25)
    data = p.map(get_node_stats, [node for node in leicester.nodes])
    p.close()
    p.join()

    # Calculate base stats
    for node_stats in data:
        if node_stats is not None:
            if leicester_node_stats_df is None:
                leicester_node_stats_df = node_stats
            else:
                leicester_node_stats_df = pd.concat([leicester_node_stats_df, node_stats])

    leicester_node_stats_df = leicester_node_stats_df.merge(
        leicester_closeness_centrality, on="node_id"
        ).merge(
        leicester_betweenness_centrality, on="node_id"
        )

    # Save
    leicester_node_stats_df[
        ["node_id", "closeness_networkwide", "betweenness_networkwide", "closeness_egograph", "betweenness_egograph"]
    ].to_csv(this_repo_directory + "/data/leicester-1864_stats_node_centrality_with_egograph_dist500.csv", index=False)
