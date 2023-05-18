# Base libraries
import os
from os.path import isfile, join
from datetime import datetime
import sys
import math
import random
# NetworkX
import networkx as nx
import osmnx as ox
# Torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
# GNN models
from gnnuf_models import *

class OSMnxDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None, pre_transform=None, pre_filter=None,
                 neighbourhood_sample=0.01, neighbourhood_min_nodes=1, max_distance=1000
                 ):
        self.neighbourhood_sample = neighbourhood_sample
        self.neighbourhood_min_nodes = neighbourhood_min_nodes
        self.max_distance = max_distance
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["aberdeen-1809.graphml", "ashford-1995.graphml", "atherton-1776.graphml", "aylesbury-1889.graphml",
                "ayr-1703.graphml", "bangor-1678.graphml", "basildon-1978.graphml", "basingstoke-1869.graphml",
                "bath-1782.graphml", "bedford-1923.graphml", "belfast-1667.graphml", "birmingham-1800.graphml",
                "birtley-1836.graphml", "blackburn-1778.graphml", "blackpool-1755.graphml", "blackwater-1891.graphml",
                "bognor_regis-1896.graphml", "bournemouth-1806.graphml", "bracknell-1893.graphml",
                "bridgend-1728.graphml", "brighton-1929.graphml", "bristol-1771.graphml", "burnley-1794.graphml",
                "burton_on_trent-1831.graphml", "bury-1792.graphml", "cambridge-1965.graphml",
                "canterbury-2000.graphml", "cardiff-1741.graphml", "carlisle-1761.graphml", "chatham-1981.graphml",
                "chelmsford-1979.graphml", "cheltenham-1803.graphml", "chester-1760.graphml",
                "chesterfield-1845.graphml", "coatbridge-1723.graphml", "colchester-1993.graphml",
                "corby-1900.graphml", "coventry-1834.graphml", "crawley-1945.graphml", "crewe-1781.graphml",
                "cudworth-1841.graphml", "darlington-1837.graphml", "derby-1839.graphml", "doncaster-1867.graphml",
                "dundee-1763.graphml", "dunfermline-1742.graphml", "east_kilbride-1716.graphml",
                "eastbourne-1973.graphml", "eastleigh-1848.graphml", "edinburgh-1749.graphml",
                "ellesmere_port-1759.graphml", "exeter-1729.graphml", "fazeley-1828.graphml",
                "folkestone-2004.graphml", "glasgow-1709.graphml", "gloucester-1790.graphml",
                "great_wyrley-1810.graphml", "great_yarmouth-2032.graphml", "greenock-1699.graphml",
                "grimsby-1951.graphml", "guildford-1914.graphml", "hamilton-1721.graphml", "harlow-1960.graphml",
                "harrogate-1840.graphml", "hartlepool-1865.graphml", "hemel_hempstead-1924.graphml",
                "hereford-1770.graphml", "high_wycombe-1892.graphml", "huddersfield-1815.graphml", "hull-1930.graphml",
                "ipswich-2003.graphml", "keighley-1813.graphml", "kettering-1899.graphml",
                "kidderminster-1793.graphml", "lancaster-1764.graphml", "larbert-1731.graphml", "leeds-1819.graphml",
                "lincoln-1915.graphml", "littlehampton-1919.graphml", "liverpool-1751.graphml",
                "livingston-1739.graphml", "london-1912.graphml", "londonderry_derry-1636.graphml",
                "loughborough-1862.graphml", "lowestoft-2033.graphml", "luton-1920.graphml", "maidenhead-1895.graphml",
                "maidstone-1983.graphml", "manchester-1780.graphml", "mansfield-1860.graphml", "margate-2012.graphml",
                "middlesbrough-1856.graphml", "milton_keynes-1887.graphml", "neath-1718.graphml",
                "newcastle_upon_tyne-1830.graphml", "newport-1752.graphml", "northampton-1880.graphml",
                "norwich-2005.graphml", "nottingham-1857.graphml", "oxford-1858.graphml", "peterborough-1939.graphml",
                "plymouth-1707.graphml", "pontefract-1851.graphml", "portsmouth-1861.graphml", "preston-1773.graphml",
                "reading-1877.graphml", "redditch-1811.graphml", "redhill-1948.graphml", "rishton-1786.graphml",
                "royal_tunbridge_wells-1971.graphml", "rugby-1859.graphml", "runcorn-1772.graphml",
                "scunthorpe-1905.graphml", "sheffield-1838.graphml", "shrewsbury-1767.graphml", "slough-1906.graphml",
                "southampton-1835.graphml", "southend_on_sea-1984.graphml", "southport-1756.graphml",
                "st_albans-1936.graphml", "st_leonards-1980.graphml", "stafford-1805.graphml", "stevenage-1946.graphml",
                "stoke_on_trent-1795.graphml", "sunderland-1849.graphml", "swansea-1713.graphml",
                "swindon-1816.graphml", "taunton-1746.graphml", "telford-1777.graphml", "torquay-1726.graphml",
                "warrington-1775.graphml", "warwick-1833.graphml", "wath_upon_dearne-1847.graphml",
                "weston_super_mare-1753.graphml", "wishaw-1722.graphml", "worcester-1796.graphml", "york-1874.graphml"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        print(r"Haven't got time to implement this ¯\_(ツ)_/¯ ")
        print(r"Please download the data manually")

    def process(self):
        graphml_file_names = [
            join(self.root + "/raw", f)
            for f in os.listdir(self.root + "/raw")
            if f[-8:] == ".graphml"
            if isfile(join(self.root + "/raw", f))]

        neighbourhoods_list = []

        # print(graphml_file_names)
        # sys.stdout.write(graphml_file_names)

        for graphml_file_name in graphml_file_names:
            # Log files being processed
            # print(f"Processing: {graphml_file_name}")
            # sys.stdout.write(f"Processing: {graphml_file_name}")
            f_log = open(join(self.root + "/log", graphml_file_name.split("/")[-1].replace(".graphml", ".txt")), "w")
            f_log.write("Started processing..." + "\n")
            f_log_timestamp = datetime.now().strftime('%Y%m%d_%H:%M:%S - ')
            f_log.write(f_log_timestamp + "\n")

            # Load whole street network
            G = ox.io.load_graphml(graphml_file_name)
            # Sample street nodes
            sample_nodes = random.sample(list(G.nodes), math.ceil(len(G.nodes) * self.neighbourhood_sample))

            # Create Pytorch Geometric graph for street networks around sampled street nodes
            for sampled_node in sample_nodes:
                f_log.write(str(sampled_node) + "\n")

                # Ego graph including street network within distance
                sampled_ego_graph = nx.generators.ego_graph(
                    G, sampled_node,
                    radius=self.max_distance,
                    undirected=True, distance="length"
                )

                # Only keep the sampled area if it has a minimum number of nodes
                if len(sampled_ego_graph.nodes) > self.neighbourhood_min_nodes:
                    neighbourhoods_list.append(self.to_pyg_linegraph(sampled_ego_graph))

            f_log_timestamp = datetime.now().strftime('%Y%m%d_%H:%M:%S - ')
            f_log.write(f_log_timestamp + "\n")
            f_log.write("done." + "\n")
            f_log.close()

        self.data, self.slices = self.collate(neighbourhoods_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    # Convert linegraph to Pytorch Geometric linegraph
    def to_pyg_linegraph(self, ego_graph):

            # Extract lengths and slopes
            seg_edges_length = nx.get_edge_attributes(ego_graph, "length")
            # seg_edges_grade_abs = nx.get_edge_attributes(ego_graph, "grade_abs")

            # Create line graph
            seg_linegraph = nx.line_graph(ego_graph)
            # Add street lenghts and slopes as attribute x
            for seglg_node in seg_linegraph.nodes():
                seg_edge_length = seg_edges_length[(seglg_node[0], seglg_node[1], 0)]
                # seg_edge_grade_abs = seg_edges_grade_abs[(seglg_node[0], seglg_node[1], 0)]
                seg_linegraph.nodes[seglg_node]["x"] = (seg_edge_length / self.max_distance) # Normalisation
                # [, ((seg_edge_grade_abs /0.05) if seg_edge_grade_abs<0.05 else 1.0)]
            del seglg_node

            # Return Pytorch Geometric graph
            return from_networkx(seg_linegraph)
