"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



############## Task 1

path = "../datasets/CA-HepTh.txt"

graph = nx.read_edgelist(path, delimiter='\t')

print(f"Le nombre d'arrête est {len(graph.edges)}")
print(f"Le nombre de noeuds est {len(graph.nodes)}")

############## Task 2

connex = nx.connected_components(graph)

len_graph = [(len(c),c) for c in sorted(connex,reverse=True)]

print(f"Nombre de composante connexe : {len(len_graph)}")

large_sub_graph_nodes = len_graph[0][1]

large_sub_graph = graph.subgraph(large_sub_graph_nodes).copy()

proportion_edges = len(large_sub_graph.edges)/len(graph.edges)
proportion_nodes = len(large_sub_graph.nodes)/len(graph.nodes)

print(f"Proportion de noeuds = {proportion_nodes}")
print(f"Proportion de edges = {proportion_edges}")

dict_nodes = dict(enumerate(graph.nodes))
dict_nodes_reverse = {v:k for k,v in dict_nodes.items() }

edges = graph.edges