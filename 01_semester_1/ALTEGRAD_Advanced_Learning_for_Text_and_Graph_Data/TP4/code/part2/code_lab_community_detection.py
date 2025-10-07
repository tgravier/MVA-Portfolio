"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans




############## Task 3
# Perform spectral clustering to partition graph G into k clusters






def adjency_matrix(G):
    # Créer un dictionnaire d'index pour chaque noeud
    dict_nodes = dict(enumerate(G.nodes))
    dict_nodes_reverse = {v: k for k, v in dict_nodes.items()}  # Ne pas forcer le type int
    
    # Initialiser la matrice d'adjacence avec des zéros
    A = np.zeros((len(G.nodes), len(G.nodes)))
    
    # Itérer sur les arêtes et mettre à jour la matrice
    for edge in G.edges:
        node1, node2 = edge
        i, j = dict_nodes_reverse[node1], dict_nodes_reverse[node2]
        A[i, j] = 1
        A[j, i] = 1  # Matrice symétrique pour un graphe non orienté
    
    return A

def spectral_clustering(G, k):
    
    ##################
    # your code here #
    ##################
    
    
    
    
    A = adjency_matrix(G)
    D_inv = np.diag(1/np.sum(A, axis = 1))
    
    
    L_rw = np.eye(len(G.nodes)) - D_inv @ A

    
    eigenvalues, eigenvectors = eigs(L_rw, k=k, which='SM')
    
    # K-means sur les vecteurs propres
    eigenvectors = eigenvectors.real  # Prendre la partie réelle si nécessaire
    kmeans = KMeans(n_clusters=k).fit(eigenvectors)
    labels = kmeans.labels_
    
    return dict(zip(G.nodes,labels))
    

    
    



############## Task 4


path = "../datasets/CA-HepTh.txt"

graph = nx.read_edgelist(path, delimiter='\t')




graph = nx.read_edgelist(path, delimiter='\t')
connex = nx.connected_components(graph)
len_graph = [(len(c),c) for c in sorted(connex,reverse=True)]
large_sub_graph_nodes = len_graph[0][1]
large_sub_graph = graph.subgraph(large_sub_graph_nodes).copy()

clustering = spectral_clustering(large_sub_graph, 50)

############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    A = adjency_matrix(G)
    D = np.sum(A,axis =0)

    
    
    nc = len(np.unique(list(clustering.values())))
    
    lc = np.zeros(nc)
    for edge in G.edges:
        if clustering[edge[0]] == clustering[edge[1]]:
            
            lc[clustering[edge[0]]] +=1
    
        
    dc = np.zeros(nc)
    degree = G.degree
    
    
    for node in G.nodes:
        
        dc[clustering[node]] += degree[node]
    
    m = len(G.edges)
    Q = np.sum(lc / m - (dc / (2 * m)) ** 2)

        
        
        
        
    
    return Q


############## Task 6

##################
random_clustering = {k:randint(0, 49) for k in large_sub_graph.nodes}


print(f"La modularité de la plus grande composante connexe avec le clustering kmeans est {modularity(large_sub_graph, clustering)}")
print(f"La modularité de la plus grande composante connexe avec un clustering aléatoire est {modularity(large_sub_graph, random_clustering)}")

##################







