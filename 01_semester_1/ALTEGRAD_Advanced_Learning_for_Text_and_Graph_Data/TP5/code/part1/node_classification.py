"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score, classification_report


from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network
colors = ['red' if label == 1 else 'green' for label in y]

nx.draw_networkx(G, node_color =colors)


##################

##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


# Initialisation et entraînement du modèle
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = logistic_model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))



############## Task 8
# Generates spectral embeddings
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
    
    
    
    return  eigenvectors

dim_eigen = 2 

embeddings_spectral = spectral_clustering(G, dim_eigen)

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings_spectral[idx_train,:]
X_test = embeddings_spectral[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]

# Initialisation et entraînement du modèle
logistic_model_spectral = LogisticRegression(random_state=42, max_iter=1000)
logistic_model_spectral.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = logistic_model_spectral.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy spectral : {accuracy:.4f}")
print("Classification Report Spectral:")
print(classification_report(y_test, y_pred))
