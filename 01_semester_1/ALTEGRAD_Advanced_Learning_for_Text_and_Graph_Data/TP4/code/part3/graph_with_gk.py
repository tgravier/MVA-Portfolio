import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')


def load_file(filename):
    labels = []
    docs = []

    with open(filename, encoding='utf8', errors='ignore') as f:
        for line in f:
            content = line.split(':')
            labels.append(content[0])
            docs.append(content[1][:-1])

    return docs, labels


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()


def preprocessing(docs):
    preprocessed_docs = []
    n_sentences = 0
    stemmer = PorterStemmer()

    for doc in docs:
        clean_doc = clean_str(doc)
        preprocessed_docs.append([stemmer.stem(w) for w in clean_doc])

    return preprocessed_docs


def get_vocab(train_docs, test_docs):
    vocab = dict()

    for doc in train_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    for doc in test_docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


path_to_train_set = '../datasets/train_5500_coarse.label'
path_to_test_set = '../datasets/TREC_10_coarse.label'

# Read and pre-process train data
train_data, y_train = load_file(path_to_train_set)
train_data = preprocessing(train_data)

# Read and pre-process test data
test_data, y_test = load_file(path_to_test_set)
test_data = preprocessing(test_data)

# Extract vocabulary
vocab = get_vocab(train_data, test_data)
print("Vocabulary size: ", len(vocab))


# Task 11


def create_graphs_of_words(docs, vocab, window_size):
    graphs = list()
    vocab_reverse = {v:k for k,v in vocab.items()}
    for idx, doc in enumerate(docs):
        G = nx.Graph()
        for word in doc:
            if word in vocab:
               # G.add_node(vocab[word], word = word)
               G.add_node(vocab[word])

        for i in range(len(doc)):
            if doc[i] in vocab:
                node_i = vocab[doc[i]]
                for j in range(i+1,window_size+i):
                    if j < len(doc) and doc[j] in vocab:
                        node_j = vocab[doc[j]]
                        if not G.has_edge(node_i, node_j):
                            G.add_edge(node_i, node_j)



        graphs.append(G)

    return graphs


# Create graph-of-words representations
G_train_nx = create_graphs_of_words(train_data, vocab, 3)
G_test_nx = create_graphs_of_words(test_data, vocab, 3)


vocab_reverse = {v:k for k,v in vocab.items()}

for G in G_train_nx:
    
    nx.set_node_attributes(G, vocab_reverse, 'label')
    
for G in G_test_nx:
    nx.set_node_attributes(G, vocab_reverse, 'label')
    

print("Example of graph-of-words representation of document")
nx.draw_networkx(G_train_nx[3], with_labels=True)
plt.show()


from grakel.utils import graph_from_networkx
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath, RandomWalk
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# # Task 12

# # Transform networkx graphs to grakel representations
G_train =  graph_from_networkx(G_train_nx, node_labels_tag='label') # faire gaffe label nodes
G_test = graph_from_networkx(G_test_nx, node_labels_tag='label')

# # Initialize a Weisfeiler-Lehman subtree kernel
gk = WeisfeilerLehman(n_iter=5, base_graph_kernel = VertexHistogram)

# # Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)

# #Task 13

# # Train an SVM classifier and make predictions

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf_WL = SVC(kernel='precomputed')
clf_WL.fit(K_train, y_train)

y_pred = clf_WL.predict(K_test)

# # Evaluate the predictions
print("Accuracy pour Weisfeiler-Lehman subtree kernel:", accuracy_score(y_pred, y_test)) #0.938 pour WeisfeilerLehman


# #Task 14
# Tests avec d'autres kernel



## Shortest path ##
G_train =  graph_from_networkx(G_train_nx, node_labels_tag='label') # faire gaffe label nodes
G_test = graph_from_networkx(G_test_nx, node_labels_tag='label')

gk = ShortestPath(normalize=True)

# # Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)





clf_SP = SVC(kernel='precomputed')
clf_SP.fit(K_train, y_train)

y_pred = clf_WL.predict(K_test)

# # Evaluate the predictions
print("Accuracy pour Shortest Path Kernel:", accuracy_score(y_pred, y_test)) # accuracy 0.188 pour Shortest path

##

## Random Walk ##
G_train =  graph_from_networkx(G_train_nx, node_labels_tag='label') # faire gaffe label nodes
G_test = graph_from_networkx(G_test_nx, node_labels_tag='label')



gk = RandomWalk(normalize=True, lamda=0.5)



# # Construct kernel matrices
K_train = gk.fit_transform(G_train)
K_test = gk.transform(G_test)





clf_RW = SVC(kernel='precomputed')
clf_RW.fit(K_train, y_train)

y_pred = clf_WL.predict(K_test)

# # Evaluate the predictions
print("Accuracy pour Random Walk kernel:", accuracy_score(y_pred, y_test)) # accuracy 0.7 pour Random Walk

##


