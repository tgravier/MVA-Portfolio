"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2024
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    n_train = 100000
    max_train_card = 10
    
    ############## Task 1
    
    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)
    
    for i in range(n_train):
        card = np.random.randint(1, max_train_card + 1)
        X_train[i, max_train_card - card:] = np.random.randint(1, 11, card)
        y_train[i] = np.sum(X_train[i])

    return X_train, y_train




def create_test_dataset():
    
    ############## Task 2


    min_train_card = 5
    max_train_card = 100

    X_test = list()
    y_test = list()

    cards = np.linspace(min_train_card, max_train_card, 100 // 5, dtype=int)
    for card in cards: 
        
        X_test_card = np.zeros((10000, card))
        y_test_card = np.zeros(10000)
        for i in range(10000):
            
            X_test_card[i] = np.random.randint(1, 11, card)
            y_test_card[i] = np.sum(X_test_card[i])
        X_test.append(X_test_card)
        y_test.append(y_test_card)

    return X_test, y_test

