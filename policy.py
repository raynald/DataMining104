#!/usr/bin/env python2.7

import numpy as np
import numpy.random
from numpy.linalg import inv


alpha = 2
M = dict()
b = dict()
w = dict()
#z = dict()
z_t = 0
Artikel = {}
t = ""
x_t = "Nah"

# Evaluator will call this function and pass the article features.
# Check evaluator.py description for details.
def set_articles(art):
    global Artikel, M, b
    Artikel = art
    for art in Artikel:
        x = str(art)
        M[x] = np.identity(6)
        b[x] = np.zeros(6)

# This function will be called by the evaluator.
# Check task description for details.
def update(reward):
    global M, b
    if reward == -1:
        return
    y_t = reward
    #for art in Artikel:
    x = str(x_t)
    M[x] = np.add(M[x], np.dot(z_t, z_t.transpose()))
    b[x] = np.add(b[x], np.multiply(z_t, y_t))

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global t, z_t, M, b, x_t, w
    t = str(timestamp)
    z_t = np.array(user_features)
    Zaehler = 0
    Ausweis = 0
    for art in articles:
        Zaehler += 1
        x = str(art)

        if x_t == "Nah" or x == x_t:
            w[x] = np.dot(inv(M[x]), b[x])

        UCB = np.dot( w[x].transpose(), z_t ) + np.multiply(np.sqrt(np.dot(np.dot(z_t.transpose(), inv(M[x])), z_t)), alpha)
        if Zaehler == 1 or UCB > antwort:
            antwort = UCB
            Ausweis = art
    x_t = Ausweis
    return Ausweis
    #return numpy.random.choice(articles, size=1)
