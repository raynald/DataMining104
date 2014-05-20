#!/usr/bin/env python2.7

import numpy as np
import numpy.random
from numpy.linalg import inv


alpha = 2
M = dict()
b = dict()
z = dict()
Artikel = {}
t = ""
x_t = ""

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
    y_t = reward
    #for art in Artikel:
    x = str(x_t)
    M[x] = np.add(M[x], np.dot(z[t], z[t].transpose()))
    b[x] = np.add(b[x], np.multiply(z[t], y_t))

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    global t, z, M, b, x_t
    t = str(timestamp)
    z[t] = np.array(user_features)
    Zaehler = 0
    Ausweis = 0
    for art in articles:
        Zaehler += 1
        x = str(art)

        w_x = np.dot(inv(M[x]), b[x])

        UCB = np.dot( w_x.transpose(), z[t] ) + np.multiply(np.sqrt(np.dot(np.dot(z[t].transpose(), inv(M[x])), z[t])), alpha)
        if Zaehler == 1 or UCB > antwort:
            antwort = UCB
            Ausweis = art
    x_t = Ausweis
    return Ausweis
    #return numpy.random.choice(articles, size=1)
