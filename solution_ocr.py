# -*- coding: utf-8 -*-

#####
# Étienne Boutet - boue2327
# Raphael Valois - valr2802
###

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

import numpy as np


def logistic(x):
    return 1. / (1. + np.exp(-x))


class ReseauDeNeurones:

    def __init__(self, alpha, T):
        self.alpha = alpha
        self.T = T


    def initialisation(self, W, w):
        self.W = W
        self.w = w


    def parametres(self):
        return (self.W, self.w)


    def prediction(self, x):
        a = logistic(np.dot(self.W, x.T))
        output = logistic(np.dot(self.w, a.T))

        if output >= 0.5:
            return 1
        else:
            return 0


    def mise_a_jour(self, x, y):
        a = logistic(np.dot(self.W, x.T))
        y_hat = logistic(np.dot(self.w, a.T))

        delta_output = y - y_hat

        delta_hidden = np.zeros((10, ))
        for idx, node in enumerate(a):
            delta_hidden[idx] = node * (1.0 - node) * self.w[idx] * delta_output     

        # Update des poids de la couche d'output
        for idx, _ in enumerate(self.w):
            self.w[idx] += self.alpha * a[idx] * delta_output
   
        # Update des poids de la couche hidden
        for idx, xi in enumerate(x):
            for idx2, delta in enumerate(delta_hidden):
                self.W[idx2][idx] += self.alpha * xi * delta


    def entrainement(self, X, Y):
        for _ in range(self.T):
            for x, y in zip(X, Y):
                self.mise_a_jour(x, y)