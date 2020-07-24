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
        #TODO: .~= À COMPLÉTER =~.
        pass

    def mise_a_jour(self, x, y):
        #TODO: .~= À COMPLÉTER =~.
        pass

    def entrainement(self, X, Y):
        #TODO: .~= À COMPLÉTER =~.
        pass
