# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pickle
import os
import argparse

from importlib.machinery import SourceFileLoader


def show_recognized_characters(reseau_de_neurones, dataset):
    import matplotlib.pyplot as plt

    i = 0
    for x, y in zip(*dataset):
        y_pred = reseau_de_neurones.prediction(x)
        plt.subplot(4, 5, i + 1)
        plt.subplots_adjust(hspace=0.5)
        plt.imshow(x.reshape(16, 8), cmap=plt.cm.Greys,
                   interpolation='nearest')
        plt.xticks([])
        plt.yticks([])

        plt.title('y=' + str(y) + u', h(x)=' + str(y_pred))
        i += 1

    plt.show()


def recognize_characters(
    ocr, train, test, validation_file, T=25, alpha=0.1, n_neurones_caches=10
):
    ocr = os.path.abspath(ocr)
    name = ocr.replace('/', '.').replace('.', '_')
    sol = SourceFileLoader(
        name, ocr).load_module(name)

    X, Y = train
    X_test, Y_test = test

    reseau_de_neurones = sol.ReseauDeNeurones(alpha, T)

    # Initialisation du réseau
    rng = np.random.RandomState(1234)
    W_init = rng.randn(n_neurones_caches, 128) / 128
    w_init = rng.randn(n_neurones_caches) / n_neurones_caches
    reseau_de_neurones.initialisation(W_init, w_init)

    # Entraînement du réseau
    reseau_de_neurones.entrainement(X, Y)

    # Calcul des erreurs de classification
    erreur_train = np.mean(
        [y != reseau_de_neurones.prediction(x) for x, y in zip(X, Y)])
    print('Erreur de classification sur ensemble d\'entraînement:',
          str(erreur_train * 100) + '%')

    erreur_test = np.mean([y != reseau_de_neurones.prediction(x)
                           for x, y in zip(X_test, Y_test)])
    print('Erreur de classification sur ensemble de test:',
          str(erreur_test * 100) + '%')

    return reseau_de_neurones


def validate_ocr(reseau_de_neurones, validation_file):
    Ww = reseau_de_neurones.parametres()
    W, w = Ww
    with open(validation_file, 'rb') as f:
        W_attendu, w_attendu = pickle.load(f, encoding='latin1')

    diff_W = ((W_attendu - W) ** 2).sum()
    diff_w = ((w_attendu - w) ** 2).sum()
    if diff_W < 1e-20:
        print('Différence entre le W trouvé et celui attendu:', diff_W, '(OK!)')
    else:
        print('Différence entre le W trouvé et celui attendu:', diff_W, '(Erreur!)')

    if diff_w < 1e-20:
        print('Différence entre le w trouvé et celui attendu:', diff_w, '(OK!)')
    else:
        print('Différence entre le w trouvé et celui attendu:', diff_w, '(Erreur!)')


#####
# Execution en tant que script
###
DESCRIPTION = "Lancer la reconnaissance optique de caractères."


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paramètres globaux
    p.add_argument('-ocr', dest="ocr", action='store', type=str, required=False,
                   default="solution_ocr.py", metavar="FICHIER",
                   help="solution à évaluer.")

    p.add_argument('-valider', dest="validation_file", action='store', type=str, required=False,
                   default="ocr_validation.pkl", metavar="FICHIER",
                   help="fichier pickle contenant les résultats de la validation.")

    p.add_argument('-train', dest="train_file", action='store', type=str, required=False,
                   default="train_ocr.pkl", metavar="FICHIER",
                   help="fichier pickle contenant les exemples d'entraînement.")

    p.add_argument('-test', dest="test_file", action='store', type=str, required=False,
                   default="test_ocr.pkl", metavar="FICHIER",
                   help="fichier pickle contenant les exemples de test.")

    p.add_argument('-vizu', dest='is_vizu', action='store_true', required=False,
                   help='visualiser les résultats')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    ocr = args.ocr
    validation_file = args.validation_file
    train_file = args.train_file
    test_file = args.test_file
    is_vizu = args.is_vizu

    if not os.path.isfile(ocr):
        parser.error(
            "-ocr '{0}' doit être un fichier!".format(os.path.abspath(ocr)))

    if not os.path.isfile(train_file):
        parser.error(
            "-train '{0}' doit être un fichier!".format(os.path.abspath(train_file)))

    if not os.path.isfile(test_file):
        parser.error(
            "-test '{0}' doit être un fichier!".format(os.path.abspath(test_file)))

    if not os.path.isfile(validation_file):
        parser.error(
            "-valider '{0}' doit être un fichier!".format(os.path.abspath(validation_file)))

    with open(train_file, 'rb') as f:
        train = pickle.load(f, encoding='latin1')

    with open(test_file, 'rb') as f:
        test = pickle.load(f, encoding='latin1')

    reseau_de_neurones = recognize_characters(
        ocr, train, test, validation_file)

    validate_ocr(reseau_de_neurones, validation_file)

    if is_vizu:
        # Visualisation des prédictions sur l'ensemble de test
        show_recognized_characters(reseau_de_neurones, test)


if __name__ == "__main__":
    main()
