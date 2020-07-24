# -*- coding: utf-8 -*-

import pickle
import os
import argparse
import numpy as np

from importlib.machinery import SourceFileLoader


SPAM = 0
HAM = 1
TARGETS = [SPAM, HAM]


def compare_dict(a, b):
    if len(set(a.keys()) - set(b.keys())) != 0:
        return False

    if len(set(b.keys()) - set(a.keys())) != 0:
        return False

    for k in a.keys():
        if a[k] != b[k]:
            print(a[k], b[k])
            return False

    return True


def pretraiter(corpus, V, sol):
    corpusTraite = []
    for d, c in corpus:
        corpusTraite.append((sol.pretraiter(d, V), c))

    return corpusTraite


def detect_spams(detector, train, test, delta=1, freq_threshold=5):
    detector = os.path.abspath(detector)
    name = detector.replace('/', '.').replace('.', '_')
    sol = SourceFileLoader(
        name, detector).load_module(name)

    # Création du vocabulaire
    print("Création du vocabulaire")
    V = sol.creerVocabulaire([d[0] for d in train], seuil=freq_threshold)

    # Prétraitement des corpus
    print("Prétraitement des corpus")
    train_corpus = pretraiter(train, V, sol)
    test_corpus = pretraiter(test, V, sol)

    print("Entraînement et prédiction")
    # Entraînement
    P = sol.Probabilite()
    P.vocabulaire = V

    sol.entrainer(train_corpus, P)

    # Prédiction
    # Calcul des erreurs de classification
    erreur_train = np.mean([y != sol.predire(x, P, TARGETS, delta=delta)[
                        0] for x, y in train_corpus])
    print('Erreur de classification sur ensemble d\'entraînement (' +
          str(len(train_corpus)) + ' items): %.4f' % (erreur_train * 100) + '%')

    erreur_test = np.mean([y != sol.predire(x, P, TARGETS, delta=delta)[
                       0] for x, y in test_corpus])
    print('Erreur de classification sur ensemble de test (' +
          str(len(test_corpus)) + ' items): %.4f' % (erreur_test * 100) + '%')


def validate_detector(detector, validation_file, delta=1):
    passed = np.array([False] * 4)

    detector = os.path.abspath(detector)
    name = detector.replace('/', '.').replace('.', '_')
    sol = SourceFileLoader(
        name, detector).load_module(name)

    with open(validation_file, 'rb') as f:
        res_train, res_train_corpus, res_V, res_prob, res_nbMotsParClasse, res_nbDocsParClasse, res_freqWC = pickle.load(
            f, encoding='latin1')

    print("Test de création du vocabulaire... ", end='')
    testing_V = sol.creerVocabulaire([d[0] for d in res_train], seuil=10)
    if len(res_V - testing_V) == 0 and len(testing_V - res_V) == 0:
        print('OK!')
        passed[0] = True
    else:
        print('Erreur!')

    print("Test de prétraitement... ", end='')
    testing_pretraiter = sol.pretraiter(res_train[0][0], res_V)
    same_lengths = len(testing_pretraiter) == len(res_train_corpus[0][0])
    same_words = all([w1 == w2 for w1, w2 in zip(
        res_train_corpus[0][0], testing_pretraiter)])
    if same_lengths and same_words:
        print('OK!')
        passed[1] = True
    else:
        print('Erreur!')

    print("Test d'entraînement... ", end='')
    testing_P = sol.Probabilite()
    testing_P.vocabulaire = res_V

    sol.entrainer(res_train_corpus, testing_P)

    same_nbMotsParClasse = compare_dict(
        testing_P.nbMotsParClasse, res_nbMotsParClasse)
    same_nbDocsParClasse = compare_dict(
        testing_P.nbDocsParClasse, res_nbDocsParClasse)
    same_freqWC = compare_dict(testing_P.freqWC, res_freqWC)
    if same_nbMotsParClasse and same_nbDocsParClasse and same_freqWC:
        print('OK!')
        passed[2] = True
    else:
        print('Erreur!')

    print("Test de prédiction... ", end='')
    res_P = sol.Probabilite()

    for k, v in res_nbMotsParClasse.items():
        res_P.nbMotsParClasse[k] = v

    for k, v in res_nbDocsParClasse.items():
        res_P.nbDocsParClasse[k] = v

    for k, v in res_freqWC.items():
        res_P.freqWC[k] = v

    res_P.vocabulaire = res_V

    testing_prob = sol.predire(
        res_train_corpus[0][0], res_P, TARGETS, delta=delta)

    if res_prob[0] == testing_prob[0] and ((res_prob[1] - testing_prob[1]) ** 2 < 1e-10):
        print('OK!')
        passed[3] = True
    else:
        print('Erreur!')

    return passed


#####
# Execution en tant que script
###
DESCRIPTION = "Lancer le détecteur de pourriels."


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paramètres globaux
    p.add_argument('-detecteur', dest="detector", action='store', type=str, required=False,
                   default="solution_pourriels.py", metavar="FICHIER",
                   help="solution à évaluer.")

    p.add_argument('-valider', dest="validation_file", action='store', type=str, required=False,
                   default="pourriels_validation.pkl", metavar="FICHIER",
                   help="fichier pickle contenant les résultats de la validation.")

    p.add_argument('-train', dest="train_file", action='store', type=str, required=False,
                   default="train_pourriels.pkl", metavar="FICHIER",
                   help="fichier pickle contenant les exemples d'entraînement.")

    p.add_argument('-test', dest="test_file", action='store', type=str, required=False,
                   default="test_pourriels.pkl", metavar="FICHIER",
                   help="fichier pickle contenant les exemples de test.")

    p.add_argument('-t', dest='is_test_only', action='store_true', required=False,
                   help='effectuer seulement les tests unitaires')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()
    detector = args.detector
    validation_file = args.validation_file
    train_file = args.train_file
    test_file = args.test_file
    is_test_only = args.is_test_only

    if not os.path.isfile(detector):
        parser.error(
            "-detecteur '{0}' doit être un fichier!".format(os.path.abspath(detector)))

    if not is_test_only:

        if not os.path.isfile(train_file):
            parser.error(
                "-train '{0}' doit être un fichier!".format(os.path.abspath(train_file)))

        if not os.path.isfile(test_file):
            parser.error(
                "-test '{0}' doit être un fichier!".format(os.path.abspath(test_file)))

        with open(train_file, 'rb') as f:
            train = pickle.load(f, encoding='latin1')

        with open(test_file, 'rb') as f:
            test = pickle.load(f, encoding='latin1')

        detect_spams(detector, train, test)

    if not os.path.isfile(validation_file):
        parser.error(
            "-valider '{0}' doit être un fichier!".format(os.path.abspath(validation_file)))

    validate_detector(detector, validation_file)


if __name__ == "__main__":
    main()
