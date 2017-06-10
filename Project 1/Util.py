# -*- coding: utf-8 -*-
import os, csv
import numpy as np
from scipy.spatial import distance

'''
    Método para leitura do arquivo da base de dados
'''
def readBase(path=str):
    database_shape = []
    database_rgb = []
    labels = []
    with open(path, 'rb') as csvfile:

        for row in csvfile.readlines():
            row = str(row)
            row = row.replace("\\r", "")
            row = row.replace("\\n", "")
            row = row.replace("b'", "")
            row = row.replace("'", "")
            row = row.split(",")
            labels.append(row[0])
            database_shape.append(row[1:10])
            database_rgb.append(row[10:])

    return np.array(database_shape).astype(np.float), np.array(database_rgb).astype(np.float), np.array(labels)

def getAccuracy(test_labels, predictions):
    matches = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            matches += 1
    return (matches / len(test_labels))

def stratify(dataset, folds, labels):
    strata = [[] for f in range(folds)]
    idx = 0

    if len(dataset) % len(set(labels)) != 0 or (len(dataset)/len(set(labels))) %  folds != 0:
        raise ValueError('The dataset does not support stratification by the number of folds selected!')
    else:
        for sample in dataset:
            strata[idx].append(sample)
            idx = (idx + 1) % folds

    return strata

def split_set(dataset_strata, offset):
    test_set = np.asarray(dataset_strata[offset])
    train_set = []
    for j in (i for i in range(len(dataset_strata)) if i != offset):
        for sample in dataset_strata[j]:
            train_set.append(sample)
    train_set = np.asarray(train_set)

    return train_set, test_set

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 2, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def generate_dissimilarity_matrix(data=[]):
    """
        Função que gera uma matriz de dissimilaridade de um utilizando a distância euclidiana
    :param data: vetor de entrada para o qual será calculada a matriz de dissimilaridade.
    :return: matriz de dissimilaridade
    """
    dissimilarity_matrix = []

    for patter in data:
        temp = []
        for patter2 in data:
            temp.append(distance.euclidean(patter, patter2))
        dissimilarity_matrix.append(temp)

    return dissimilarity_matrix

def generate_dissimilarity_matrix_scipy(data=[]):
    return distance.cdist(data, data)


