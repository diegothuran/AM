from sklearn.model_selection import StratifiedKFold

import os, csv, time
import numpy as np

def read_base(path=str):
    database = []
    labels = []
    with open(path, 'rb') as csvfile:

        for row in csvfile.readlines():
            row = str(row)
            row = row.replace("\\r", "")
            row = row.replace("\\n", "")
            row = row.replace("b'", "")
            row = row.replace("'", "")
            row = row.split(",")
            labels.append(row[-1])
            database.append(row[:-1])

    return np.array(database).astype(np.float), np.array(labels)

def segment(path=str):
    with open(path, 'rb') as csvfile:

        for row in csvfile.readlines():
            row = str(row)
            row = row.replace("\\r", "")
            row = row.replace("\\n", "")
            row = row.replace("b'", "")
            row = row.replace("'", "")
            row = row.split(",")
            label = row[-1]
            if label in map(str,list(range(1,9))):
                row[-1] = '1'
            elif label in map(str,list(range(9,11))):
                row[-1] = '2'
            elif label in map(str,list(range(11,30))):
                row[-1] = '3'
            print(",".join(row))

def stratify(dataset, folds, labels):
    skf = StratifiedKFold(n_splits=folds)
    strata = []

    for train, test in skf.split(dataset, labels):
        strata.append([dataset[i] for i in test])

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

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{} function took {} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap