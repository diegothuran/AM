import os, csv
import numpy as np

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
