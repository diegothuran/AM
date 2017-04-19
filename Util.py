import os, csv
import numpy as np


def readBase(path=str):
    database_shape = []
    database_rgb = []
    labels = []
    with open(path, 'rb') as csvfile:
        #spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvfile.readlines():
            row = str(row)
            row = row.replace("\\n'","")
            row = row.split(",")
            labels.append(row[0])
            database_shape.append(row[1:9])
            database_rgb.append(row[10:])

    return np.array(database_shape).astype(np.float), np.array(database_rgb).astype(np.float), np.array(labels)