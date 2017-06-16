import os, csv
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