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