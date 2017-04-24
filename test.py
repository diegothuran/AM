from __future__ import division
import numpy as np
import bayesian_classifier
import Util

with open('segmentation.data', 'rb') as csvfile:
    hits = {}
    total = 210
    i = 1
    for row in csvfile.readlines():
        row = str(row)
        row = row.replace("\\n'","")
        row = row.replace("'","")
        row = row.split(",")
        target = row[0][1:]
        xk = np.array(row[1:]).astype(np.float)

        result, score = bayesian_classifier.test(xk,'rgb')
        if result == target:
            if target in hits:
                hits[target] += 1
            else:
                hits[target] = 1

        step = i/total*100
        print("{0:.2f}% completed...".format(step))
        i += 1

    print('{')
    for key in hits.keys():
        print(key+":"+"{0:.2f}".format(hits[key]/30)+",")
    print('}')
    print(sum(hits.values())/total)