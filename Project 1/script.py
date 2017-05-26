import Orange
from scipy.stats import friedmanchisquare
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

a = [1/x for x in normal(78.5,3.1,30)]
b = [1/x for x in normal(81.0,4.0,30)]
c = [1/x for x in normal(79.9,2.2,30)]
ranks = {'a':[],'b':[],'c':[]}

for i in range(30):
	results = ss.rankdata([a[i], b[i], c[i]])
	ranks['a'].append(results[0])
	ranks['b'].append(results[1])
	ranks['c'].append(results[2])

avranks = [np.mean(ranks['a']), np.mean(ranks['b']), np.mean(ranks['c'])]
names = ['a','b','c']
cd = Orange.evaluation.compute_CD(avranks,30)
Orange.evaluation.graph_ranks(avranks,names,cd=cd,width=6,textspace=1.5)
plt.show()