import Orange
from scipy.stats import friedmanchisquare
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

def makelist(l):
	inverse = [1-i for i in l]
	return sorted(inverse)

bayes_shape = [0.5142857142857142, 0.5, 0.49047619047619045, 0.46190476190476193, 0.5095238095238095, 0.5142857142857142, 0.5190476190476191, 0.5190476190476191, 0.48095238095238096, 0.5238095238095238]
bayes_rgb = [0.7047619047619048, 0.6761904761904762, 0.7857142857142857, 0.7714285714285715, 0.7619047619047619, 0.7380952380952381, 0.7047619047619048, 0.7571428571428571, 0.7904761904761904, 0.7761904761904762]
knn_shape = [0.5761904761904761, 0.5476190476190477, 0.5952380952380952, 0.5619047619047619, 0.5571428571428572, 0.580952380952381, 0.5714285714285714, 0.5904761904761905, 0.5476190476190477, 0.5952380952380952]
knn_rgb = [0.8904761904761904, 0.8761904761904762, 0.8952380952380953, 0.919047619047619, 0.8666666666666667, 0.9095238095238095, 0.8761904761904762, 0.8809523809523809, 0.9047619047619048, 0.8952380952380953]
combined = [0.8142857142857143, 0.8047619047619048, 0.8476190476190476, 0.8238095238095238, 0.8238095238095238, 0.8571428571428571, 0.8, 0.8095238095238095, 0.8333333333333334, 0.8571428571428571]

stat, p = friedmanchisquare(bayes_shape, bayes_rgb, knn_shape, knn_rgb, combined)
print("Friedman chi-square statistic: "+str(stat)+", p-value: "+str(p))

if(p < 0.05):
	bayes_shape = makelist(bayes_shape)
	bayes_rgb = makelist(bayes_rgb)
	knn_shape = makelist(knn_shape)
	knn_rgb = makelist(knn_rgb)
	combined = makelist(combined)
	ranks = {'Bayes:Shape':[],'Bayes:RGB':[],'KNN:Shape':[], 'KNN:RGB':[], 'Combined':[]}

	for i in range(len(bayes_shape)):
		results = ss.rankdata([bayes_shape[i], bayes_rgb[i], knn_shape[i], knn_rgb[i], combined[i]])
		ranks['Bayes:Shape'].append(results[0])
		ranks['Bayes:RGB'].append(results[1])
		ranks['KNN:Shape'].append(results[2])
		ranks['KNN:RGB'].append(results[3])
		ranks['Combined'].append(results[4])

	avranks = [
		np.mean(ranks['Bayes:Shape']), 
		np.mean(ranks['Bayes:RGB']), 
		np.mean(ranks['KNN:Shape']), 
		np.mean(ranks['KNN:RGB']), 
		np.mean(ranks['Combined'])
	]
	cd = Orange.evaluation.compute_CD(avranks,len(bayes_shape))
	print("Critical nemenyi distance: "+str(cd))
	names = ['Bayes:Shape','Bayes:RGB','KNN:Shape', 'KNN:RGB', 'Combined']
	Orange.evaluation.graph_ranks(avranks,names,cd=cd,width=6,textspace=1.5)
	plt.show()