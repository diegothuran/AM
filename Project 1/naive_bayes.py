from __future__ import division
from optparse import OptionParser
import numpy as np
import math
import sys
import Util

parser = OptionParser()
parser.add_option("-t","--view",dest="view",help="select view type")
parser.add_option("-v","--verbose",dest="verbose",help="select verbose level",default=1)

def naive_bayes(train_set, train_labels, test_set, test_labels):
	predictions = []
	for i in range(len(test_set)):
		xk = test_set[i]
		result, score = test(xk, train_set, train_labels)
		predictions.append(result)
		Util.printProgressBar(i, (len(test_set)-1), prefix = 'Testing:', suffix = 'Complete')
	accuracy = Util.getAccuracy(test_labels, predictions)
	return predictions, accuracy


def test(xk, samples, labels):
	max_score = 0
	scores = []
	for wi in set(labels):
		score = Posteriori(wi, xk, samples, labels)
		scores.append(score)
		if score > max_score:
			max_score = score
			wl = wi
	return wl, max_score

def Posteriori(wi, xk, samples, labels):
	return p(xk, wi, samples, labels)*Priori(wi, labels)#/sum([p(xk,wr)*Priori(wr) for wr in classes])

def Priori(wi, labels):
	return len([x for x in labels if x == wi])/len(labels)

def p(xk, wi, samples, labels):
	means, covariances = estimate(wi, samples, labels)
	d = len(samples[0])
	val = ( (2*math.pi)**(-d/2) * 
		np.prod(np.array([(1+covariances[j]) for j in range(0,d)]))**(-1/2) *
		math.exp((-1/2)*sum([(xk[j] - means[j])**2/((sys.float_info.min)+covariances[j]) for j in range(0,d)]))
	)
	return val

def estimate(wi, samples, labels):
	d = len(samples[0])
	means = []
	covariances = []
	for j in range(0,d):
		xj = column(viewclass(wi, samples, labels),j)
		mean = sum(xj)/len(xj)
		covariance = sum([(xj[k] - mean)**2 for k in range(0,len(xj))])/len(xj)
		means.append(mean)
		covariances.append(covariance)

	return means, covariances

def column(matrix, i):
    return [row[i] for row in matrix]

def viewclass(wi, samples, labels):
	subview = []
	for k in range(0,len(samples)):
		if labels[k] == wi:
			subview.append(samples[k])

	return subview

def basic():
	shape_train_set, rgb_train_set, train_labels = Util.readBase('segmentation.test')
	shape_test_set, rgb_test_set, test_labels = Util.readBase('segmentation.data')

	print('Executing naive bayes for Shape View...')
	predictions, accuracy = naive_bayes(
		train_set=shape_train_set, 
		train_labels=train_labels, 
		test_set=shape_test_set,
		test_labels=test_labels
	)
	print("Final result: accuracy = {0:.2f}%".format(accuracy*100))

	print('Executing naive bayes for RGB View...')
	predictions, accuracy = naive_bayes(
		train_set=rgb_train_set, 
		train_labels=train_labels, 
		test_set=rgb_test_set,
		test_labels=test_labels
	)
	print("Final result: accuracy = {0:.2f}%".format(accuracy*100))

def cross_validation(view, folds = 10, times = 30, verbose = 1):
	shape_set, rgb_set, labels = Util.readBase('segmentation.test')
	shape_strata = Util.stratify(shape_set, folds, labels)
	rgb_strata = Util.stratify(rgb_set, folds, labels)
	labels_strata = Util.stratify(labels, folds, labels)

	print('Executing naive bayes for %s View...' % view)
	general_accuracy = [[] for k in range(times)]
	mean_accuracy = []

	for i in range(times):
		print("Running cross validaton iteration #"+str(i+1)+" out of "+str(times))
		for j in range(folds):
			print("Fold "+str(j+1)+" out of "+str(folds))
			if view.lower() == 'SHAPE'.lower():
				train_set, test_set = Util.split_set(shape_strata, j)
			elif view.lower() == 'RGB'.lower():
				train_set, test_set = Util.split_set(rgb_strata, j)
			else:
				raise ValueError('View %s not supported!' % view)

			train_labels, test_labels = Util.split_set(labels_strata, j)

			predictions, accuracy = naive_bayes(
				train_set, train_labels, test_set, test_labels
			)

			general_accuracy[i].append(accuracy)

	for i in range(times):
		mean_accuracy.append(np.mean(general_accuracy[i]))

	print("\nFinal result: mean accuracy = {0:.2f}%".format(np.mean(mean_accuracy)*100))
	if verbose >= 2:
		print("Accuracy results report: ")
		print(mean_accuracy)
	if verbose >= 3:
		print("Accuracy detailed report: ")
		print(general_accuracy)

def main():
	# basic()
	(options, args) = parser.parse_args()
	cross_validation(options.view, verbose=int(options.verbose))

if __name__ == "__main__":
	main()