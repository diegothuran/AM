from __future__ import division
import numpy as np
import math
import sys
import Util

def naive_bayes(train_set, train_labels, test_set, test_labels):
	predictions = []
	for xk in test_set:
		result, score = test(xk, train_set, train_labels)
		predictions.append(result)
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

def main():
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

if __name__ == "__main__":
	main()