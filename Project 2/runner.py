from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from Util import timing
from oversampling import random_oversample
import time

import random, math
import numpy as np

import Util
import ignore_warnings

@timing
def cross_validation(model, data, labels, folds = 10, times = 30):
	data_strata = Util.stratify(data, folds, labels)
	labels_strata = Util.stratify(labels, folds, labels)

	general_accuracy = [[] for k in range(times)]
	mean_accuracy = []
	execution_time = []
	for i in range(times):
		print("Running cross validaton iteration #"+str(i+1)+" out of "+str(times))
		start_time = time.time()
		for j in range(folds):
			print("Fold "+str(j+1)+" out of "+str(folds))
			train_set, test_set = Util.split_set(data_strata, j)
			train_labels, test_labels = Util.split_set(labels_strata, j)

			model.fit(train_set,train_labels)

			accuracy = test(model, test_set, test_labels)

			general_accuracy[i].append(accuracy)
		execution_time.append(time.time() - start_time)
	for i in range(times):
		mean_accuracy.append(np.mean(general_accuracy[i]))

	print("\nFinal result: mean accuracy = {0:.2f}%".format(np.mean(mean_accuracy)*100))
	print("Accuracy results report: ")
	print(mean_accuracy)
	print("Mean time execution:")
	print(np.mean(execution_time))


def test(model, test_samples, test_labels):
	hits = 0
	for i in range(len(test_samples)):
		sample = test_samples[i]
		target = test_labels[i]

		prediction = model.predict(sample)
		if prediction == target:
			hits += 1
	return hits/len(test_samples)

def main():
	data, labels = random_oversample('abalone-processed.data')
	# run(KNeighborsClassifier(n_neighbors=10,weights="uniform",algorithm="ball_tree",leaf_size=10), data, labels)
	# run(DecisionTreeClassifier(criterion="gini",max_features="log2",min_samples_split=2,min_samples_leaf=3), data, labels)
	# run(SVC(C=100.0,kernel="linear",gamma=0.001), data, labels)
	# run(MLPClassifier(), data, labels)

	#cross_validation(KNeighborsClassifier(), data, labels)
	#cross_validation(DecisionTreeClassifier(), data, labels)
	#cross_validation(SVC(), data, labels)
	cross_validation(MLPClassifier(), data, labels)

main()