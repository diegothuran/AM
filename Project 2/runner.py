from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import random
import math
import numpy as np

import Util
import ignore_warnings

def run(classifier, data, labels):
	scores = []

	for k in range(30):
		training_ids = random.sample(list(range(len(data))),math.ceil(len(data)*2/3))
		test_ids = [i for i in list(range(len(data))) if i not in training_ids]

		training_samples = [data[i] for i in training_ids]
		training_labels = [labels[i] for i in training_ids]
		test_samples = [data[i] for i in test_ids]
		test_labels = [labels[i] for i in test_ids]

		classifier.fit(training_samples, training_labels)

		hits = 0
		for i in range(len(test_samples)):
			sample = test_samples[i]
			target = test_labels[i]

			prediction = classifier.predict(sample)
			if prediction == target:
				hits += 1

		scores.append(hits/len(test_samples))
		Util.printProgressBar(k, 29, prefix = 'Running:', suffix = 'Complete')
	print("Score = "+str(np.mean(scores)))

def main():
	data, labels = Util.read_base('abalone-ACNN96.data')
	run(KNeighborsClassifier(n_neighbors=10,weights="uniform",algorithm="ball_tree",leaf_size=10), data, labels)
	run(DecisionTreeClassifier(criterion="gini",max_features="log2",min_samples_split=2,min_samples_leaf=3), data, labels)
	run(SVC(C=100.0,kernel="linear",gamma=0.001), data, labels)
	run(MLPClassifier(), data, labels)

	print("-------------------")

	run(KNeighborsClassifier(), data, labels)
	run(DecisionTreeClassifier(), data, labels)
	run(SVC(), data, labels)
	run(MLPClassifier(), data, labels)

main()