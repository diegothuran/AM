import naive_bayes
import knn
import Util
from collections import Counter
import numpy as np

def plurality_vote(*votes):
	count = Counter(list(votes))
	return count.most_common()[0][0]

def combined_classifier(shape_train_set, rgb_train_set, train_labels, shape_test_set, rgb_test_set, test_labels, shape_k, rgb_k):
	predictions = []
	for i in range(len(shape_test_set)):
		shape_xk = shape_test_set[i]
		rgb_xk = rgb_test_set[i]

		result_1, score = naive_bayes.test(shape_xk, shape_train_set, train_labels)
		result_2, score = naive_bayes.test(rgb_xk, rgb_train_set, train_labels)

		result_3 = knn.classify(shape_k, shape_xk, shape_train_set, train_labels)
		result_4 = knn.classify(rgb_k, rgb_xk, rgb_train_set, train_labels)

		result = plurality_vote(result_1, result_2, result_3, result_4)

		predictions.append(result)
		Util.printProgressBar(i, (len(shape_test_set)-1), prefix = 'Testing:', suffix = 'Complete')
	accuracy = Util.getAccuracy(test_labels, predictions)
	return predictions, accuracy

# def combined_classifier(shape_train_set, rgb_train_set, train_labels, shape_test_set, rgb_test_set, test_labels, k):
# 	predictions = []
# 	for i in range(len(shape_test_set)):
# 		shape_xk = shape_test_set[i]
# 		rgb_xk = rgb_test_set[i]
#
# 		result_1, score = naive_bayes.test(shape_xk, shape_train_set, train_labels)
# 		result_2, score = naive_bayes.test(rgb_xk, rgb_train_set, train_labels)
#
# 		neighbors = knn.getNeighbors(k, shape_xk, shape_train_set, train_labels)
# 		result_3 = knn.getVotes(neighbors)
# 		neighbors = knn.getNeighbors(k, rgb_xk, rgb_train_set, train_labels)
# 		result_4 = knn.getVotes(neighbors)
#
# 		result = plurality_vote(result_1, result_2, result_3, result_4)
#
# 		predictions.append(result)
# 		Util.printProgressBar(i, (len(shape_test_set)-1), prefix = 'Testing:', suffix = 'Complete')
# 	accuracy = Util.getAccuracy(test_labels, predictions)
# 	return predictions, accuracy

def basic():
	shape_train_set, rgb_train_set, train_labels = Util.readBase('segmentation.test')
	shape_test_set, rgb_test_set, test_labels = Util.readBase('segmentation.data')

	predictions, accuracy = combined_classifier(
		shape_train_set, rgb_train_set, train_labels, shape_test_set, rgb_test_set, test_labels, k=2
	)
	print("Final result: accuracy = {0:.2f}%".format(accuracy*100))

def cross_validation(folds = 10, times = 30, verbose = 1):
	shape_set, rgb_set, labels = Util.readBase('segmentation.test')
	shape_strata = Util.stratify(shape_set, folds, labels)
	rgb_strata = Util.stratify(rgb_set, folds, labels)
	labels_strata = Util.stratify(labels, folds, labels)

	general_accuracy = [[] for k in range(times)]
	mean_accuracy = []

	# ajustando os k vizinhos de cada view
	print("Fitting k for each view...")
	k1 = knn.fitK(shape_set, labels, iterations=2, k_max=11)
	k2 = knn.fitK(rgb_set, labels, iterations=2, k_max=11)

	for i in range(times):
		print("Running cross validaton iteration #"+str(i+1))
		for j in range(folds):
			print("Fold "+str(j+1)+" out of "+str(folds))
			shape_train_set, shape_test_set = Util.split_set(shape_strata, j)

			rgb_train_set, rgb_test_set = Util.split_set(rgb_strata, j)

			train_labels, test_labels = Util.split_set(labels_strata, j)

			# predictions, accuracy = combined_classifier(
			# 	shape_train_set, rgb_train_set, train_labels, shape_test_set, rgb_test_set, test_labels, k=2
			# )

			# usando os valores de k para view
			predictions, accuracy = combined_classifier(
				shape_train_set, rgb_train_set, train_labels, shape_test_set, rgb_test_set, test_labels, shape_k=k1,
				rgb_k=k2
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
	cross_validation(times=1, verbose=3)

main()