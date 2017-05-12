import naive_bayes
import knn
import Util
from collections import Counter

def choose(opt1, opt2, opt3, opt4):
	count = Counter([opt1,opt2,opt3,opt4])
	return count.most_common()[0][0]

def combined_classifier(shape_train_set, rgb_train_set, train_labels, shape_test_set, rgb_test_set, test_labels, k):
	predictions = []
	for i in range(len(shape_test_set)):
		print('{0:.2f}%'.format(i/len(shape_test_set) * 100))
		shape_xk = shape_test_set[i]
		rgb_xk = rgb_test_set[i]

		result_1, score = naive_bayes.test(shape_xk, shape_train_set, train_labels)
		result_2, score = naive_bayes.test(rgb_xk, rgb_train_set, train_labels)

		neighbors = knn.getNeighbors(k, shape_xk, shape_train_set, train_labels)
		result_3 = knn.getVotes(neighbors)
		neighbors = knn.getNeighbors(k, rgb_xk, rgb_train_set, train_labels)
		result_4 = knn.getVotes(neighbors)

		result = choose(result_1, result_2, result_3, result_4)

		predictions.append(result)
	accuracy = Util.getAccuracy(test_labels, predictions)
	return predictions, accuracy

def main():
	shape_train_set, rgb_train_set, train_labels = Util.readBase('segmentation.test')
	shape_test_set, rgb_test_set, test_labels = Util.readBase('segmentation.data')

	predictions, accuracy = combined_classifier(
		shape_train_set, rgb_train_set, train_labels, shape_test_set, rgb_test_set, test_labels, k=2
	)
	print("Final result: accuracy = {0:.2f}%".format(accuracy*100))

main()