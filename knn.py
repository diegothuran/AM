import math
import operator
from Util import readBase

def normalize(raw_set):
    normalized = raw_set - raw_set.mean() / raw_set.std()
    return normalized

def getEuclideanDistance(instance_1, instance_2, length):
	distance = 0
	for i in range(length):
		distance += pow((instance_1[i] - instance_2[i]), 2)
	return math.sqrt(distance)

def getNeighbors(test_instance, train_set, train_labels, k):
    distances = []
    view_length = len(test_instance) - 1
    for i in range(len(train_set)):
        distance = getEuclideanDistance(test_instance, train_set[i], view_length)
        distances.append((distance, train_labels[i]))
    distances.sort(key=operator.itemgetter(0))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

def getVotes(neighbors):
    votes = {}
    for i in range(len(neighbors)):
        if neighbors[i] in votes:
            votes[neighbors[i]] += 1
        else:
            votes[neighbors[i]] = 1
    votes = sorted(votes.items(), key=operator.itemgetter(0), reverse=True)
    return votes[0][0]

def getAccuracy(test_labels, predictions):
    matches = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            matches += 1
    return (matches / float(len(test_labels))) * 100.0

def knn(test_set, test_labels, train_set, train_labels, k):
    predictions = []
    # train_set = normalize(train_set)
    # test_set = normalize(test_set)
    for i in range(len(test_set)):
        neighbors = getNeighbors(test_set[i], train_set, train_labels, k)
        result = getVotes(neighbors)
        predictions.append(result)
        # print("instance " + repr(i) + ":")
        # print("neighbors = " + repr(neighbors))
        # print('predicted = ' + repr(result) + ', actual = ' + repr(test_labels[i]))
        # print("====== end =====")
    accuracy = getAccuracy(test_labels, predictions)
    print("accuracy: " + repr(accuracy) + "%")
    return predictions

def main():
    shape_train, rgb_train, labels_train = readBase('segmentation.data')
    shape_test, rgb_test, labels_test = readBase('segmentation.test')
    k = 7
    knn(shape_test, labels_test, shape_train, labels_train, k)
    knn(rgb_test, labels_test, rgb_train, labels_train, k)

main()