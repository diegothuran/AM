from Util import readBase
import math
import operator

def getEuclideanDistance(instance1, instance2, length):
	distance = 0
	for i in range(length):
		distance += pow((instance1[i] - instance2[i]), 2)
	return math.sqrt(distance)

def getNeighbors(testInstance, trainingSet, trainingLabels, k):
    distances = []
    viewLength = len(testInstance) - 1
    for i in range(len(trainingSet)):
        distance = getEuclideanDistance(testInstance, trainingSet[i], viewLength)
        distances.append((distance, trainingLabels[i]))
    distances.sort(key=operator.itemgetter(0))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

def getVotes(neighbors):
    labelsVotes = {}
    for i in range(len(neighbors)):
        result = neighbors[i]
        if result in labelsVotes:
            labelsVotes[result] += 1
        else:
            labelsVotes[result] = 1
    votes = sorted(labelsVotes.items(), key=operator.itemgetter(0), reverse=True)
    return votes[0][0]

def getAccuracy(testLabels, predictions):
    matches = 0
    for i in range(len(testLabels)):
        if testLabels[i] == predictions[i]:
            matches += 1
    return (matches / float(len(testLabels))) * 100.0

def knn(testSet, testLabels, trainingSet, trainingLabels, k):
    predictions = []
    for i in range(len(testSet)):
        print("instance " + repr(i) + ":")
        neighbors = getNeighbors(testSet[i], trainingSet, trainingLabels, k)
        print("neighbors = " + repr(neighbors))
        result = getVotes(neighbors)
        predictions.append(result)
        print('predicted = ' + repr(result) + ', actual = ' + repr(testLabels[i]))
        print("====== end =====")
    accuracy = getAccuracy(testLabels, predictions)
    print("Accuracy: " + repr(accuracy) + "%")
    return predictions

def main():
    shapeTraining, rgbTraining, labelsTraining = readBase('segmentation.data')
    shapeTest, rgbTest, labelsTest = readBase('segmentation.test')
    k = 7
    # knn(shapeTest, labelsTest, shapeTraining, labelsTraining, k)
    knn(rgbTest, labelsTest, rgbTraining, labelsTraining, k)
main()
