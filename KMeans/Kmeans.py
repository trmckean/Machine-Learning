# Machine Learning - March 8th, 2016 - Tyler McKean
# This program implements the K-means algorithm in order to train and test on the optdigits data
# The main loop runs the K-means algorithm on the optidigits data where K is defined by a
# variable declaration below in the program and produces values for the Sum Squared Error,
# Sum Squared Separation, and the Mean Entropy of a Clustering. It also produces the accuracy
# of the resulting clustering when tested using the test data and outputs an image file
# containing a visual representation of the resulting clustering.

# Note: This program can take up to a couple minutes to complete, especially when K is set to a
# larger value such as 30

# To run this program it needs to be in the same directory as the files "optidigits.train" and "optidigits.test".
# It needs to be a python file and can be executed by typing "python Kmeans.py"
# To change the value of K, change the value of the variable numK. It is currently set to 30
numK = 30

# Imports
import math
import random
from prettytable import PrettyTable

# Initalize the data from the external data set files and create the structures that use them
rawTrainData = open("optdigits.train", "r")
rawTestData = open("optdigits.test", "r")
trainingData = []
testData = []

# Put the raw training data into a list to be used throughout the program
for line in rawTrainData:
    line = line.rstrip('\n').rstrip('\r').split(',')
    values = []
    for i in range(65):
        values.append(float(line[i]))
    trainingData.append(values)

# Put the raw test data into a list to be used throughout the program
for line in rawTestData:
    line = line.rstrip('\n').rstrip('\r').split(',')
    values = []
    for i in range(65):
        values.append(float(line[i]))
    testData.append(values)

# Experiment - Run K-means clustering using Euclidean distance 5 times with different random
# number seeds where K can be set by the user to obtain the final cluster centers.
sseList = []
sssList = []
entropyList = []
resultingCentroids = []
clusterClassFrequencyList = []

for i in range(5):
    # Set initial cluster centers
    centroids = []
    random.seed()
    for x in range(numK):
        values = []
        for y in range(64):
            values.append(random.uniform(0, 16))
        centroids.append(values)

    # The while loop runs K means until the sum squared error is at the minimum
    prevSumSquaredError = 1.0
    sumSquaredError = 0.0
    count = 0
    while (sumSquaredError < prevSumSquaredError) or count <= 1:
        prevSumSquaredError = sumSquaredError
        distances = []
        closestClusters = []

        # Calculate which center each datapoint is closest to
        for x in range(len(trainingData)):
            minDistance = 0.0
            minDistanceClass = 0
            tempDistance = []

            # Calculate the Euclidean distance from each center
            for z in range(numK):
                tempDistanceSum = 0.0
                for j in range(64):
                    tempDistanceAttribute = 0.0
                    tempDistanceAttribute = (trainingData[x][j] - centroids[z][j])
                    tempDistanceAttribute = tempDistanceAttribute * tempDistanceAttribute
                    tempDistanceSum += tempDistanceAttribute
                tempDistance.append(math.sqrt(tempDistanceSum))
            minDistance = min(tempDistance)
            for m in range(len(tempDistance)):
                if (minDistance == tempDistance[m]):
                    minDistanceClass = m
            closestClusters.append(minDistanceClass)

        # Each center finds the centroid of the points it owns
        for t in range(numK):
            tempClusteredData = []
            for u in range(len(trainingData)):
                if (closestClusters[u] == t):
                    tempClusteredData.append(trainingData[u])
            if (len(tempClusteredData) != 0):
                for a in range(64):
                    tempSum = 0.0
                    for b in range(len(tempClusteredData)):
                        tempSum += tempClusteredData[b][a]
                    tempSum = (tempSum / len(tempClusteredData))
                    centroids[t][a] = tempSum
            # If a center has no points in its cluster, randomly re-assign the center
            else:
                for a in range(64):
                    centroids[t][a] = random.uniform(0, 16)

        # Calculate the Sum Squared Error
        sumSquaredError = 0.0
        for v in range(len(trainingData)):
            tempError = 0.0
            relevantCentroid = centroids[closestClusters[v]]
            tempSum = 0.0
            for x in range(64):
                tempTerm = 0.0
                tempTerm = (trainingData[v][x] - relevantCentroid[x])
                tempTerm = tempTerm * tempTerm
                tempSum += tempTerm
            tempError = math.sqrt(tempSum)
            tempError = tempError * tempError
            sumSquaredError += tempError
        count = count + 1
        print "K-Means Iterations = ", count
    resultingCentroids.append(centroids)
    sseList.append(sumSquaredError)

    # Calculate the Sum Squared Separation
    sumSquaredSeparation = 0.0
    for g in range(numK):
        for k in range(g + 1, numK):
            tempSum = 0.0
            tempDistance = 0.0
            for h in range(64):
                tempTerm = 0.0
                tempTerm = (centroids[g][h] - centroids[k][h])
                tempTerm = tempTerm * tempTerm
                tempSum += tempTerm
            tempDistance = math.sqrt(tempSum)
            tempDistance = tempDistance * tempDistance
            sumSquaredSeparation += tempDistance
    sssList.append(sumSquaredSeparation)

    # Calculate the Mean Entropy of the resulting clustering
    meanEntropy = 0.0
    clusterClassFrequency = []
    for e in range(numK):
        entropy = 0.0
        numClusterInstances = 0.0
        clusterList = []
        for r in range(len(closestClusters)):
            if (e == closestClusters[r]):
                numClusterInstances += 1
                clusterList.append(r)
        classFrequency = []

        # Calculate the entropy of the given cluster
        for c in range(10):
            clusterClassProb = 0.0
            numClusterClassInstances = 0.0
            for x in range(len(clusterList)):
                if (trainingData[clusterList[x]][-1] == c):
                    numClusterClassInstances += 1
            classFrequency.append(numClusterClassInstances)
            if (numClusterInstances == 0):
                clusterClassProb = (1 * (10 ** -50))
            else:
                clusterClassProb = numClusterClassInstances / numClusterInstances
            if (clusterClassProb == 0):
                clusterClassProb = (1 * (10 ** -50))
            entropy += (clusterClassProb * math.log(clusterClassProb, 2))
        entropy = (entropy * -1)
        meanEntropy += (numClusterInstances / len(closestClusters)) * entropy
        clusterClassFrequency.append(classFrequency)
    clusterClassFrequencyList.append(clusterClassFrequency)
    entropyList.append(meanEntropy)
    print i

# Determine the experiment iteration that resulted in the lowest Sum Squared Error
minSSE = min(sseList)
for c in range(len(sseList)):
    if (minSSE == sseList[c]):
        bestRun = c
for error in range(len(sseList)):
    print sseList[error]
print "The run that resulted in the lowest sumSquaredError was: ", bestRun
print "The Sum Squared Error for that run was: ", sseList[bestRun]
print "The Sum Squared Separation for that run was: ", sssList[bestRun]
print "The Mean Entropy of the clustering was: ", entropyList[bestRun]

# Run the test data on the best set of resulting cluster centers
testCentroids = resultingCentroids[bestRun]
testClusterList = clusterClassFrequencyList[bestRun]
clusterClasses = []
for d in range(len(testClusterList)):
    clusterClass = 0.0
    clusterClass = max(testClusterList[d])
    for j in range(len(testClusterList[d])):
        if (clusterClass == testClusterList[d][j]):
            clusterClasses.append(j)
            break

# Output each cluster center's associated class
for z in range(len(clusterClasses)):
    print clusterClasses[z]

# Calculate the accuracy after assigning each test instance to a class of the closest center
accuracy = 0.0
predicted = []
for x in range(len(testData)):
    minDistance = 0.0
    minDistanceClass = 0.0
    tempDistance = []
    instanceClass = 0.0

    # Calculate the closest center for each test instance and calculate accuracy
    for z in range(numK):
        tempDistanceSum = 0.0
        for j in range(64):
            tempDistanceAttribute = 0.0
            tempDistanceAttribute = (testData[x][j] - testCentroids[z][j])
            tempDistanceAttribute = tempDistanceAttribute * tempDistanceAttribute
            tempDistanceSum += tempDistanceAttribute
        tempDistance.append(math.sqrt(tempDistanceSum))
    minDistance = min(tempDistance)
    for m in range(len(tempDistance)):
        if (minDistance == tempDistance[m]):
            minDistanceClass = m
            break
    instanceClass = clusterClasses[m]
    if (instanceClass == testData[x][-1]):
        accuracy += 1
    predicted.append(instanceClass)
accuracy = (accuracy / len(testData)) * 100
print "Accuracy on the test data using the best run with K = ", numK, "was: ", accuracy, "%"

# Create a confusion matrix for the results on the test data
confusionMatrix = PrettyTable([' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
for actualValue in range(10):
    row = []
    row.append(actualValue)
    for predictedValue in range(10):
        tempSum = 0
        for instance in range(len(testData)):
            if (predicted[instance] == predictedValue and testData[instance][-1] == actualValue):
                tempSum += 1
        row.append(tempSum)
    confusionMatrix.add_row(row)
print confusionMatrix

# Create a visualization of each cluster center
width = 8
height = 8 * numK
filename = 'centroids.pgm'
fout = open(filename, 'w')
fout.write("P2\n")
fout.write(str(width) + " ")
fout.write(str(height) + '\n')
fout.write("16\n")
for f in range(len(testCentroids)):
    for x in range(len(testCentroids[f])):
        fout.write(str(int(testCentroids[f][x])))
        fout.write(" ")
fout.close()
