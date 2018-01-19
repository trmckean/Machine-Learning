#Tyler McKean - February 25th, 2016 - Machine Learning
#This program uses the Naive Bayes classifier to classify test set examples as either not spam (0) or spam (1)

#The program operates as follows:
#1) Initialize the data into a training set and test set where each set is ~60% not spam and ~40% spam
#2) Compute the prior probabilities for each class (0 or 1) using the training set
#3) Compute the mean of each feature i given a class (0 or 1) using the training set
#4) Compute the standard deviation of each feature i given a class (0 or 1) using the training set
#5) For each example in the test set use the Naive Bayes classifer in order to predict the class
#6) Calculate the Accuracy, Precision, Recall, and Confusion Matrix based on testing each prediction

#Imports - PrettyTable is a library used to make convenient ASCII tables with data
import math
import random
from prettytable import PrettyTable

#Initialize the data from the external file and create the lists that will be used throughout the program
data = open("spambase.data", "r")
negativeExamples = []
positiveExamples = []
trainingData = []
testData = []
negMeans = []
posMeans = []
negStandardDeviations = []
posStandardDeviations = []

#Split the raw data into separate lists for negative (not spam) and positive (spam) examples
for line in data:
    line = line.rstrip('\n').rstrip('\r').split(',')
    values = []
    for i in range(58):
	values.append(float(line[i]))
    if (values[-1] == 0):
	negativeExamples.append(values)
    else:
	positiveExamples.append(values)

random.shuffle(negativeExamples)
random.shuffle(positiveExamples)

#Create the training set and test set with 40% spam and 60% not spam by using half of each set of examples
for line in range(1394):
    trainingData.append(negativeExamples[line])
for line in range(1394, 2788):
    testData.append(negativeExamples[line])
for line in range(906):
    trainingData.append(positiveExamples[line])
for line in range(906, 1813):
    testData.append(positiveExamples[line])

#Compute the prior probability for the spam (1) and not spam (0) class from the training data
#Probability = number of instances with class n divided by total training instances
priorProbSpam = 906.0/2300.0
priorProbNotSpam = 1394.0/2300.0
print "Prior probability P(1) (spam): ", priorProbSpam
print "Prior probability P(0) (not-spam): ", priorProbNotSpam

#Compute the mean for each feature given each class
#Iterate through each training example by feature and calculate the sum of all feature values for the specified feature
#Mean of feature i given class (0 or 1) = sum of every feature value i given the class divided by the number of instances in that class
for i in range(57):
    tempNegSum = 0.0
    negCount = 0.0
    tempPosSum = 0.0
    posCount = 0.0
    negMean = 0.0
    posMean = 0.0
    for j in range(len(trainingData)):
	if (trainingData[j][-1] == 0):
	    tempNegSum += trainingData[j][i]
	    negCount += 1
	else:
	    tempPosSum += trainingData[j][i]
	    posCount += 1
    negMean = tempNegSum / negCount
    posMean = tempPosSum / posCount

    negMeans.append(negMean)
    posMeans.append(posMean)

#Compute the standard deviation for each feature given each class
#For each feature i, calculate the standard deviation on the training set given the class (0 or 1)
#If the standard deviation computes to 0, re-assign it a low value to not cause errors in later calculations
for i in range(57):
    tempNegVariance = 0.0
    tempPosVariance = 0.0
    tempNegDifference = 0.0
    tempPosDifference = 0.0
    negCount = 0.0
    posCount = 0.0
    negStandardDev = 0.0
    posStandardDev = 0.0
    for j in range(len(trainingData)):
	if (trainingData[j][-1] == 0):
	    tempNegDifference = trainingData[j][i] - negMeans[i]
	    tempNegDifference = tempNegDifference * tempNegDifference
	    tempNegVariance += tempNegDifference
	    negCount += 1
	else:
	    tempPosDifference = trainingData[j][i] - posMeans[i]
	    tempPosDifference = tempPosDifference * tempPosDifference
	    tempPosVariance += tempPosDifference
	    posCount += 1
    negStandardDev = tempNegVariance / (negCount - 1)
    negStandardDev = math.sqrt(negStandardDev)
    if(negStandardDev == 0):
	negStandardDev = 0.1
    negStandardDeviations.append(negStandardDev)

    posStandardDev = tempPosVariance / (posCount - 1)
    posStandardDev = math.sqrt(posStandardDev)
    if(posStandardDev == 0):
	posStandardDev = 0.1
    posStandardDeviations.append(posStandardDev)

#Values that are amended and computed using the following for loop
accuracy = 0.0
precision = 0.0
recall = 0.0
truePositive = 0.0
falsePositive = 0.0
trueNegative = 0.0
falseNegative = 0.0

#Use the Naive Bayes Classifier to classify each test example
#For each instance in the test set, calculate the probability of each feature xi given the class (0 or 1)
#Use predicted class value to determine accuracy, precision, and recall
for i in range(len(testData)):
    negFeatureProbs = []
    posFeatureProbs = []
    negClassify = 0.0
    posClassify = 0.0
    tempNegSum = 0.0
    tempPosSum = 0.0
    #For each feature j, calculate the probability of the feature given each class and add it to a list
    for j in range(57):
	negFeatureProb = 0.0
	posFeatureProb = 0.0
	negFeatureProb = (1 / (math.sqrt(2 * math.pi) * negStandardDeviations[j])) * (math.e ** -(((testData[i][j] - negMeans[j]) ** 2) / (2 * (negStandardDeviations[j] ** 2))))
	posFeatureProb = (1 / (math.sqrt(2 * math.pi) * posStandardDeviations[j])) * (math.e ** -(((testData[i][j] - posMeans[j]) ** 2) / (2 * (posStandardDeviations[j] ** 2))))
	negFeatureProbs.append(negFeatureProb)
	posFeatureProbs.append(posFeatureProb)
    #If a computed probability is 0 or less than 0, assign it a low number in order to avoid erros in future calculations.
    #Take the log of each computed probability and sum them
    for x in range(57):
	if(negFeatureProbs[x] <= 0):
	    negFeatureProbs[x] = 1 * (10 ** -50)
	tempNegSum += math.log(negFeatureProbs[x])
    for y in range(57):
	if(posFeatureProbs[y] <= 0):
	    posFeatureProbs[y] = 1 * (10 ** -50)
	tempPosSum += math.log(posFeatureProbs[y])
    negClassify = math.log(priorProbNotSpam) + tempNegSum
    posClassify = math.log(priorProbSpam) + tempPosSum
    #If the classifer returns equal values for both classes, choose one at random. Otherwise, take the maximum value
    if(negClassify == posClassify):
	predictedClass = random.choice([0, 1])
    else:
	predictedClass = max(negClassify, posClassify)
	if(predictedClass == negClassify):
	    predictedClass = 0
	else:
	    predictedClass = 1
    #Compare the predicted class to the test instance target and assign it as a true positive, false positive, true negative, or false negative prediction
    if(predictedClass == 0):
	if(predictedClass == testData[i][-1]):
	    trueNegative += 1
	else:
	    falseNegative += 1
    else:
	if(predictedClass == testData[i][-1]):
	    truePositive += 1
	else:
	    falsePositive += 1

#Calculate accuracy by dividing the sum of correct predictions by the total number of test instances
totalAccuracy = ((truePositive + trueNegative) / len(testData))
print "Total accuracy on the test set: ", (totalAccuracy * 100), "%"
#Calculate precision by dividing the number of true positive predictions by the sum of the number or true positives and false positives
precision = (truePositive / (truePositive + falsePositive))
print "Precision on the test set: ", (precision * 100), "%"
#Calculate recall by dividing the number of true positive predictions by the sum of the number of true positives and false negatives
recall = (truePositive / (truePositive + falseNegative))
print "Recall on the test set: ", (recall * 100), "%"

#Uses a library to conveniently print a confusion matrix
confusionMatrix = PrettyTable(['', 'Not Spam', 'Spam'])
confusionMatrix.add_row(['Not Spam', trueNegative, falsePositive])
confusionMatrix.add_row(['Spam', falseNegative, truePositive])
print confusionMatrix
