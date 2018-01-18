#Tyler McKean - February 16th, 2016 - Machine Learning
#The following program uses SVM's to predict whether or not a specific email is spam. There are three experiments carried out.
#Imports including the package for the SVM to be used as well as the package to produce an ROC curve
from sklearn import svm
import random
import math
import matplotlib.pyplot as plt

#Initial lists to contain the data and manipulate it to be used with the SVM package
data = open("spambase.data", "r")
negativeExamples = []
positiveExamples = []
trainingData = []
testData = []
means = []
standardDeviations = []
trainingFeatures = []
trainingTarget = []
testFeatures = []
testTarget = []
disjointTraining = []
disjointTrainingTarget = []

#Splits the data according to positive examples and negative examples in order to create training and test sets
for line in data:
    line = line.rstrip('\n').rstrip('\r').split(',')
    values = []
    for i in range(58):
	values.append(float(line[i]))
    if (values[-1] == 0):
	negativeExamples.append(values)
    else:
	positiveExamples.append(values)

#Creates the training Data
for line in range(906):
    trainingData.append(positiveExamples[line])
    trainingData.append(negativeExamples[line])

#Creates the test Data
for line in range(906,1812):
    testData.append(positiveExamples[line])
    testData.append(negativeExamples[line])

#Finds the mean value for each feature in the training data
for i in range(57):
    tempSum = 0
    for j in range(len(trainingData)):
	tempSum += trainingData[j][i]
    mean = tempSum / len(trainingData)
    means.append(mean)

#Finds the variance and Standard deviation for each feature in the training data
for i in range(57):
    tempVariance = 0
    tempDifference = 0
    for j in range(len(trainingData)):
	tempDifference = trainingData[j][i] - means[i]
	tempDifference = tempDifference * tempDifference
	tempVariance += tempDifference
    standardDev = tempVariance / (len(trainingData) - 1)
    standardDev = math.sqrt(standardDev)
    standardDeviations.append(standardDev)

#Preprocesses each piece of data in both the training and test sets by subtracting the mean and dividing by the standard deviation
#of the specified feature
for i in range(57):
    for j in range(len(trainingData)):
	trainingData[j][i] = ((trainingData[j][i] - means[i]) / standardDeviations[i])
	testData[j][i] = ((testData[j][i] - means[i]) / standardDeviations[i])

#Shuffles the training data
random.shuffle(trainingData)

#Splits the training data into one list containing only features and another containing target values
for i in range(len(trainingData)):
    trainingFeatures.append(trainingData[i][:-1])
    trainingTarget.append(trainingData[i][-1])

#Splits the test data into a list containing features and another containing target values
for i in range(len(testData)):
    testFeatures.append(testData[i][:-1])
    testTarget.append(testData[i][-1])

#Experiment 1
#Creates 10 disjoint sets from the training data to be used in Experiment 1
for i in range(10):
    tempSetFeatures = []
    tempSetTarget = []
    for j in range((181*i), 181 + (i*181)):
	tempSetFeatures.append(trainingFeatures[j])
	tempSetTarget.append(trainingTarget[j])
    disjointTraining.append(tempSetFeatures)
    disjointTrainingTarget.append(tempSetTarget)

#Lists to hold temporary accuracies in order to determine the best C value for Experiment 1
jAccuracies = []
averageAccuracies = []

#Using 10-fold cross validation, tests each value of C on the disjoint training data and calculates accuracy
for j in range (11):
    tempAverageAccuracy = 0.0
    #Each value of C tests using a different validation set from the disjoint training sets
    for i in range(10):
	tempAccuracy = 0.0
	tempTrainingSet = []
	tempTrainingTarget = []
	linearPredictions = []
	validationSet = disjointTraining[i]
	validationTarget = disjointTrainingTarget[i]
	#Removes the current validation set from the training set 
	for x in range(i) + range(i+1, 10):
	    for y in range(181):
		tempTrainingSet.append(disjointTraining[x][y])
		tempTrainingTarget.append(disjointTrainingTarget[x][y])
	#Sets the value of C to the one currently being tested and creates a linear SVM
	if (j == 0):
	    linearSVM = svm.LinearSVC(C = 0.00000001)
	else:
	    linearSVM = svm.LinearSVC(C = (j/10.0))
	#Trains the SVM on the current training set
	linearSVM.fit(tempTrainingSet, tempTrainingTarget)
	#Uses the learned SVM to test on the validation set
	linearPredictions = linearSVM.predict(validationSet)
	#Calculates the accuracy of the current linear SVM
	for z in range(len(linearPredictions)):
	    if linearPredictions[z] == validationTarget[z]:
		tempAccuracy += 1
	jAccuracies.append(tempAccuracy/(len(validationSet)))
    #Calculates the average accuracy for each C value over the course of their 10 validation/training set combinations
    for t in range((10*j), (10 + (10*j))):
	tempAverageAccuracy += jAccuracies[t]
    averageAccuracies.append(tempAverageAccuracy/10.0)
for line in averageAccuracies:
    print line

#Determines the maximum average accuracy and returns the C value with the highest one
Cval = 0.0
maximum = averageAccuracies[0]
for i in range(len(averageAccuracies)):
    if maximum < averageAccuracies[i]:
	maximum = averageAccuracies[i]
	Cval = i

print "The Best C Value was: ", (Cval / 10.0), " at position: ", Cval

#Uses the above "best" C Value to create a new linear SVM
if Cval == 0.0:
    linearSVM = svm.LinearSVC(C = 0.00000001)
else:
    linearSVM = svm.LinearSVC(C = Cval/10.0)

#Use the final SVM to learn on the entire training set and test on the test data
linearSVM.fit(trainingFeatures, trainingTarget)
finalPredictions = linearSVM.decision_function(testFeatures)
exOnePredictions = linearSVM.predict(testFeatures)
count = 0.0
#Calculates the accuracy of the final SVM's predictions vs. the targets in the test data
for i in range(len(exOnePredictions)):
    if exOnePredictions[i] == testTarget[i]:
	count += 1.0
print "Accuracy for Experiment 1 on Test Data: ", (count / (float(len(exOnePredictions))))

#The following loop calculates the number of true positives, false positives, true negatives, and false negatives for each threshold
#The threshold starts at -10 and is incremented by 0.1 each iteration so that there are 200 thresholds over -10 to 10
#It then calculates the precision, recall, and False Positive Rate to be used in the ROC curve
threshold = -10.0
thresholdData = []
for i in range(200):
    tempData = []
    truePositive = 0.0
    falsePositive = 0.0
    trueNegative = 0.0
    falseNegative = 0.0
    #Checks the current threshold against predictions and assigns TP, FP, TN, or FN to each 
    for j in range(len(finalPredictions)):
	if finalPredictions[j] < threshold:
	    prediction = 0
	    if prediction == testTarget[j]:
		trueNegative += 1
	    else:
		falseNegative += 1
	else:
	    prediction = 1
	    if prediction == testTarget[j]:
		truePositive += 1
	    else:
		falsePositive += 1
    tempData.append(threshold)
    tempData.append(truePositive)
    tempData.append(falsePositive)
    tempData.append(trueNegative)
    tempData.append(falseNegative)
    tempData.append(truePositive / (truePositive + falseNegative)) #recall True Positive Rate
    tempData.append(truePositive / (truePositive + falsePositive)) #precision
    tempData.append(falsePositive / (trueNegative + falsePositive)) #False Positive Rate
    thresholdData.append(tempData)
    if(threshold < 0.01 and threshold > -0.01):
	print "At Threshold 0, Accuracy: ", ((truePositive + trueNegative) / len(finalPredictions))
	print "At Threshold 0, Recall: ", (truePositive / (truePositive + falseNegative))
	print "At Threshold 0, Precision: ", (truePositive / (truePositive + falsePositive))
    threshold += 0.1

#The following code uses an external library to create a pop up containing a plot of the ROC curve.
#It should be noted that the pop up plot needs to be closed in order for the rest of the program to continue
falsePositiveRate = []
truePositiveRate = []
for i in range(len(thresholdData)):
    falsePositiveRate.append(thresholdData[i][7])
    truePositiveRate.append(thresholdData[i][5])
plt.plot(falsePositiveRate, truePositiveRate)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()






#Experiment 2
#Uses the resulting weight vector from experiment one in order to determine features to be selected
weight = linearSVM.coef_.tolist()[0]
#Change each weight to its absolute value
weight = map(abs, weight)
#Find the most important feature in the vector and grab its index
maximumVal = max(weight, key=float)
maxIndex = weight.index(maximumVal)
featureIndices = []
featureIndices.append(maxIndex)
weight[maxIndex] = -1000000
tempTrainingData = []
tempTestData = []
exTwoAccuracies = []
#Create a new training set and test set that only contain the most important feature
for i in range(len(trainingFeatures)):
    tempTrainingInstance = []
    tempTrainingInstance.append(trainingFeatures[i][maxIndex])
    tempTrainingData.append(tempTrainingInstance)
for i in range(len(testFeatures)):
    tempTestInstance = []
    tempTestInstance.append(testFeatures[i][maxIndex])
    tempTestData.append(tempTestInstance)
#For n = 2 to 57, choose the most important n features and create a training and test set using only those features
#Then train and test a linear SVM using the newly created data sets to return a list of the most important features
#And to also return a list of the accuracy of the linear SVM using n features
for i in range(2, 58):
    accuracyCount = 0.0
    exTwoPredictions = []
    maximumVal = max(weight, key=float)
    maxIndex = weight.index(maximumVal)
    featureIndices.append(maxIndex)
    weight[maxIndex] = -1000000
    for j in range(len(trainingFeatures)):
	tempTrainingData[j].append(trainingFeatures[j][maxIndex])
    for x in range(len(testFeatures)):
	tempTestData[x].append(testFeatures[x][maxIndex])
    if Cval == 0:
	linearSVM2 = svm.LinearSVC(C = 0.00000001)
    else:
	linearSVM2 = svm.LinearSVC(C = Cval/10.0)
    linearSVM2.fit(tempTrainingData, trainingTarget)
    exTwoPredictions = linearSVM2.predict(tempTestData)
    for y in range(len(exTwoPredictions)):
	if (exTwoPredictions[y] == testTarget[y]):
	    accuracyCount += 1
    exTwoAccuracies.append((accuracyCount/len(exTwoPredictions)))
print "The below table shows the most important features used in Experiment 2"
print featureIndices
print "The below table shows the accuracy on the test data corresponding to the features used"
print exTwoAccuracies



#Experiment 3
#This algorithm operates the same as Experiment 2 except the n features are chosen at random instead of based on importance
exThreeAccuracies = []
for i in range(2, 58):
    chosenFeatures = []
    exThreeTraining = []
    exThreeTest = []
    exThreePredictions = []
    accuracyCounter = 0.0
    #Chooses n number of features from all available features and returns a list of them
    chosenFeatures = random.sample(range(0, 57), i)
    #For n random features chosen, create a new training and test set containing only the features specified at random (using indices)
    for j in range(len(trainingFeatures)):
	tempTrainingInstance = []
	for x in range(len(chosenFeatures)):
	    tempTrainingInstance.append(trainingFeatures[j][chosenFeatures[x]])
	exThreeTraining.append(tempTrainingInstance)
    for y in range(len(testFeatures)):
	tempTestInstance = []
	for z in range(len(chosenFeatures)):
	    tempTestInstance.append(testFeatures[y][chosenFeatures[z]])
	exThreeTest.append(tempTestInstance)
    #Create a linear SVM using the passed down C value and train and test it using the newly created data sets with features = n
    if Cval == 0:
	linearSVM3 = svm.LinearSVC(C = 0.00000001)
    else:
	linearSVM3 = svm.LinearSVC(C = Cval/10.0)
    linearSVM3.fit(exThreeTraining, trainingTarget)
    exThreePredictions = linearSVM3.predict(exThreeTest)
    #Calculate the accuracy of the SVM using n random features and add it to a list
    for v in range(len(exThreePredictions)):
	if(exThreePredictions[v] == testTarget[v]):
	    accuracyCounter += 1
    exThreeAccuracies.append((accuracyCounter/len(exThreePredictions)))
print "The below table shows the accuracies on the test data corresponding to the features used"
print exThreeAccuracies
