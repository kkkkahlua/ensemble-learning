from input import Input
import random
from sklearn import tree
import numpy as np
import math

def vote(predict, T, testNum):
	sum = np.dot(np.ones(T), predict).reshape(testNum)

	result = np.zeros(testNum)
	for i in range(testNum):
		if sum[i] >= 0:
			result[i] = 1
		else:
			result[i] = -1
	return result

def compare(v1, v2, num):
	return v1.shape[0] - sum(abs(v1-v2))/2

def bootstrap(X, Y):
	num = X.shape[0]
	retX = np.zeros((X.shape[0], X.shape[1]))
	retY = np.zeros(num)
	for i in range(num):
		idx = random.randint(0, num-1)
		retX[i, :] = X[idx, :]
		retY[i] = Y[idx]
	return retX, retY

def main():
	input = Input()
	trainX, trainY, testX, testY = input.init()
	trainNum = trainX.shape[0]
	testNum = testX.shape[0]

	T = 100
	clfs = list()
	for i in range(T):
		print('i = ', i)
		partX, partY = bootstrap(trainX, trainY)
		clf = tree.DecisionTreeClassifier(min_samples_split=3, max_features=3)
		clf = clf.fit(partX, partY)
		print(clf.feature_importances_)
		clfs.append(clf)

	testPredict = np.zeros((T, testNum))
	for i in range(T):
		testPredict[i, :] = clfs[i].predict(testX)

	result = vote(testPredict, T, testNum)

	correct = compare(result, testY, testNum)
	print('test correct num = ', correct, ', ratio = ', correct/16281)

main()