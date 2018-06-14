from input import Input
from sklearn import tree
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import random

def vote(predict, T, testNum):
	aveSum = np.dot(np.ones(T), predict).reshape(testNum)

	result = np.zeros(testNum)
	for i in range(testNum):
		if aveSum[i] >= 0:
			result[i] = 1
		else:
			result[i] = -1
	return aveSum, result

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

	tot = 0
	for times in range(5):
		print('times = ', times)

		kf = KFold(n_splits=5)
		cur = 0
		rocs = np.zeros(5)

		for train_index, test_index in kf.split(trainX, trainY):
			print('fold = ', cur)

			X_train, X_test = trainX[train_index], trainX[test_index]
			Y_train, Y_test = trainY[train_index], trainY[test_index]

			trainNum = X_train.shape[0]
			testNum = X_test.shape[0]

			T = 150
			clfs = list()
			for i in range(T):
				partX, partY = bootstrap(X_train, Y_train)
				clf = tree.DecisionTreeClassifier(min_samples_split=3, max_features=3)
				clf = clf.fit(partX, partY)
				clfs.append(clf)

			testPredict = np.zeros((T, testNum))
			for i in range(T):
				testPredict[i, :] = clfs[i].predict(X_test)

			aveSum, result = vote(testPredict, T, testNum)

			rocs[cur] = roc_auc_score(Y_test, aveSum)

			print(compare(Y_test, result, testNum), testNum)
			cur += 1

		tot += sum(rocs)/5

	print(tot/5)

main()