from input import Input
from sklearn import tree
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def calc_error(h, f, D):
	error = np.dot(D, (abs(h-f)/2).reshape(h.shape[0]))
	
	if error > 0.5:
		return error, False
	
	return error, True

def update(h, f, error, D):
	alpha = math.log((1-error)/error)/2

	D = D * (math.e ** np.dot(-alpha, h*f).reshape(h.shape[0]))
	D /= sum(D)

	return D, alpha

def weighted(alpha, predict, T, testNum):
	weightedSum = np.dot(alpha.reshape((1, T)), predict).reshape(testNum)

	result = np.zeros(testNum)
	for i in range(testNum):
		if weightedSum[i] >= 0:
			result[i] = 1
		else:
			result[i] = -1

	return result

def compare(v1, v2, num):
	return v1.shape[0] - sum(abs(v1-v2))/2

def main():
	input = Input()
	trainX, trainY, testX, testY = input.init()

	kf = KFold(n_splits=5)
	cur = 0
	rocs = np.zeros(5)

	for train_index, test_index in kf.split(trainX, trainY):
		X_train, X_test = trainX[train_index], trainX[test_index]
		Y_train, Y_test = trainY[train_index], trainY[test_index]

		trainNum = X_train.shape[0]
		testNum = X_test.shape[0]

		T = 100
		clfs = list()
		for i in range(T):
			print('i = ', i)
			partX, partY = bootstrap(X_test, Y_train)
			clf = tree.DecisionTreeClassifier(min_samples_split=3, max_features=3)
			clf = clf.fit(partX, partY)
			clfs.append(clf)

		testPredict = np.zeros((T, testNum))
		for i in range(T):
			testPredict[i, :] = clfs[i].predict(X_test)

		result = vote(testPredict, T, testNum)

		rocs[cur] = roc_auc_score(Y_test, result)
		cur += 1

	print(rocs)
	print(sum(rocs)/5)

main()