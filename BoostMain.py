from input import Input
from sklearn import tree
import numpy as np
import math
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

	return weightedSum, result

def compare(v1, v2, num):
	return v1.shape[0] - sum(abs(v1-v2))/2

def output(cor, error, alpha):	
	print('correct = ')
	print(cor/32561)

	print('error = ')
	print(error)

	print('alpha = ')
	print(alpha)

def main():
	input = Input()
	trainX, trainY, testX, testY = input.init()
	trainNum = trainX.shape[0]
	testNum = testX.shape[0]

	D = np.zeros((trainNum))
	for i in range(trainNum):
		D[i] = 1/trainNum

	T = 300

	trainPredict = np.zeros((T, trainNum))
	testPredict = np.zeros((T, testNum))
	alpha = np.zeros(T)
	cor = np.zeros(T)
	error = np.zeros(T)
	clfs = list()
	for i in range(T):
		print('i = ', i)
		clf = tree.DecisionTreeClassifier(min_samples_split=3)
		clf = clf.fit(trainX, trainY, sample_weight = D)
		clfs.append(clf)

		trainPredict[i, :] = clf.predict(trainX)
		cor[i] = compare(trainPredict[i, :], trainY, trainNum)
	#	print('correct num = ', cor[i])

		error[i], ok = calc_error(trainPredict[i, :], trainY, D)
		if not ok:
			T = i
			break
		D, alpha[i] = update(trainPredict[i, :], trainY, error[i], D)

	output(cor, error, alpha)
	'''
	result = weighted(alpha, trainPredict, T, trainNum)
	correct = compare(result, trainY, trainNum)
	print('train correct num = ', correct, ', ratio = ', correct/32561)
	'''

	for i in range(T):
		testPredict[i, :] = clfs[i].predict(testX)

	weightedSum, result = weighted(alpha, testPredict, T, testNum)
	correct = compare(result, testY, testNum)
	print('test correct num = ', correct, ', ratio = ', correct/16281)

	auc = roc_auc_score(testY, weightedSum)
	print('auc = ', auc)

main()