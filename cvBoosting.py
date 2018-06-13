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

	return weightedSum, result

def compare(v1, v2, num):
	return v1.shape[0] - sum(abs(v1-v2))/2

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
			
			T = 65

			trainPredict = np.zeros((T, trainNum))
			testPredict = np.zeros((T, testNum))
			alpha = np.zeros(T)
			cor = np.zeros(T)
			error = np.zeros(T)
			clfs = list()

			D = np.zeros((trainNum))
			for i in range(trainNum):
				D[i] = 1/trainNum

			for i in range(T):
				clf = tree.DecisionTreeClassifier(min_samples_split=3)
				clf = clf.fit(X_train, Y_train, sample_weight = D)
				clfs.append(clf)

				trainPredict[i, :] = clf.predict(X_train)
				cor[i] = compare(trainPredict[i, :], Y_train, trainNum)
	#			print('correct num = ', cor[i])

				error[i], ok = calc_error(trainPredict[i, :], Y_train, D)
				if not ok:
					T = i
					break
				D, alpha[i] = update(trainPredict[i, :], Y_train, error[i], D)

			for i in range(T):
				testPredict[i, :] = clfs[i].predict(X_test)

			weightedSum, result = weighted(alpha, testPredict, T, testNum)

			rocs[cur] = roc_auc_score(Y_test, weightedSum)
			cur += 1

			print(compare(Y_test, result, testNum), testNum)

		tot += sum(rocs)/5

	print(tot/5)

main()