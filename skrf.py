from input import Input
import random
from sklearn import tree
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier  


def compare(v1, v2, num):
	return v1.shape[0] - sum(abs(v1-v2))/2

def main():
	input = Input()
	trainX, trainY, testX, testY = input.init()
	trainNum = trainX.shape[0]
	testNum = testX.shape[0]

	clf = RandomForestClassifier(n_estimators=100, max_features=3, min_samples_split=3)
	clf = clf.fit(trainX, trainY)

	result = clf.predict(trainX)
	correct = compare(result, trainY, trainNum)
	print('train correct num = ', correct, ', ratio = ', correct/32561)

	result = clf.predict(testX)
	correct = compare(result, testY, testNum)
	print('test correct num = ', correct, ', ratio = ', correct/16281)

main()