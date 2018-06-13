from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import numpy as np

class Input():
	def __init__(self):
		pass

	def get(self, filename1, filename2, row, col):
		features = np.zeros((row, col))
		label = np.zeros((row))

		row = 0
		fin = open(filename1, 'r')
		lines = fin.readlines()
		for line in lines:
			l = line.strip('\n').split(' ')
			features[row] = l
			row += 1
#		print(features[:4])

		fin = open(filename2, 'r')
		lines = fin.readlines()
		row = 0
		for line in lines:
			l = line.strip('\n').split(' ')
			label[row] = l[0]
			if label[row] == 0:
				label[row] = -1
			row = row + 1
			#label.append(l[0])
#		print(label[:10])
		return features, label

	def init(self):
		trainFeature, trainLabel = self.get('../adult_dataset/adult_train_feature.txt', '../adult_dataset/adult_train_label.txt', 32561, 14)
		testFeature, testLabel = self.get('../adult_dataset/adult_test_feature.txt', '../adult_dataset/adult_test_label.txt', 16281, 14)

	#	vec = DictVectorizer()
	#	lb = preprocessing.LabelBinarizer()
		
	#	print(len(trainFeature))

	#	trainX = vec.fit_transform(trainFeature).toarray()
	#	print(vec.get_feature_names())
	#	trainY = lb.fit_transform(trainLabel)
	#	testX = vec.transform(testFeature).toarray()
	#	testY = lb.transform(testLabel)
		print(trainFeature.shape)

		return trainFeature, trainLabel, testFeature, testLabel