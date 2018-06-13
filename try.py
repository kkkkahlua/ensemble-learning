import numpy as np 
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Image
#import pydotplus


'''
data = np.array(np.random.randint(0,100,24).reshape(6,4))

train = data[:4]

test = data[4:]

print(train)

print(test)

minmaxTransformer = MinMaxScaler(feature_range=(0,1))
train_transformer = minmaxTransformer.fit_transform(train)
test_transformer = minmaxTransformer.transform(test)

print(train_transformer)
print(test_transformer)
'''

iris = load_iris()

#print(iris.data)
#print(iris.target)

print(iris.data.shape)
print(iris.target.shape)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)
print(predicted.shape)

if (predicted == iris.target).all():
	print('yes')
'''
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") 
'''
'''
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 
'''