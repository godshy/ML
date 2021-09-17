import sklearn as sl
from sklearn.datasets import load_iris
dataset = load_iris() #150x4 while 4 is the feature (sepal length, sepal width, petal length, petal width)

X = dataset.data
Y = dataset.target #represent for Iris Setosa, Iris Versicolour, Iris Virginica

attribute_means = X.mean(axis = 0) #每个特征的均值， axis=0是一行，1是1列
print(Y)

