# Check the versions of libraries

# Python version
import sys
#print('Python: {}'.format(sys.version))
# scipy
import scipy
#print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
#print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
#print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
#print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
#print('sklearn: {}'.format(sklearn.__version__))

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loads dataset, please note this dataset was found online via https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv
data_file = "iris_identify\iris_data.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(data_file, names=names)

# Used to print shape and head of dataset, hoping to see 150 rows with 5 columns (20 rows for the head)
print(dataset.shape)
print(dataset.head(20))

# Used for dataset description
print(dataset.describe())

# Used for class distribution
print(dataset.groupby('class').size())

# Below creates some visualizations to help us fully understand the data
# box plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
# histograms
dataset.hist()
pyplot.show()
# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# Creates array
array = dataset.values
X = array[:,0:4]
y = array[:,4]

# Creates test and train sets
X_train, X_val, Y_train, Y_val =train_test_split(X, y, test_size=0.20, random_state=1) 

# Check algorithms
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('KNearest Neighbors', KNeighborsClassifier()))
models.append(('Classification/Regression Trees', DecisionTreeClassifier()))
models.append(('Gaussian Naive Bayes', GaussianNB()))
models.append(('Support Vector Machines', SVC(gamma='auto'))) 
results = []
names =[]

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Plot Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# As the Support Vector Machines algorithm produced the best result we will use this model
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_val)

# Evaluate predictions
print(accuracy_score(Y_val, predictions))
print(confusion_matrix(Y_val, predictions))
print(classification_report(Y_val, predictions))

print(Y_val)