import numpy as np
from sklearn import svm
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.preprocessing import Imputer


#Function for cleaning data for use in scikit learn

def clean_data(dataset):
	'''returns a float array of the values obtainded by padel descriptor'''
	array = dataset.shape
	clean_dataset = np.empty(array)
	
	for columna in range(array[1]):
		for element in range(array[0]):
			if dataset[element, columna] == b'' or dataset[element, columna] == b'Infinity':
				clean_dataset[element, columna] = np.NAN
			else:
				clean_dataset[element, columna] = float(dataset[element, columna])
				
	
	return clean_dataset

#loading and selecting data

all_data = np.loadtxt ('antifungales.csv', delimiter = ',', dtype = 'S') #Training data set
header = all_data [0, :]
X_train = all_data [1:, 1:] 
test_data = np.loadtxt ("colorantespena.csv", dtype = 'S', delimiter = ',') #Test data set
X_test = test_data [1:, 1:] 

X_train1 = clean_data(X_train)
X_test1 = clean_data(X_test)

#Imputation of data

imp = Imputer (missing_values='NaN', strategy='mean', axis=0)
X_train2 = imp.fit_transform (X_train1)
X_test2 = imp.fit_transform (X_test1)


#Tranining the model with one class svm

clf = svm.OneClassSVM (nu = 0.1, kernel = 'rbf', gamma = 0.1)
clf.fit (X_train2)

#Test the data

y_pred_train = clf.predict(X_train2)
y_pred_test = clf.predict(X_test2)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size

print ("error train:" , n_error_train, "/70", "error test:", n_error_test, "/29")

