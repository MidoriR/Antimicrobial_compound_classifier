import numpy as np
datos_completos = np.loadtxt('colorantespena.csv', dtype = 'S', delimiter = ',')
datos = datos_completos[1:, 1:]

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
	
a = clean_data(datos)
print(a)	
