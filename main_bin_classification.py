from data_processing import preprocessing
import numpy as np
from binary_classification import CustomDataset, ModelClass, bin_classifier

data = preprocessing('./Dataset/constraint_Hindi_Train - Sheet1.csv')
arr = data.processed()

data_valid = preprocessing('./Dataset/Constraint_Hindi_Valid - Sheet1.csv')
arr_valid = data_valid.processed()

arr = arr.tolist()
arr_valid = arr_valid.tolist()

for row in arr_valid:
	arr.append(row)
arr = np.array(arr)
arr_valid = np.array(arr_valid)

mod = bin_classifier(arr, "ai4bharat/indic-bert", 'non-hostile', 10, 1e-5)
mod.train_model()




