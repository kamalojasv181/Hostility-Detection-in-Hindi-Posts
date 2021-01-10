from data_processing import preprocessing
from multitask_learning import CustomDataset, ModelClass, multitask_classifier

data = preprocessing('./Dataset/constraint_Hindi_Train - Sheet1_combined.csv')
arr = data.processed()
