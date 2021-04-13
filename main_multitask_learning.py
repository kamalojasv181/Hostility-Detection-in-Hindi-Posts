from data_processing import preprocessing
from multitask_learning import CustomDataset, ModelClass, multitask_classifier

data = preprocessing('./Dataset/constraint_Hindi_Train - Sheet1_combined.csv')
arr = data.processed()

mod = multitask_classifier(arr,'ai4bharat/indic-bert', 'non-hostile',10, 1e-5)
mod.train_model()