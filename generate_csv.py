import pandas as pd
import numpy as np
import torch
import random
from sklearn import metrics
import transformers
import re
import emoji
import string
from sklearn.model_selection import train_test_split
import csv

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from pathlib import Path

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'



csv_file_path = "./Dataset/Test Set Complete - test.csv"

arr = []
with open(csv_file_path) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        arr.append(row)

def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text

def strip_all_entities(text):
        entity_prefixes = ['@']
        for separator in  string.punctuation:
            if separator not in entity_prefixes :
                text = text.replace(separator,' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        return ' '.join(words)

for row in arr:
    row[0] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'http', row[0])
    row[0] = give_emoji_free_text(row[0])
    row[0] = row[0].replace('\n','')
    row[0] = strip_all_entities(row[0])

# Returns true if s1 is substring of s2
def isSubstring(s1, s2):
    M = len(s1)
    N = len(s2)
 
    # A loop to slide pat[] one by one 
    for i in range(N - M + 1):
 
        # For current index i,
        # check for pattern match 
        for j in range(M):
            if (s2[i + j] != s1[j]):
                break
             
        if j + 1 == M :
            return i
 
    return -1

encoding = []
for row in arr:
    vector = [0,0,0,0,0]
    if row[1][0:3] == 'non':
        vector[0]= 1
        encoding.append(vector)
    else:
        if isSubstring('hate', row[1]) >= 0:
            vector[1] = 1
        if isSubstring('fake', row[1]) >= 0:
            vector[2] = 1
        if isSubstring('defamation', row[1]) >= 0:
            vector[3] = 1
        if isSubstring('offensive', row[1]) >= 0:
            vector[4] = 1
        encoding.append(vector)

for i in range(0,len(encoding)):
   arr[i].append(encoding[i])



for i in range(len(arr)):
  arr[i] = [arr[i][0], arr[i][1], arr[i][5]]
print(arr[0])

arr = np.array(arr)
df = pd.DataFrame(arr)

df.head()

new_df = df[[0, 2]].copy()
test_dataset = new_df.rename(columns={0:"comment_text", 2:"list"})
test_dataset.head()

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class Args():
    def __init__(self):
        self.output_dir = 'ResultCSVs'
        self.model_path = "ai4bharat/indic-bert"      # or try: 'mrm8488/HindiBERTa' or "monsoon-nlp/hindi-tpu-electra" or "ai4bharat/indic-bert"
        self.tokenizer_path = "ai4bharat/indic-bert"
        self.max_len = 200
        self.TRAIN_BATCH_SIZE = 16
        self.VALID_BATCH_SIZE = 8
        self.epochs = 10
        self.lr = 1e-5
        self.target_labels = 3
        self.dropout = 0.3
        self.train_size = 0.8
        self.seed = 23
        self.random_state = 200
        self.n_gpu=1
args = Args()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

set_seed(args)

test_params = {'batch_size': args.VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

tokenizer = AutoTokenizer.from_pretrained(args.model_path)

cd "./../../../"

testing_set = CustomDataset(test_dataset, tokenizer, args.max_len)
testing_loader = DataLoader(testing_set, **test_params)

def validation(model, epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets



cd "./.."

model = "indic-bert"

# change these path as per your model paths
model_path = ["./final_models/"+ model +"_non-hostile_model.pt",
              "./final_models/"+ model +"_fake_model_f1_aux.pt",
              "./final_models/"+model +"_defamation_model_f1_aux.pt",
              "./final_models/" + model+"_hate_model_f1_aux.pt",
              "./final_models/"+model+"_offensive_model_f1_aux.pt"]

for i in range(5):
  model = torch.load(model_path[i])
  if i==0:
    host, _ = validation(model, 0)
  if i==1:
    fake, _ = validation(model, 0)
  if i==2:
    defame, _ = validation(model, 0)
  if i==3:
    hate, _ = validation(model, 0)
  if i==4:
    offen, _ = validation(model, 0)
  print(".", end='')

hostile = np.array(host)>0.5

result = []
for i in range(len(hostile)):
  li = [0,0,0,0,0]
  if hostile[i]==1:
    li[0]=1
    result.append(li)
  else:
    x = [hate[i], fake[i], defame[i], offen[i]];
    x_np = np.array(x)>0.5
    flag=0
    if x_np[0]==1:
      li[1]=1
      flag=1
    if x_np[1]==1:
      li[2]=1
      flag=1
    if x_np[2]==1:
      li[3]=1
      flag=1
    if x_np[3]==1:
      li[4]=1
      flag=1
    if flag==0:
      li[np.argmax(np.array(x))+1]=1
    
    result.append(li)

final_result = []
for i in range(len(hostile)):
  li = [i]
  if result[i][0]==1:
    st = "non-hostile"
  else:
    st_list = []
    x = result[i][1:]
    if x[0]==1:
      st_list.append("hate")
    if x[1]==1:
      st_list.append("fake")
    if x[2]==1:
      st_list.append("defamation")
    if x[3]==1:
      st_list.append("offensive")
    
    st = ",".join(st_list)
  li = [i+1, st]
  final_result.append(li)

# Unique ID and Labels Set

import pandas as pd
from pandas import DataFrame

df = DataFrame(final_result, columns=["Unique ID", "Labels Set"])

df.head()

df.to_csv("Submission1.csv", index=False)


### Order of Labels --> [Hostile,defamation,fake,hate,offensive,non-hostile]
### An example      --> [1,0,1,1,0,0]

###########################################################################
#################  Test data Evaluation based on Submission ###############
###########################################################################

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score



def preprocess(df):
    
    df = df.dropna()
    
    df.insert(len(df.columns)-1,'Hostile', np.zeros(len(df),dtype=int))
    df.insert(len(df.columns)-1,'Defamation', np.zeros(len(df),dtype=int))
    df.insert(len(df.columns)-1,'Fake', np.zeros(len(df),dtype=int))
    df.insert(len(df.columns)-1,'Hate', np.zeros(len(df),dtype=int))
    df.insert(len(df.columns)-1,'Offensive', np.zeros(len(df),dtype=int))
    df.insert(len(df.columns)-1,'Non-Hostile', np.zeros(len(df),dtype=int))    
    
    for i in range(len(df)):
        text = df['Labels Set'][i]
        text = text.lower()
        text = text.replace('\n',"")
        text = text.replace('"',"")
        text = text.replace(" ","")
        text = text.split(',')


        for word in text:
            if word == 'defamation':
                df.at[i,'Hostile']    = 1
                df.at[i,'Defamation'] = 1
    
            if word == 'fake':
                df.at[i,'Hostile']    = 1
                df.at[i,'Fake'] = 1
    
            if word == 'hate':
                df.at[i,'Hostile']    = 1
                df.at[i,'Hate'] = 1
    
            if word == 'offensive':
                df.at[i,'Hostile']    = 1
                df.at[i,'Offensive'] = 1
    
            if word == 'non-hostile' and df['Hostile'][i]==0:
                df.at[i,'Hostile']    = 0
                df.at[i,'Non-Hostile'] = 1

    return df 
  


def get_scores(y_true, y_pred):
    
    hostility_true = y_true['Hostile']
    hostility_pred = y_pred['Hostile']
    
    hostility_f1 = f1_score(y_true=hostility_true, y_pred=hostility_pred, average='weighted')
    
    fine_true = y_true[['Defamation','Fake','Hate','Offensive']]
    fine_pred = y_pred[['Defamation','Fake','Hate','Offensive']]
    
    fine_f1          = f1_score(y_true=fine_true, y_pred=fine_pred, average=None)
    defame_f1        = fine_f1[0]
    fake_f1          = fine_f1[1]
    hate_f1          = fine_f1[2]
    offensive_f1     = fine_f1[3]
    weighted_fine_f1 = f1_score(y_true=fine_true, y_pred=fine_pred, average='weighted')

    return [hostility_f1, defame_f1, fake_f1, hate_f1, offensive_f1, weighted_fine_f1]




ground_truth_path      = r"./Dataset/Test Set Complete - test.csv"
submission_file_path   = r"Submission1.csv"


y_true = pd.read_csv(ground_truth_path, names=["Text", "Labels Set"])
y_pred = pd.read_csv(submission_file_path)

y_true = preprocess(y_true)
y_pred = preprocess(y_pred)

team_score = get_scores(y_true,y_pred)
    
        
print("Coarse Grained F1-score: ", team_score[0])
print("Defamation F1-score:     ", team_score[1])
print("Fake F1-score:           ", team_score[2])
print("Hate F1-score:           ", team_score[3])
print("Offensive F1-score:      ", team_score[4])
print("Fine Grained F1-score:   ", team_score[5])

