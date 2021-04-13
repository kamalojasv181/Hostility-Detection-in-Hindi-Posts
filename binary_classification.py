from data_processing import preprocessing
import pandas as pd
import numpy as np
import torch
import random
from sklearn import metrics
import transformers
import re
import emoji
import os
from torch import cuda
from sklearn.model_selection import train_test_split

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

class ModelClass(torch.nn.Module):
    def __init__(self, model_path, dropout, target_labels):
        super(ModelClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_path)
        self.l2 = torch.nn.Dropout(dropout)
        if "hindi-bert" in model_path:
            self.hidden_size = 256
        else:
            self.hidden_size = 768
        self.l3 = torch.nn.Linear(self.hidden_size, target_labels)

    def forward(self, ids, mask, token_type_ids):
        output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1[0][:,0,:])
        output = self.l3(output_2)
        return output



class bin_classifier():
    def __init__(self, arr, model_name, target, epochs, lr):
        self.output_dir = './models'
        self.model_path = model_name      # try: 'ai4bharat/indic-bert' or "monsoon-nlp/hindi-tpu-electra", "monsoon-nlp/hindi-bert"
        self.tokenizer_path = model_name
        self.max_len = 200
        self.TRAIN_BATCH_SIZE = 8
        self.VALID_BATCH_SIZE = 4
        self.epochs = epochs
        self.lr = lr
        self.target_labels = 1
        self.dropout = 0.3
        self.train_size = 0.8759
        self.seed = 23
        self.random_state = 200
        self.n_gpu=1
        self.target = target
        self.encoding = arr[:, 2]
        self.arr = arr.tolist()
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        if self.target=='non-hostile':
            for i in range(0,len(self.encoding)):
                self.arr[i].append([self.encoding[i][0]])
            self.arr = np.array(self.arr)
            tr = self.arr[0:5728]
            ts = self.arr[5728:]
            df_tr = pd.DataFrame(tr)
            df_ts = pd.DataFrame(ts)
        else:
            if self.target == 'hate':
                x = 1
            elif self.target == 'fake':
                x = 2
            elif self.target == 'defamation':
                x = 3
            else:
                x = 4
            a = []
            for i in range(0, len(self.encoding)):
                if self.encoding[i][0]==0:
                    a.append(self.arr[i])

            for i in range(0, len(a)):
                a[i].append([a[i][-1][x]])

            a = np.array(a)
            tr = a[0:2700]
            ts = a[2700:]
            df_tr = pd.DataFrame(tr)
            df_ts = pd.DataFrame(ts)
        new_df_tr = df_tr[[0, 3]].copy()
        new_df_ts = df_ts[[0, 3]].copy()
        self.new_df_tr = new_df_tr.rename(columns={0:"comment_text", 3:"list"})
        self.new_df_ts = new_df_ts.rename(columns={0:"comment_text", 3:"list"})
        self.set_seed()

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        
        train_dataset=self.new_df_tr
        test_dataset=self.new_df_ts
        train_dataset = train_dataset.reset_index(drop=True)
        


        print("TRAIN Dataset: {}".format(train_dataset.shape))
        print("TEST Dataset: {}".format(test_dataset.shape))

        training_set = CustomDataset(train_dataset, tokenizer, self.max_len)
        testing_set = CustomDataset(test_dataset, tokenizer, self.max_len)
        train_params = {'batch_size': self.TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

        test_params = {'batch_size': self.VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

        self.training_loader = DataLoader(training_set, **train_params)
        self.testing_loader = DataLoader(testing_set, **test_params)
        self.model = ModelClass(self.model_path, self.dropout, self.target_labels)
        self.model.to(self.device);
        self.optimizer = torch.optim.Adam(params =  self.model.parameters(), lr=self.lr)
        os.makedirs(self.output_dir, exist_ok=True)

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)

    def validation(self, epoch):
        device = self.device
        self.model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(self.testing_loader, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = self.model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    def train(self, epoch):
        device = self.device
        self.model.train()
        for _,data in enumerate(self.training_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = self.model(ids, mask, token_type_ids)

            self.optimizer.zero_grad()
            loss = self.loss_fn(outputs, targets)
            if _%100==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_model(self):
        Best_score = 0;
        for epoch in range(self.epochs):
            self.train(epoch)
            outputs, targets = self.validation(epoch)
            print(outputs[0:5])
            outputs = np.array(outputs) >= 0.5
            accuracy = metrics.accuracy_score(targets, outputs)
            f1_score_micro = metrics.f1_score(targets, outputs, average='binary')
            print(f"F1 Score (Weighted) = {f1_score_micro}")
            print()
            if f1_score_micro>Best_score:
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                torch.save(self.model,os.path.join(self.output_dir, self.model_path[-10:] + "_" + self.target + "_model.pt"))

                # Good practice: save your training arguments together with the trained model
                # torch.save(args, os.path.join(self.output_dir, "training_args.bin"))
                Best_score = f1_score_micro
 
        
        f1_str = "best f1 score for binary " + self.target + " classification is " + str(Best_score) + " for " + self.model_path[-10:] + "\n"
        file_object = open('results.txt', 'a')
        file_object.write(f1_str)
        file_object.close()