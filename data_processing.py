import pandas as pd
import numpy as np
import re
import emoji
import string
import csv


class preprocessing():
    def __init__(self, csv_file_path):
        self.arr = []
        with open(csv_file_path) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                self.arr.append(row)

    def give_emoji_free_text(self, text):
        allchars = [str for str in text]
        emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
        clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
        return clean_text

    def strip_all_entities(self, text):
        entity_prefixes = ['@','#', '_', '-']
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

    def isSubstring(self, s1, s2):
        M = len(s1)
        N = len(s2)
	 
        for i in range(N - M + 1):
 
	        # For current index i,
	        # check for pattern match 
            for j in range(M):
                if (s2[i + j] != s1[j]):
                    break     
            
            if j + 1 == M :
                return i
	 
        return -1

    def processed(self):
        for row in self.arr:
            row[0] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'http', row[0])
            row[0] = self.give_emoji_free_text(row[0])
            row[0] = row[0].replace('\n','')
            row[0] = self.strip_all_entities(row[0])
        encoding = []
        for row in self.arr:
            vector = [0,0,0,0,0]
            if row[1][0:3] == 'non':
                vector[0]= 1
                encoding.append(vector)
            else:
                if self.isSubstring('hate', row[1]) >= 0:
                    vector[1] = 1
                if self.isSubstring('fake', row[1]) >= 0:
                    vector[2] = 1
                if self.isSubstring('defamation', row[1]) >= 0:
                    vector[3] = 1
                if self.isSubstring('offensive', row[1]) >= 0:
                    vector[4] = 1
                encoding.append(vector)
        for i in range(0,len(encoding)):
            self.arr[i].append(encoding[i])
        self.arr = np.array(self.arr)
        return self.arr 