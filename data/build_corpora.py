import csv
import nltk
import numpy as np
import random
from collections import defaultdict

sample_size = 1000
labels = []
docs = []
with open('train.csv') as f:
    reader = csv.reader(f)
    for item in reader:
        labels.append(int(item[0]) - 1)
        docs.append(item[-1])

index_dict = defaultdict(list)
for i, l in enumerate(labels):
    index_dict[l].append(i)

selected_index = []
for l, ids in index_dict.items():
    tmp_index = np.random.choice(ids, sample_size, replace=False)
    selected_index.extend(tmp_index)

random.shuffle(selected_index)

def process(doc):
    doc = filter(lambda x:ord(x)<128,doc)
    sents = nltk.sent_tokenize(doc)
    sents = '\t'.join([' '.join(nltk.word_tokenize(sent)) for sent in sents])
    return sents

s_docs = [process(docs[i]) for i in selected_index]
s_labels = [labels[i] for i in selected_index]

with open('p_train.csv','w') as f:
    writer = csv.writer(f)
    for i in range(len(s_docs)):
        writer.writerow([int(s_labels[i]), s_docs[i]])
