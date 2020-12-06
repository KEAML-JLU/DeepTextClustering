import csv
import nltk
import numpy as np
import random
from collections import defaultdict
import re

class_num = 10
# sample_size = [1000] * 14
sample_size = [500 + i * 150 for i in range(class_num)]
labels = []
docs = []
# with open('train.csv') as f:
with open('raw_train.csv') as f:
    reader = csv.reader(f)
    for item in reader:
        labels.append(int(item[0]) - 1)
        docs.append('\t'.join(item[1:]))

index_dict = defaultdict(list)
for i, l in enumerate(labels):
    index_dict[l].append(i)

selected_index = []
for i, (l, ids) in enumerate(sorted(index_dict.items(), key=lambda x:x[0])):
    print l, sample_size[i]
    tmp_index = np.random.choice(ids, sample_size[i], replace=False)
    selected_index.extend(tmp_index)

random.shuffle(selected_index)

def clean_doc(text):
    text = text.replace(r'\?{2,}','\?')
    text = text.replace(r'!{2,}','!')
    text = text.replace(r'\.{2,}','\\.')
    text = text.replace('\\n',' ')
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r'\s{2,}',' ', text)
    return text

def process(doc):
    doc = clean_doc(doc)
    doc = filter(lambda x:ord(x)<128,doc)
    sents = []
    for tmp_sent in doc.split('\t'):
        tmp_sents = nltk.sent_tokenize(tmp_sent)
        sents.extend(tmp_sents)
    sents = '\t'.join([' '.join(nltk.word_tokenize(sent)) for sent in sents])
    return sents

s_docs = [process(docs[i]) for i in selected_index]
s_labels = [labels[i] for i in selected_index]

with open('p_train.csv','w') as f:
    writer = csv.writer(f)
    for i in range(len(s_docs)):
        writer.writerow([int(s_labels[i]), s_docs[i]])
