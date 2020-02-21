from bertviz import head_view

import pandas as pd
import numpy as np
import csv
import torch
from transformers import BertTokenizer,BertModel, BertForMaskedLM,BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader,RandomSampler
import math
import time
import os
from sklearn.metrics import roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt

train_file='./data/train_discharge_hua.csv'
val_file='./data/val_discharge_hua.csv'
test_file='./data/test_discharge_hua.csv'

df1=pd.read_csv(test_file)
df1=df1.iloc[1:3255]
filename='./data/trainhua.csv'
df1.to_csv(filename,index=False)

df1=pd.read_csv(test_file)
df1=df1.iloc[10000:10255]
filename2='./data/evalhua.csv'
df1.to_csv(filename2,index=False)

class InputExamples(object):
    def __init__(self,admid=None,tokens=None,input_ids=None,segment_ids=None,input_mask=None,label=None):
        self.admid=admid
        self.tokens=tokens
        self.input_ids=input_ids
        self.segment_ids=segment_ids
        self.input_mask=input_mask
        self.label=label
###dont use simple reader, cuz it cant handle , inside quote
def readfile(filename,max_seq_len):
    examples=[]
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    num_lines = sum(1 for line in open(filename, 'r'))
    with open(filename,'r') as f:
        reader=csv.reader(f,delimiter=',')
        next(reader)
        for line in tqdm(reader,total=num_lines):

            subjid,admid,label,text_a=line
            label=label=='True'

            text_a=tokenizer.tokenize(text_a)
            if (len(text_a)>max_seq_len-2):
                text_a=text_a[0:max_seq_len-2]
            tokens=['[CLS]']
            segment_ids=[0]
            for token in text_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append('[SEP]')
            segment_ids.append(0)
            input_mask=[1]*len(tokens)
            input_ids=tokenizer.convert_tokens_to_ids(tokens)
            while (len(segment_ids)<max_seq_len):
                segment_ids.append(0)
                input_mask.append(0)
                input_ids.append(0)
            example=InputExamples(admid=admid,tokens=tokens,input_ids=input_ids,segment_ids=segment_ids,input_mask=input_mask,label=label)
            examples.append(example)
    return examples

train_features=readfile(filename,32)

input_ids=torch.tensor([item.input_ids for item in train_features],dtype=torch.long)
segment_ids=torch.tensor([item.segment_ids for item in train_features],dtype=torch.long)
input_mask=torch.tensor([item.input_mask for item in train_features],dtype=torch.long)
labels=torch.tensor([item.label for item in train_features],dtype=torch.long)


###
BATCH_SIZE=1
EPOCHS=5
LR=2e-5
WARMUP_PROPORTION=0.1
NUM_TRAIN_STEPS=math.ceil(len(train_features)/BATCH_SIZE)*EPOCHS
NUM_WARMUP_STEPS=NUM_TRAIN_STEPS*WARMUP_PROPORTION
max_grad_norm=1.0

train_data = TensorDataset(input_ids, input_mask, segment_ids, labels)
train_sampler=RandomSampler(train_data)
train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,sampler=train_sampler)
model=BertForSequenceClassification.from_pretrained('bert-base-uncased',output_attentions=True)
optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=NUM_TRAIN_STEPS)  # PyTorch scheduler

model.train()
batch=next(iter(train_loader))
input_ids, input_mask, segment_ids, labels = batch

outputs = model(input_ids, input_mask, segment_ids, labels=labels)
loss, logits, attentions = outputs

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens=tokenizer.convert_ids_to_tokens(input_ids.numpy()[0])
tokens=[str(item) for item in tokens]
head_view(attentions, tokens,None)


####neuron view
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show
model_type = 'bert'
model_version = 'bert-base-uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
sentence_a = "date of birth: sex: fservice: medicineallergies:patient recorded as having no known allergies to drugsattending:chief complaint:hematemesismajor surgical or invasive procedure:banding x 4 of esophageal variceshistory of present illness:pt is a 74yo woman with pmh of ms, autoimmune hepatitis"
show(model, model_type, tokenizer, sentence_a, None)