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
from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show

model_file='./trained_model/model_20200219_epoch_1_step_3000.mdl'
sentence='husband relates several days of malaise, then in theevening of the night of admission'
def get_attention(model_file,sentence):
    model_type = 'bert'
    model_version = 'bert-base-uncased'
    do_lower_case = True
    model=BertForSequenceClassification.from_pretrained('bert-base-uncased',output_attentions=True)
    model=torch.nn.DataParallel(model)
    chpt = torch.load(model_file,map_location=torch.device('cpu'))
    model.load_state_dict(chpt['state_dict'])
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
    tokens=[tokenizer.cls_token] + tokenizer.tokenize(sentence) + [tokenizer.sep_token]
    input_ids=tokenizer.convert_tokens_to_ids(tokens)
    input_ids=torch.tensor(input_ids).unsqueeze(0)
    ##head view
    model.eval()
    outputs = model(input_ids)
    attentions = outputs[-1]
    head_view(attentions, tokens, None)
    ####neuron view
    show(model, model_type, tokenizer, sentence, None)

get_attention(model_file,sentence)

