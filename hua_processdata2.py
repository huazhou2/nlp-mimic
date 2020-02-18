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

train_features=readfile(filename,128)

input_ids=torch.tensor([item.input_ids for item in train_features],dtype=torch.long)
segment_ids=torch.tensor([item.segment_ids for item in train_features],dtype=torch.long)
input_mask=torch.tensor([item.input_mask for item in train_features],dtype=torch.long)
labels=torch.tensor([item.label for item in train_features],dtype=torch.long)


###
BATCH_SIZE=32
EPOCHS=5
LR=2e-5
WARMUP_PROPORTION=0.1
NUM_TRAIN_STEPS=math.ceil(len(train_features)/BATCH_SIZE)*EPOCHS
NUM_WARMUP_STEPS=NUM_TRAIN_STEPS*WARMUP_PROPORTION
max_grad_norm=1.0

train_data = TensorDataset(input_ids, input_mask, segment_ids, labels)
train_sampler=RandomSampler(train_data)
train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,sampler=train_sampler)
model=BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=NUM_TRAIN_STEPS)  # PyTorch scheduler

model.train()



eval_features=readfile(filename2,512)

adm_ids=torch.tensor([int(item.admid) for item in eval_features],dtype=torch.long)
input_ids=torch.tensor([item.input_ids for item in eval_features],dtype=torch.long)
segment_ids=torch.tensor([item.segment_ids for item in eval_features],dtype=torch.long)
input_mask=torch.tensor([item.input_mask for item in eval_features],dtype=torch.long)
labels=torch.tensor([item.label for item in eval_features],dtype=torch.long)

eval_data = TensorDataset(adm_ids,input_ids, input_mask, segment_ids, labels)
eval_sampler=RandomSampler(eval_data)
eval_loader=DataLoader(eval_data,batch_size=BATCH_SIZE,sampler=eval_sampler)


def get_roc(res,filename):
    accuracy_overal=sum(res.pred_labels==res.true_labels)/res.shape[0]*100
    res=res.sort_values(by='admid')
    score=(res.groupby('admid').pred_probs.agg(max)+res.groupby('admid').pred_probs.agg(sum)/2)/(1+res.groupby('admid').pred_probs.agg(len)/2)
    label=res.groupby('admid').true_labels.agg(min)
    pred_labels=res.groupby('admid').pred_labels.agg(sum)/res.groupby('admid').pred_labels.agg(len)>0.5
    accuracy_subj=sum(label==pred_labels)/len(label)*100

    tpr,fpr,thresholds=roc_curve(label,score)
    auc_score=auc(tpr,fpr)

    ##doing plot
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot((0,1),(0,1),'k--')
    plt.plot(tpr,fpr,label='Validation (area={:.3f}'.format(auc_score))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    basename=os.path.basename(filename)
    plt.title(os.path.splitext(basename)[0])
    plt.legend(loc='best')
    ###doing precision recall
    precision,recall,thresholds=precision_recall_curve(label,score)
    auc_score2=auc(recall,precision)
    plt.subplot(122)
    plt.plot(recall,precision,label='Validation (area={:.3f}'.format(auc_score2))
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title(os.path.splitext(basename)[0])
    plt.legend(loc='best')
    plt.savefig(filename)
    prec_rec_df=pd.DataFrame(list(zip(precision,recall,thresholds)),columns=['precision','recall','thresholds'])
    prec_rec_df=prec_rec_df[prec_rec_df.precsion>0.799999].reset_index()
    rp80=np.NaN
    if prec_rec_df.size()>0:
        rp80=prec_rec_df.iloc[0].recall

    return auc_score,auc_score2,accuracy_overal,accuracy_subj,rp80

for epoch in range(EPOCHS):
    print('doing epoch : ',epoch+1,'\n')
    for step,batch in enumerate(train_loader):
        #batch=next(iter(train_loader))
        input_ids,input_mask,segment_ids,labels=batch

        outputs=model(input_ids,input_mask,segment_ids,labels=labels)
        loss, logits=outputs[:2]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        if step % 5 == 0:
            pred=torch.max(logits,1)[1]
            correct=pred.eq(labels).numpy()
            print('step: %d  loss: %.2f   accuracy: %.2f%%' % (step,loss,(np.sum(correct) / len(correct) * 100)))
        if step and step % 5 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'accuracy':(np.sum(correct) / len(correct) * 100),
                'loss':loss
            }
            os.makedirs('trained_model',exist_ok=True)
            filename='./trained_model/model_'+time.strftime("%Y%m%d")+'_epoch_'+str(epoch)+'_step_'+str(step) +'.mdl'
            torch.save(state,filename)
            chpt=torch.load(filename)
            model.load_state_dict(chpt['state_dict'])
            model.eval()
            eval_accuracy,eval_loss=0,0
            true_labels=[]
            pred_labels=[]
            pred_probs=[]
            admids=[]
            for adm_ids,input_ids,input_mask,segment_ids,labels in tqdm(eval_loader):
                with torch.no_grad():
                    print('.')
                    outputs=model(input_ids,input_mask,segment_ids,labels=labels)
                    loss,logits=outputs[:2]
                    admids.extend(adm_ids.numpy())
                    true_labels.extend(labels.numpy())
                    pred_labels.extend(torch.max(logits,1)[1].numpy())
                    pred_probs.extend(torch.softmax(logits,1)[:,1].detach().numpy())
            res=pd.DataFrame({'admid':admids,'true_labels':true_labels,'pred_labels':pred_labels,'pred_probs':pred_probs})
            res.to_csv('pred_res.csv',index=False)
            os.makedirs('val_model',exist_ok=True)
            filename='./val_model/model_'+time.strftime("%Y%m%d")+'_epoch_'+str(epoch)+'_step_'+str(step)+'.png'
            auc_score,auc_score2,accuracy1,accuracy2,rp80=get_roc(res,filename)
            print('evaluation model:\n epoch: %d  step: %d auc score: %.3f  precision_recall_auc: %.3f  accuracy average: %.3f accuracy byadm: %.3f  recall at prec_80: %.3f' % ( epoch,step,auc_score,auc_score2,accuracy1,accuracy2,rp80 ))








