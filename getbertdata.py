import pandas as pd
import numpy as np


pd.set_option('display.max_columns',None)
adm_file='./mimic-iii-clinical-database-1.4/ADMISSIONS.csv'
evt_file='./mimic-iii-clinical-database-1.4/NOTEEVENTS.csv'

adm_df=pd.read_csv(adm_file)

#d1=adm_df.groupby('SUBJECT_ID').agg({'SUBJECT_ID':'count'})
#dup_ids=list(d1[d1.SUBJECT_ID>2].index)[:5]
#adm_df=pd.concat([adm_df.head(5),adm_df[adm_df.SUBJECT_ID.isin( dup_ids)]],ignore_index=True)

adm_df.ADMITTIME=pd.to_datetime(adm_df.ADMITTIME,format='%Y-%m-%d %H:%M:%S')
adm_df.DISCHTIME=pd.to_datetime(adm_df.DISCHTIME,format='%Y-%m-%d %H:%M:%S')
adm_df.sort_values(['SUBJECT_ID','ADMITTIME'])


adm_df['NEXT_ADMITTIME']=adm_df.groupby(['SUBJECT_ID']).ADMITTIME.shift(-1)
adm_df['NEXT_ADMISSION_TYPE']=adm_df.groupby(['SUBJECT_ID']).ADMISSION_TYPE.shift(-1)

####get rid of elective readmission
adm_df.loc[adm_df.NEXT_ADMISSION_TYPE=='ELECTIVE','NEXT_ADMITTIME']=pd.NaT
adm_df.loc[adm_df.NEXT_ADMISSION_TYPE=='ELECTIVE','NEXT_ADMITTYPE']=np.NaN
adm_df[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']]=adm_df.groupby('SUBJECT_ID')[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method='bfill')



adm_df['Label']=(adm_df.NEXT_ADMITTIME-adm_df.DISCHTIME).dt.total_seconds()/24/60/60<30 ##58976
##filter out newborn and death
adm_df=adm_df[adm_df.ADMISSION_TYPE !='NEWBORN'] #51113

adm_df=adm_df[adm_df.DEATHTIME.isnull()] #45321


###get notes
evt_df=pd.read_csv(evt_file) #2083180
evt_df.CHARTDATE=pd.to_datetime(evt_df.CHARTDATE,format='%Y-%m-%d %H:%M:%S')
evt_df.sort_values(['SUBJECT_ID','HADM_ID','CHARTDATE'])
adm_df=pd.merge(adm_df[['SUBJECT_ID','HADM_ID','Label']],
                evt_df[['SUBJECT_ID','HADM_ID','CHARTDATE','CATEGORY','TEXT']],
                how='left',on=['SUBJECT_ID','HADM_ID']) #1202336


###discharge summary
discharge_df=adm_df[adm_df.CATEGORY=='Discharge summary'] #50072
discharge_df=discharge_df.groupby(['SUBJECT_ID','HADM_ID']).nth(-1).reset_index() #43880

###preprocess discharge notes

import re
def preprocess(df,chunksize):
    def subre(text):
        text=text.strip()
        text=text.lower()
        text=re.sub('\[.*\]','',text)
        text=re.sub('\n','',text)
        text=re.sub('\r','',text)
        text=re.sub('[0-9]+?\.','',text)
        text=re.sub('dr\.','doctor',text)
        text=re.sub('m\.d\.','md',text)
        text=re.sub('admission date:','',text)
        text=re.sub('discharge date:','',text)
        text=re.sub('__|--|==','',text)
        return text
    df.TEXT=df.TEXT.fillna(' ')
    df.TEXT=df.TEXT.apply(lambda x: subre(x))
    from tqdm import tqdm
    #res=pd.DataFrame({'SUBJECT_ID':[],'HADM_ID':[],'Label':[],'TEXT':[]})
    res=pd.DataFrame(columns=df.columns)
    for i in tqdm(range(len(df.TEXT))):
        x=df.TEXT.iloc[i].split()
        num=len(x)//chunksize
        subj_id=df.SUBJECT_ID.iloc[i]
        hadm_id=df.HADM_ID.iloc[i]
        curlab=df.Label.iloc[i]
        for j in range(num):
            res=res.append({'SUBJECT_ID':subj_id,'HADM_ID':hadm_id,'Label':curlab,'TEXT':' '.join(x[j*chunksize:(j+1)*chunksize])},ignore_index=True)
        if len(x) % chunksize >50:
            res=res.append({'SUBJECT_ID':subj_id,'HADM_ID':hadm_id,'Label':curlab,'TEXT':' '.join(x[-(len(x) % chunksize):])},ignore_index=True)
    return res

discharge_df=preprocess(discharge_df,400)





###get train data

adm_ids=discharge_df[adm_df.Label==1].HADM_ID #3095
no_adm_ids=discharge_df[adm_df.Label==0].HADM_ID #40785


train_adm_ids=adm_ids.sample(frac=0.8,random_state=1)
train_no_adm_ids=no_adm_ids.sample(frac=0.8,random_state=1)

test_eval_adm_ids=adm_ids.drop(train_adm_ids.index)
test_eval_no_adm_ids=no_adm_ids.drop(train_no_adm_ids.index)

test_adm_ids=test_eval_adm_ids.sample(frac=0.5,random_state=1)
test_no_adm_ids=test_eval_no_adm_ids.sample(frac=0.5,random_state=1)

val_adm_ids=test_eval_adm_ids.drop(test_adm_ids.index)
val_no_adm_ids=test_eval_no_adm_ids.drop(test_no_adm_ids.index)


train_data=pd.concat([discharge_df.loc[discharge_df.HADM_ID.isin(train_adm_ids),['SUBJECT_ID','HADM_ID','Label','TEXT']],
                        discharge_df.loc[discharge_df.HADM_ID.isin(train_no_adm_ids),['SUBJECT_ID','HADM_ID','Label','TEXT']] ],ignore_index=True)
val_data=pd.concat([discharge_df.loc[discharge_df.HADM_ID.isin(val_adm_ids),['SUBJECT_ID','HADM_ID','Label','TEXT']],
                      discharge_df.loc[discharge_df.HADM_ID.isin(val_no_adm_ids),['SUBJECT_ID','HADM_ID','Label','TEXT']] ],ignore_index=True)
test_data=pd.concat([discharge_df.loc[discharge_df.HADM_ID.isin(test_adm_ids),['SUBJECT_ID','HADM_ID','Label','TEXT']],
                      discharge_df.loc[discharge_df.HADM_ID.isin(test_no_adm_ids),['SUBJECT_ID','HADM_ID','Label','TEXT']] ],ignore_index=True)

###write to file

train_data.to_csv('./data/train_discharge_hua.csv',index=False)
val_data.to_csv('./data/val_discharge_hua.csv',index=False)
test_data.to_csv('./data/test_discharge_hua.csv',index=False)



