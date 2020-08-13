#from unittest.mock import inplace

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

chunked_train_data = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/train_chunk_data_complete.csv')
chunked_test_Data = pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/test_chunk_data_complete.csv')

unlabelled_data = chunked_train_data[chunked_train_data['genre'] == 'Unlabelled']
unlabelled_data.to_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/unlabelled_dat.csv', index = False)
chunked_train_data = chunked_train_data[chunked_train_data['genre'] != 'Unlabelled']

chunk_label = chunked_train_data.filter(['genre',])#1. train labels
temp = chunked_train_data.filter(['TTR', 'Positive_Sentiment','Neutral_Sentiment', 'Negative_Sentiment'])#2. add to scaled data
chunked_feature_data = chunked_train_data.drop(['book_id','genre', 'TTR', 'Positive_Sentiment','Neutral_Sentiment', 'Negative_Sentiment'],axis=1)#2. scale it and then add temp


def scaling_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return (scaled_data)

def syn_data(x,y,strategy):
    sm = SMOTE(random_state=None, sampling_strategy=strategy)
    x_train_with_syn, y_train_with_syn= sm.fit_sample(x, y)

    return (x_train_with_syn, y_train_with_syn)


x_train = pd.DataFrame(scaling_data(chunked_feature_data))
x_train = x_train.reset_index(drop=True)

chunk_label = chunk_label.reset_index(drop=True)
temp = temp.reset_index(drop=True)
chunk_data = pd.concat([temp,x_train], axis=1,sort=False)

chunk_label.to_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/chunk_label_without_syn.csv', index= False)
chunk_data.to_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/chunk_data_without_syn.csv',index=False)
#{'Literary':10447, 'Detective and Mystery':1344, 'Sea and Adventure': 415, 'Western Stories': 208, 'Love and Romance': 186, 'Humorous and Wit and Satire': 79,'Christmas Stories': 45, 'Ghost and Horror': 36, 'Allegories': 6}

strategy1 = {'Literary':10447, 'Detective and Mystery':1900, 'Sea and Adventure': 550, 'Western Stories': 300, 'Love and Romance': 260, 'Humorous and Wit and Satire': 110,
                         'Christmas Stories': 63, 'Ghost and Horror': 50, 'Allegories': 10}

x_train_syn, y_train_syn = syn_data(chunk_data,chunk_label,strategy1)



x_train_syn.to_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/x_train_syn.csv',index= False)
y_train_syn.to_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/y_train_syn.csv', index=False)