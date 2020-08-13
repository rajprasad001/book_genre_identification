import pandas as pd
import numpy as np
import nltk
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree
import textstat
from sklearn.model_selection import train_test_split

#__________________________Reading Full Data Frame____________________________________________________________________
complete_df= pd.read_csv('full_data.csv')

#__________________________Seperating Unlabelled Data_________________________________________________________________
col_df = ['book_id', 'content', 'genre']

progressive_id = 0
progressive_id2row_df = {}
column_def= ['book_id','content','genre']

for idx, item in enumerate(complete_df['genre']):
    if item=='Unlabelled':
        book_id = complete_df.at[idx, 'book_id']
        content = complete_df.at[idx, 'content']
        genre = complete_df.at[idx, 'genre']
        row = (book_id, content, genre)

        progressive_id2row_df.update({progressive_id: row})
        progressive_id += 1
        # no_of_words = textstat.lexicon_count(str(new_data), removepunct=True)

unlabelled_df = pd.DataFrame.from_dict(progressive_id2row_df, orient='index', columns=column_def)
unlabelled_df = unlabelled_df[pd.notnull(unlabelled_df['content'])] #concatenate this df to the train data later on

#_________________________Splitting Test Train Data___________________________________________________________________
genre_df = complete_df.filter(['genre'])
df_without_genre = complete_df.drop(['book_name','book_name', 'author_name','genre'],axis=1)

x_train, x_test, y_train, y_test = train_test_split(df_without_genre,genre_df, test_size=0.2, random_state=0)# Splitting the data
x_train_df = pd.DataFrame(x_train)
y_train_df = pd.DataFrame(y_train)
x_test_df = pd.DataFrame(x_test)
y_test_df = pd.DataFrame(y_test)

train_data = pd.concat([x_train_df,y_train_df],axis=1,sort=False)
test_data = pd.concat([x_test,y_test],axis=1,sort=False)

#_________________________Counting Data in Test and Train split_______________________________________________________
#x_train, x_test = train_test_split(complete_df,test_size=0.3,random_state=42)

#train_data = pd.DataFrame(x_train)
genre_dict={}
genre_list =[]
for item in train_data['genre']:
    if item in genre_list:
        pass
    else:
        genre_list.append(item)
#print(genre_list)

instance_class_count = 0
for genre in genre_list:

    target_label = genre

    for target in train_data['genre']:
        if genre == target:
            instance_class_count = instance_class_count +  1
    genre_dict[genre] = (instance_class_count)
    instance_class_count = 0

#________________________Merging Unlabelled data with Train and removing from Test Data_______________________________
train_data = train_data[train_data['genre'] != 'Unlabelled']
test_data = test_data[test_data['genre'] != 'Unlabelled']

frames = [train_data,unlabelled_df]
train_data_df = pd.concat(frames)
train_data_df.to_csv('train_data_split_200623.csv',index = False)
test_data.to_csv('test_data_split_200623.csv',index=False)

#________________________Creating Chunks on Training Data_____________________________________________________________
'''
def chunking(labels='Sea and Adventure'):
    progressive_id2 = 0
    progressive_id2row_df2 = {}
    for idx, items in enumerate(train_data_df['genre']):
        if items == labels:
            content = train_data_df.at[idx, 'content']
            chunks = content.split()
            per_chunk = 5000
            for i in range (0, len(chunks), per_chunk):
                new_data = " ".join(chunks[i:i + per_chunk])
                book_id = train_data_df.at[idx, 'book_id']
                content = new_data
                genre = labels
                row = (book_id, content, genre)
                progressive_id2row_df2.update({progressive_id2: row})
                progressive_id2 +=1
    chunked_data = pd.DataFrame.from_dict(progressive_id2row_df2, orient='index', columns=column_def)
    return (chunked_data)
'''




