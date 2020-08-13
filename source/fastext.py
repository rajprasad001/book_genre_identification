import fasttext
import pandas as pd

#model = fasttext.train_unsupervised('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/pro-test-text.txt', minn=2, maxn=10, dim=100, model='cbow',epoch =50, verbose=2)
#model.save_model('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/test.bin')

'''
df=pd.read_csv('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/book_clean.csv')

model = fasttext.load_model('C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/train.bin')

list_content = list(df['content2'])
with open("C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/book_cbow_train.tsv", 'w',encoding="utf-8") as f:
     for item in list_content:
         vec_pagetitle=model.get_sentence_vector(str(item))
         for value in vec_pagetitle:
             f.write("%s\t" % value)
         f.write("\n")
'''

df1 = pd.read_csv("C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/book_cbow_train.tsv", sep='\t', header=None)
df2 = pd.read_csv("C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/book_clean.csv", usecols=(['genre']))
df2.drop(df2.columns.difference(['genre']), 1, inplace=True)
df1.drop(columns=[100], inplace=True)
df=  pd.concat([df2.reset_index(drop=True), df1.reset_index(drop=True)], axis= 1)
# Our first dataset is now stored in a Pandas Dataframe
df.to_csv("C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/Data/bookdf_cbow_train.csv", index=None)
#Here we check the schema, and its length
print("Schema: "+str(df.columns))
print("Number of rows: "+str(len(df)))
print(df.head())
