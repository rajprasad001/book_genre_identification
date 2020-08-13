import pandas as pd
from sklearn.model_selection import train_test_split

complete_df = pd.read_csv('full_data.csv')
# Compete_df is the dataframe with book:id, content, book_name, Author_name and Genre
# Drop Book_name, Author_name
complete_df = complete_df.drop(['book_name', 'author_name'],axis=1)

# Step 1:
# 1.a) First remove unlabelled data from complete dataframe
# 1.b) Check for any NaN in Unlabeled data. Remove if any

#2. Split Test Train Data
#2.a) Once Splitting is done, Concatenate Train and Unlabelled data
#2.b) Look for Number of data in Test and Train Set

#3. Chunking
#3.1 Create chunks for Train Data
#3.2 Create chunks for Test Data

# Step 1_______________________________________________________________________________________________

unlabelled_data = complete_df[complete_df['genre'] == 'Unlabelled']
unlabelled_df = unlabelled_data[pd.notnull(unlabelled_data['content'])]

# Step 2________________________________________________________________________________________________

complete_df = complete_df[complete_df['genre'] != 'Unlabelled']
x_train, x_test = train_test_split(complete_df,train_size=0.8,random_state=6)

train_df = pd.DataFrame(x_train)
test_df = pd.DataFrame(x_test)

merge_dfs = [train_df, unlabelled_df]
train_df = pd.concat(merge_dfs)

train_df.to_csv('train_df_200623.csv',index = False)
test_df.to_csv('test_df_200623.csv', index = False)

genre_dict={}
genre_list=[]

for item in train_df['genre']:
    if item in genre_list:
        pass
    else:
        genre_list.append(item)

instance_class_count = 0
for genre in genre_list:

    target_label = genre

    for target in train_df['genre']:
        if genre == target:
            instance_class_count = instance_class_count +  1
    genre_dict[genre] = (instance_class_count)
    instance_class_count = 0

# Step 3______________________________________________________________________________________________
training_data = pd.read_csv('train_df_200623.csv')
testing_data = pd.read_csv('test_df_200623.csv')

def chunking(df, labels):
    progressive_id = 0
    progressive_id2row_df = {}
    column_def = ['book_id', 'content', 'genre']
    for idx, items in enumerate(df['genre']):
        if items == labels:
            content = df.at[idx, 'content']
            chunks= content.split()
            per_chunk = 5000
            for i in range (0, len(chunks), per_chunk):
                new_data = " ".join(chunks[i:i + per_chunk])
                book_id = df.at[idx, 'book_id']
                content = new_data
                label = labels
                row = (book_id, content, label)
                progressive_id2row_df.update({progressive_id: row})
                progressive_id += 1


    chunked_data = pd.DataFrame.from_dict(progressive_id2row_df, orient= 'index', columns=column_def)
    return (chunked_data)

# Creating Chunks for Training Dataset
literary_chunk = chunking(training_data,'Literary')
allegory_chunk = chunking(training_data,'Allegories')
unlabelled_chunk = chunking(training_data,'Unlabelled')
wes_story_chunk = chunking(training_data,'Western Stories')
christmas_chunk = chunking(training_data,'Christmas Stories')
sea_and_adv_chunk = chunking(training_data,'Sea and Adventure')
love_and_roam_chunk = chunking(training_data,'Love and Romance')
det_and_mys_chunk = chunking(training_data,'Detective and Mystery')
ghost_and_horror_chunk = chunking(training_data,'Ghost and Horror')
hum_wit_sat_chunk = chunking(training_data,'Humorous and Wit and Satire')

concat_train_chunks = [literary_chunk,allegory_chunk,unlabelled_chunk,wes_story_chunk,christmas_chunk,sea_and_adv_chunk,
                 love_and_roam_chunk,det_and_mys_chunk,ghost_and_horror_chunk,hum_wit_sat_chunk]

chunked_train_df = pd.concat(concat_train_chunks)
chunked_train_df.to_csv('training_chunks.csv',index = False)


#Creating chunks for Testing Dataset

literary_chunk = chunking(testing_data,'Literary')
allegory_chunk = chunking(testing_data,'Allegories')
wes_story_chunk = chunking(testing_data,'Western Stories')
christmas_chunk = chunking(testing_data,'Christmas Stories')
sea_and_adv_chunk = chunking(testing_data,'Sea and Adventure')
love_and_roam_chunk = chunking(testing_data,'Love and Romance')
det_and_mys_chunk = chunking(testing_data,'Detective and Mystery')
ghost_and_horror_chunk = chunking(testing_data,'Ghost and Horror')
hum_wit_sat_chunk = chunking(testing_data,'Humorous and Wit and Satire')

concat_test_chunks = [literary_chunk,allegory_chunk,wes_story_chunk,christmas_chunk,sea_and_adv_chunk,
                 love_and_roam_chunk,det_and_mys_chunk,ghost_and_horror_chunk,hum_wit_sat_chunk]

chunked_test_df = pd.concat(concat_test_chunks)
chunked_test_df.to_csv('testing_chunks.csv', index = False)


train_genre_dict={}
test_genre_dict ={}
train_genre_list=[]
test_genre_list =[]

for item in chunked_train_df['genre']:
    if item in train_genre_list:
        pass
    else:
        train_genre_list.append(item)

instance_class_count = 0
for genre in train_genre_list:

    target_label = genre

    for target in chunked_train_df['genre']:
        if genre == target:
            instance_class_count = instance_class_count +  1
    train_genre_dict[genre] = (instance_class_count)
    instance_class_count = 0

for item in chunked_test_df['genre']:
    if item in test_genre_list:
        pass
    else:
        test_genre_list.append(item)

instance_class_count = 0
for genre in test_genre_list:

    target_label = genre

    for target in chunked_test_df['genre']:
        if genre == target:
            instance_class_count = instance_class_count +  1
    test_genre_dict[genre] = (instance_class_count)
    instance_class_count = 0

print('Splitting Criteria : 5000 words')
print('Training Inst/Genre after Split: ' + str(genre_dict))
print('Training chunks in each Genre  : ' + str(train_genre_dict))
print('Testing chunks in each Genre   : ' + str(test_genre_dict))
print('Shape of the Chunked Training Data : ' + str(chunked_train_df.shape))
print('Shape of the Chunked Testing Data  : ' + str(chunked_test_df.shape))
