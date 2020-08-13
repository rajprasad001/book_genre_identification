import pandas as pd
import textstat


df = pd.read_csv('full_data.csv')
#df = pd.read_csv('dummy data2.csv')
genre_list = []
for items in df['genre']:
    if items in genre_list:
        pass
    else:
        genre_list.append(items)
print(genre_list)

class_dict ={}

instance_class_count = 0
for genre in genre_list:

    target_label = genre

    for target in df['genre']:
        if genre == target:
            instance_class_count = instance_class_count +  1
    class_dict[genre] = (instance_class_count)
    instance_class_count = 0

print(class_dict)


det_and_mys_dict ={}

def word_count(class_name):
    total_word = 0
    for idx, items in enumerate(df['genre']):
        if items == class_name:
            book_id = df.at[idx,'book_id']
            content = df.at[idx,'content']
            no_of_words = textstat.lexicon_count(str(content),removepunct=True)
            total_word = total_word + no_of_words
            det_and_mys_dict[book_id]=(no_of_words)

    min_word_id = min(det_and_mys_dict.items(),key=lambda x: x[1])
    max_word_id = max(det_and_mys_dict.items(),key=lambda x: x[1])
    return (min_word_id, max_word_id,total_word)
    #print(min_word_id)


#for class_label in genre_list:
min_id_book, max_id_book, total_words = word_count('Literary')
#no_of_instance = class_dict.get('Literary')
avg_words = total_words/795
print('Literary')
print('Book_id with minimum Word Count : ' + str(min_id_book))
print('Book_id with maximum Word Count : '+ str(max_id_book))
#print('Total No of Intances : {}'.format(no_of_instance))
print('Avg_words : {}'.format(avg_words))

print('_'*25)



#det_and_mys_id = word_count('Detective and Mystery')
#print('Detective and Mystery')
#print(det_and_mys_id)

#det_and_mys_id = word_count('Literary')
#print('Literary')
#print(det_and_mys_id)
'''
det_and_mys_id = word_count('Western Stories')
print('Western Stories')
print(det_and_mys_id)

det_and_mys_id = word_count('Ghost and Horror')
print('Ghost and Horror')
print(det_and_mys_id)

det_and_mys_id = word_count('Sea and Adventure')
print('Sea and Adventure')
print(det_and_mys_id)

det_and_mys_id = word_count('Christmas Stories')
print('Christmas Stories')
print(det_and_mys_id)

det_and_mys_id = word_count('Love and Romance')
print('Love and Romance')
print(det_and_mys_id)

det_and_mys_id = word_count('Allegories')
print('Allegories')
print(det_and_mys_id)

det_and_mys_id = word_count('Humorous and Wit and Satire')
print('Humorous and Wit and Satire')
print(det_and_mys_id)

det_and_mys_id = word_count('Unlabelled')
print('Unlabelled')
print(det_and_mys_id)
'''