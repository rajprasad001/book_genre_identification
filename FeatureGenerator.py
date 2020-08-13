import time
import pandas as pd
import spacy
import textstat
import re
import codecs
from nltk import sentiment
import nltk
nltk.download('vader_lexicon')

nlp = spacy.load(name="en_core_web_lg")
nlp.max_length = 2000000

class FeatureGenerator:
    'Feature Set generator'
    data = []
    list_MP = []
    list_FP = []
    list_PP = []
    list_PosP = []
    list_Punc = []
    list_Prep = []
    list_SubCon = []
    list_CCon = []
    sentimentAnalyser = ""

    def __init__(self,dataframe):
        self.data = dataframe
        print("Opening Files:")
        self.list_MP = open(r"./Res/LMP.txt").read().splitlines()
        self.list_FP = open(r"./Res/LFP.txt").read().splitlines()
        self.list_PP = open(r"./Res/PersonalPronoun.txt").read().splitlines()
        self.list_PosP = open(r"./Res/PossesivePronoun.txt").read().splitlines()
        self.list_Punc = open(r"./Res/punctuations.txt").read().splitlines()
        self.list_Prep = open(r"./Res/prepositions.txt").read().splitlines()
        self.list_SubCon = open(r"./Res/subordinatingconjunctions.txt").read().splitlines()
        self.list_CCon = open(r"./Res/Fanboys.txt").read().splitlines()
        self.sentimentAnalyser = sentiment.SentimentIntensityAnalyzer()
        print("Finished")
        try:
            print("Asserting Unique :")
            assert len(self.list_MP) == len(set(self.list_MP))
            assert len(self.list_FP) == len(set(self.list_FP))
            assert len(self.list_PP) == len(set(self.list_PP))
            assert len(self.list_PosP) == len(set(self.list_PosP))
            assert len(self.list_Punc) == len(set(self.list_Punc))
            assert len(self.list_Prep) == len(set(self.list_Prep))
            assert len(self.list_SubCon) == len(set(self.list_SubCon))
            assert len(self.list_CCon) == len(set(self.list_CCon))
            print("Assertion True")
        except AssertionError:
            print("Not All items in the files are unique, Please Check all input Files for Duplicacies")
            exit(-1)

    def removeHTMLTags(self, inp_string):
        reg_x = re.compile(pattern='<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        clean_text = re.sub(pattern=reg_x, repl='', string=str(inp_string))
        return clean_text

    def removePunctuations(self, inp_string):
        reg_x = re.compile(pattern='[^\w\s]')
        clean_text = re.sub(pattern=reg_x, repl='', string=str(inp_string))
        return clean_text

    def removeNumbers(self, inp_string):
        reg_x = re.compile(pattern='\d+')
        clean_text = re.sub(pattern=reg_x, repl='', string=str(inp_string))
        return clean_text

    def printDataframe(self):
        print(self.data.head())

    def computeParagraphCount(self):
        'do something here'
        col_FP = []
        for i in self.data['content'].iteritems():
            count_FP = 0
            count = (str(i).lower()).count(r"<p>")
            count_FP = count + count_FP
            col_FP.append(count_FP)
        self.data['Paragraphs'] = col_FP

    def addFemalePronounCount(self):
        'do something here'
        col_FP = []
        for i in self.data['content'].iteritems():
            t = self.removeHTMLTags(i[1])
            count_FP = 0
            for word in self.list_FP:
                count = (str(t).lower()).count(word)
                count_FP = count + count_FP
            col_FP.append(count_FP)
        self.data['Female_Orientation'] = col_FP

    def addMalePronounCount(self):
        'do something here'
        col_MP = []
        for i in self.data['content'].iteritems():
            count_MP = 0
            t = self.removeHTMLTags(i[1])
            for word in self.list_MP:
                count = (str(t).lower()).count(word)
                count_MP = count + count_MP
            col_MP.append(count_MP)
        self.data['Male_Orientation'] = col_MP

    def addPersonalPronounCount(self):
        'do something here'
        col_MP = []
        for i in self.data['content'].iteritems():
            count_MP = 0
            t = self.removeHTMLTags(i[1])
            for word in self.list_PP:
                count = (str(t).lower()).count(word)
                count_MP = count + count_MP
            col_MP.append(count_MP)
        self.data['Personal_Pronoun'] = col_MP

    def addPosessivePronounCount(self):
        'do something here'
        col_MP = []
        for i in self.data['content'].iteritems():
            count_MP = 0
            t = self.removeHTMLTags(i[1])
            for word in self.list_PosP:
                count = (str(t).lower()).count(word)
                count_MP = count + count_MP
            col_MP.append(count_MP)
        self.data['Possesive_Pronoun'] = col_MP

    def addPrepositionCount(self):
        'do something here'
        col_MP = []
        for i in self.data['content'].iteritems():
            t = self.removeHTMLTags(i[1])
            count_MP = 0
            for word in self.list_Prep:
                count = (str(t).lower()).count(word)
                count_MP = count + count_MP
            col_MP.append(count_MP)
        self.data['Prepositions'] = col_MP

    def addCoordinatinConjunctionCount(self):
        'do something here'
        col_MP = []
        for i in self.data['content'].iteritems():
            count_MP = 0
            t = self.removeHTMLTags(i[1])
            for word in self.list_CCon:
                count = (str(t).lower()).count(word)
                count_MP = count + count_MP
            col_MP.append(count_MP)
        self.data['Coord_Conjunctions'] = col_MP

    def addCommaCount(self):
        'do something here'
        col_FP = []
        for i in self.data['content'].iteritems():
            i = self.removeHTMLTags(i[1])
            count_FP = 0
            count = (str(i).lower()).count(r",")
            count_FP = count + count_FP
            col_FP.append(count_FP)
        self.data['Commas'] = col_FP

    def addPeriodCount(self):
        'do something here'
        col_FP = []
        for i in self.data['content'].iteritems():
            i = self.removeHTMLTags(i[1])
            count_FP = 0
            count = (str(i).lower()).count(r".")
            count_FP = count + count_FP
            col_FP.append(count_FP)
        self.data['Periods'] = col_FP

    def addColonCount(self):
        'do something here'
        col_FP = []
        for i in self.data['content'].iteritems():
            i = self.removeHTMLTags(i[1])
            count_FP = 0
            count = (str(i).lower()).count(r":")
            count_FP = count + count_FP
            col_FP.append(count_FP)
        self.data['Colons'] = col_FP

    def addSemiColonCount(self):
        'do something here'
        col_FP = []
        for i in self.data['content'].iteritems():
            i = self.removeHTMLTags(i[1])
            count_FP = 0
            count = (str(i).lower()).count(r";")
            count_FP = count + count_FP
            col_FP.append(count_FP)
        self.data['Semi_Colons'] = col_FP

    def addHyphenCount(self):
        'do something here'
        col_FP = []
        for i in self.data['content'].iteritems():
            i = self.removeHTMLTags(i[1])
            count_FP = 0
            count = (str(i).lower()).count(r"-")
            count_FP = count + count_FP
            col_FP.append(count_FP)
        self.data['Hyphens'] = col_FP

    def addInterjectionCount(self):
        'do something here'
        chars = []
        for each in self.data['content'].iteritems():
            people = 0
            each = self.removeHTMLTags(each[1])
            for token in nlp(each):
                people += 1 if (token.pos_ == "INTJ") else 0
            chars.append(people)
        self.data['Interjections'] = chars

    def addPunctuationSubordinatingCount(self):
        'do something here'
        col_MP = []
        for i in self.data['content'].iteritems():
            count_MP = 0
            t = self.removeHTMLTags(i[1])
            for word in self.list_Punc:
                count = (str(t).lower()).count(word)
                count_MP = count + count_MP
            col_MP.append(count_MP)

        idx = 0
        for i in self.data['content'].iteritems():
            count_SP = 0
            t = self.removeHTMLTags(i[1])
            for word in self.list_SubCon:
                count = (str(t).lower()).count(word)
                count_SP = count + count_SP
            col_MP[idx] = col_MP[idx] + (count_SP)
            idx += 1
        self.data['Punctuation_Subordinating'] = col_MP

    def addSentenceLengthCount(self):
        'do something here'
        avg_length = []
        for each in self.data['content'].iteritems():
            t = self.removeHTMLTags(each[1])
            sentence_length = 0
            sent_count = 1
            for sent in (nlp(t)).sents:
                sentence_length += len(sent.text)
                sent_count += 1
            if(sent_count==0):
                avg_length.append(0)
            else:
                avg_length.append(sentence_length / sent_count)
        self.data['Average_Sentence_Length'] = avg_length

    def addQuotesCount(self):
        'do something here'
        col_FP = []
        for i in self.data['content'].iteritems():
            i = self.removeHTMLTags(i[1])
            count_FP = 0
            count = (str(i)).count('\"')
            count_FP = count + count_FP
            col_FP.append(count_FP/2)
        self.data['Quotes'] = col_FP

    def addSentiments(self):
        'do something here'
        pos = []
        neg = []
        neu = []
        for each in self.data['content'].iteritems():
            each = self.removeHTMLTags(each[1])
            each = self.removeNumbers(each)
            each = self.removePunctuations(each)
            #score = (self.sentimentAnalyser).classify(each[1])
            score = self.sentimentAnalyser.polarity_scores(each)
            neg.append(score.get('neg'))
            pos.append(score.get('pos'))
            neu.append(score.get('neu'))
        self.data['Positive_Sentiment'] = pos
        self.data['Neutral_Sentiment'] = neu
        self.data['Negative_Sentiment'] = neg

    def addFleschReadingScore(self):
        'do something here'
        score = []
        for each in self.data['content'].iteritems():
            each = self.removeHTMLTags(each[1])
            score.append(textstat.flesch_reading_ease(each))
        self.data['Flesch_Score'] = score

    def addNoOfCharacters(self):
        'do something here'
        chars = []
        for each in self.data['content'].iteritems():
            people = []
            each = self.removeHTMLTags(each[1])
            for token in nlp(each).ents:
                if (token.label_ == "PERSON"):
                    people += token.text
            chars.append(len(set(people)))
        self.data['Characters'] = chars

    def addTTRRatio(self):
        'do something here'
        col_ttr = []
        for i in self.data['content'].iteritems():
            t = self.removeHTMLTags(i[1])
            b = str(t).split()
            total_count = len(b)
            unique_word = len(set(b))
            ttr = (unique_word / total_count)
            col_ttr.append(ttr)
        self.data['TTR'] = col_ttr

    def drop_content(self):
        self.data = (self.data).drop(['content'], axis = 1)

#init dataframe here
x = FeatureGenerator(pd.read_csv(r'C:/Users/anujp/Desktop/sort/semester 4/ATML/Sem project/atml_proj/testing_chunks.csv'))
print("Adding Sentiments")
x.addSentiments()
print("Adding Co-Ordinating Conjunctions")
x.addCoordinatinConjunctionCount()
print("Adding Interjections")
x.addInterjectionCount()
print("Adding People")
#x.addNoOfCharacters()
print("Adding Flesch")
x.addFleschReadingScore()
print("Adding Average Sentence Length")
x.addSentenceLengthCount()
print("Adding Male Orientation")
x.addMalePronounCount()
print("Adding Female Orientation")
x.addFemalePronounCount()
print("Adding Colons")
x.addColonCount()
print("Adding Commas")
x.addCommaCount()
print("Adding Hyphens")
x.addHyphenCount()
print("Adding Periods")
x.addPeriodCount()
print("Adding Personal Pronouns")
x.addPersonalPronounCount()
print("Adding Possesive Pronouns")
x.addPosessivePronounCount()
print("Adding Preposition")
x.addPrepositionCount()
print("Adding SemiColons")
x.addSemiColonCount()
print("Adding TTR")
#x.addTTRRatio()
print("Adding Paragraphs")
x.computeParagraphCount()
print("Adding Quotes")
x.addQuotesCount()

print("Trimming")
x.drop_content()

x.printDataframe()
print((x.data).describe())

t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
(x.data).to_csv("./Res/Export_" + timestamp + ".csv", header=True, index=None)
print("Exported to Res")