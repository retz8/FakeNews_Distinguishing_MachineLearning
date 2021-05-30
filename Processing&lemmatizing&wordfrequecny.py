import pandas as pd
import nltk
import string
import numpy as np
from collections import Counter
import operator

from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
data_path= "C:/Users/ekkic/Desktop/Programming/phyton programming/fakenewsProject/data/Fake.csv"
data = pd.read_csv(data_path, encoding="UTF-8")

def lower(dataf):
        dataf["text"] = dataf["text"].str.low()
def tokens_word(dataf):
        new_dataf = dataf["text"]
        tokens = nltk.word_tokenize(new_dataf)
        token_words = [word for word in tokens if word.isalpha()]
        return token_words
stemming = PorterStemmer()
def stem_list(datf):# delete -ed, -s
        new_dataf = datf["words"] 
        stemmed_list = [stemming.stem(word) for word in new_dataf] #stemming and store in to list
        return stemmed_list
lemmatizer = WordNetLemmatizer()
def lemmatizes (dataf):
        new_dataf = dataf["meaningful"]
        lemmatized_list = [lemmatizer.lemmatize(word, pos='v') for word  in new_dataf]
        #print (lemmatized_list)
        return lemmatized_list
stop = set(stopwords.words('english'))
stop.append("reuters") 
def remove_stop (dataf):
        new_dataf = dataf["words"]
        meaningful = [word for word in new_dataf if not word in stop]
        return meaningful
def join_words(dataf):
        new_dataf = dataf['words_lemmatized']
        joined_words = " ".join(new_dataf) # list = ['my, 'name', 'scott'] --> my name scott
        return joined_words
def word_frequency(da):
        colnames = [ "title","subject","date","body"]
        #true_processed 저장경로
        da = pd.read_csv("C:/Users/ekkic/Documents/GitHub/data_anlysis_fake_telos/True_processed.csv",names = colnames ,encoding = "UTF-8")
        da2 = da.body.tolist()
        da3 = list()
        for i in range(0,len(da2)):
                if type(da2[i]) == str:
                       da3.append(da2[i])
            
    #vocab  = list(set(w for sen in da2 for w in sen.split()))
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = word_tokenize,ngram_range=(1,1), min_df=1)
        vocab =  ' '.join(da3)
        x= vectorizer.fit_transform(vocab.split("\n"))
        vocab2 = list(vectorizer.get_feature_names())
        counts = x.sum(axis=0).A1
        #counter = Counter(chain.from_iterable(map(str.split,dataf.body.tolist())))
        dictionary = Counter(dict(zip(vocab2,counts)))
        print (dictionary)
        return dictionary

data['text'] = data['text'].str.lower()
data['words'] = data.apply(tokens_word, axis=1) #lowercase, tokenize.
#data['words_stemmed'] = data.apply(stem_list, axis=1) #lowercase, tokenize, stemmed.
data['meaningful'] = data.apply (remove_stop, axis=1) # lowercase, tokenize, stemmed, removed stop words
data["words_lemmatized"]=data.apply(lemmatizes,axis=1)
data['body'] = data.apply(join_words, axis=1) #lowercase, tokenize, stemmed, removed stop words, and joined.
columns_drop = ["words", "words_lemmatized", "meaningful", "text"]
data.drop(columns = columns_drop, inplace=True)
data.to_csv('Fake_processed.csv', index=False)
D  =word_frequency(data)
print(D)







        
