import pandas as pd
import os
from nltk.corpus import stopwords
import re
import enchant
from nltk.stem.porter import *
import numpy as np
import cPickle as pickle

# In[2]:

def readData(filename):
    cwd = os.getcwd()
    path = cwd + "/" + filename;
    #print path
    df =pd.read_csv(path);
    return df


def tokenize_and_stopwords(data_sample):
    #data_sample = list(data_sample)
    #Get all english stopwords
    try:
        words = open("common_words.txt", "r").readlines()
        for i in range(len(words)):
            words[i] = words[i].strip()
    except:
        words = []
    print "words", words
#    abb_dict = pickle.load(open("abbreviations", "r"))
    stop = stopwords.words('english') + words #list(string.punctuation) + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #Use only characters from reviews
    data_sample = data_sample.str.replace("[^a-zA-Z ]", " ")#, " ")
    data_sample = data_sample.str.lower()
    #print data_sample
    #tokenize and remove stop words
    
#    for i in range(len(data_sample)):
#        for j in data_sample[i].split():
#            if i == abb_dict.keys():
#                data_sample[i] = data_sample[i].replace(i, abb_dict[i])
                
    return [(" ").join([i for i in sentence.split() if i not in stop]) for sentence in data_sample]

# In[10]:

def cleanhtml(tweet):
      cleanr = re.compile('<.*?>')
      cleantext = re.sub(cleanr, '', tweet)
      return cleantext
def cleanUrl(tweet):
    tweet= re.sub(r"http\S+", "",  tweet)
    return tweet;
def removeMention(tweet):
    tweet = tweet.replace("rt@","").rstrip()
    tweet = tweet.replace("rt ","").rstrip()
    tweet = tweet.replace("@","").rstrip()
    return tweet;

# In[11]:

def spellCheck(word):
#    d = enchant.Dict()

#    if d.check(word) == False:
#        word =  d.suggest(word)[0] if d.suggest(word) else ""
#    #print word
    return word



def stemmer(preprocessed_data_sample):
    print "stemming "
    #Create a new Porter stemmer.
    stemmer = PorterStemmer()
    #try:
    for i in range(len(preprocessed_data_sample)):
        #Stemming
        try:
            preprocessed_data_sample[i] = preprocessed_data_sample[i].replace(preprocessed_data_sample[i], " ".join([stemmer.stem(str(word)) for word in preprocessed_data_sample[i].split()]))
        except:
        #No stemming
            preprocessed_data_sample[i] = preprocessed_data_sample[i].replace(preprocessed_data_sample[i], " ".join([str(word) for word in preprocessed_data_sample[i].split()]))
    return preprocessed_data_sample
    
# In[ ]:
def preprocess(filename):
	#filename = "Homework2_data.csv"
	df = readData(filename)
	df['text'] = df['Tweet_text'].apply(cleanhtml).apply(cleanUrl).apply(removeMention);#.apply(stemmer);
#	df['text'] = df['text'].apply(spellCheck)
#	df['text'] = stemmer(df["text"])
	df['text'] = tokenize_and_stopwords(df['text'])
	return df
