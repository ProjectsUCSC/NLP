

import pandas as pd
import os
from nltk.corpus import stopwords
import re
import enchant



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
   stop = stopwords.words('english')# + list(string.punctuation) + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
   #Use only characters from reviews
   data_sample = data_sample.str.replace("[^a-zA-Z ]", " ")#, " ")
   #print data_sample
   #tokenize and remove stop words
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
    tweet = tweet.replace("@","").rstrip()
    return tweet;


# In[11]:

def spellCheck(word):
#    d = enchant.Dict()

#    if d.check(word) == False:
#        word =  d.suggest(word)[0] if d.suggest(word) else ""
#    #print word
    return word



# In[ ]:
def preprocess(filename):
	#filename = "Homework2_data.csv"
	df = readData(filename)
	df['text'] = df['Tweet_text'].apply(cleanhtml).apply(cleanUrl).apply(removeMention);
	#df['text'] = df['text'].apply(spellCheck)
	df['text'] = tokenize_and_stopwords(df['text'])
	
	return df




