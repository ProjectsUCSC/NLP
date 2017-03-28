import pandas as pd
from pandas import DataFrame
from keras.callbacks import History
from itertools import dropwhile
import gensim.models.word2vec as wv 
history = History()
import os
from nltk.corpus import stopwords
import re
#import enchant
from nltk.stem.porter import *
import numpy as np
np.random.seed(1337)
import cPickle as pickle
from collections import Counter
from keras.models import model_from_json
import math
import signal
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.constraints import maxnorm
from keras import backend as K
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
import codecs
import pylab as plot
import random
from sklearn.utils import shuffle
import CMUTweetTagger
import nltk
K.set_image_dim_ordering('th')

genre = np.array(['tech', 'politics', 'music', 'sports'])
tech = np.array(['@microsoft', 'microsoft', 'twitter', 'nokia', 'amazon', 'amazon prime', 'amazon prime day', 'apple', 'apple watch', 'ipad', 'iphone', 'ipod', 'oracle', 'ibm', 'nintendo', 'moto g', 'google', 'google+', 'ps4', 'netflix', 't-mobile', 'windows 10'])                        

politics = np.array(['ukip', 'angela merkel',  'bernie sanders', 'david cameron', 'donald trump', 'trump', 'tory', 'hillary', 'joe biden', 'michelle obama', 'obama', 'rahul gandhi', 'tony blair', 'boehner', 'tsipras'])

music = np.array(['bee gees', 'beyonce', 'bob marley', 'chris brown', 'david bowie', 'katy perry',  'ed sheeran', 'foo fighters', 'janet jackson', 'lady gaga', 'michael jackson',  'ac/dc', 'the vamps', 'iron maiden', 'rolling stone', 'jay-z', 'snoop dogg', 'nirvana'])

sports = np.array(['arsenal', 'barca', 'federer', 'floyd mayweather', 'hulk hogan', 'john cena', 'kris bryant', 'randy orton', 'real madrid', 'serena', 'messi', 'david beckham', 'rousey', 'super eagles', 'kane', 'red sox', 'white sox', 'chelsea', 'james franklin', 'billy cundiff', 'cundiff', 'tiger woods'])

tech = np.array(['@microsoft', 'ac/dc', 'amazon', 'amazon prime', 'amazon prime day', 'amy schumer', 'angela merkel', 'ant-man', 'apple', 'apple watch', 'arsenal', 'bad blood', 'barca', 'batman', 'bbc', 'bee gees', 'bentley', 'bernie sanders', 'beyonce', 'big brother', 'bob marley', 'bobby jindal', 'boko haram', 'briana', 'brock lesnar', 'caitlyn jenner', 'calibraska', 'carly fiorina', 'cate blanchett', 'charlie hebdo', 'chelsea', 'chris brown', 'chris evans', 'christians', 'chuck norris', 'conor mcgregor', 'curtis', 'dana white', 'dark souls', 'david beckham', 'david bowie', 'david cameron', 'david price', 'david wright', 'dean ambrose', 'digi', 'disneyland', 'donald trump', 'dunkin', 'dustin johnson', 'ed sheeran', 'eid', 'erdogan', 'eric church', 'federer', 'fleetwood mac', 'floyd mayweather', 'foo fighters', 'frank gifford', 'frank ocean', 'galaxy note', 'game of thrones', 'gay', 'george harrison', 'george osborne', 'google', 'google+', 'grateful dead', 'gucci', 'hannibal', 'harper', 'harry potter', 'hillary', 'hulk hogan', 'ibm', 'ice cube', 'ihop', 'ios', 'ipad', 'iphone', 'ipod', 'ira', 'iran', 'iron maiden', 'islam', 'israel', 'janet jackson', 'jason aldean', 'jay-z', 'jeb bush', 'joe biden', 'john cena', 'john kasich', 'josh hamilton', 'jurassic park', 'jurassic world', 'justin', 'justin bieber', 'juventus', 'kane', 'kanye west', 'katy perry', 'kendrick', 'kendrick lamar', 'kerry', 'kim kardashian', 'kpop', 'kris bryant', 'kurt cobain', 'labor day', 'lady gaga', 'lexus', 'madonna', 'magic mike xxl', 'mariah carey', 'messi', 'metlife', 'michael jackson', 'michelle obama', 'milan', 'minecraft', 'miss usa', 'monsanto', 'moto g', 'murray', 'muslims', 'naruto', 'national hot dog day', 'national ice cream day', 'netflix', 'niall', 'nicki', 'nike', 'nintendo', 'nirvana', 'nokia', 'obama', 'oracle', 'paper towns', 'paul dunne', 'paul mccartney', 'planned parenthood', 'pope', 'pride parade', 'prince george', 'ps4', 'rahul gandhi', 'randy orton', 'real madrid', 'red sox', 'ric flair', 'rick perry', 'rolling stone', 'rousey', 'ryan braun', 'sam smith', 'sarah palin', 'saudi arabia', 'scott walker', 'scotus', 'seinfeld', 'serena', 'seth rollins', 'sharknado', 'shawn', 'snoop dogg', 'star wars day', 'super eagles', 'the vamps', 'thor', 'tom brady', 'tony blair', 'twilight', 'u2', 'watchman', 'white sox', 'yakub', 'yoga', 'zac brown band', 'zayn'])
politics = np.array([])
sports = np.array([])
music = np.array([])

filename1 = "tweets.txt"#twitter-2016dev-CE-output.txt_semeval_tweets.txt"

filename2 = "full-corpus.csv"#"twitter-2016dev-CE-output.txt_semeval_userinfo.txt"

req_pos_tags = ['N', 'S', 'Z', 'V', '^', 'A', 'R']
all_tags = {'N':0, 'O':1, '^':2, 'S':3, 'Z':4, 'V':5, 'L':6, 'M':7, 'A':8, 'R':9, '!':10, 'D':11, 'P':12, '&':13, 'T':14, 'X':15, 'Y':16, '#':17, '@':18, 'U':19, 'E':20, '$':21, ',':22, 'G':23, '~':24, 'NA':25}

def handler(signum, frame):
    print 'Ctrl+Z pressed'
    assert False
signal.signal(signal.SIGTSTP, handler)

# In[2]:

def get_word_frequency_lists(df, min_freq):

    words = " ".join(df['tweet'])
    counter =Counter(words.split())
    all_words_list  = counter.keys()

    for key, count in dropwhile(lambda key_count: key_count[1] >= min_freq, counter.most_common()):
       del counter[key]

    words_with_min_freq = counter.keys()
    words_del = list(set(all_words_list) - set(words_with_min_freq))

    return [all_words_list, words_with_min_freq, words_del]
    
def readData(filename1, filename2):
    global tech, politics, music, sports
    all_topics = np.concatenate((tech, politics, music, sports))
    cwd = os.getcwd()
#    req_attributes = ['tweet_id', 'topic', 'sentiment', 'tweet']#, 'user_id']#, 'followers_count', 'statuses_count', 'description', 'friends_count', 'location']
#    user_req_attributes = ['tweet_id', 'user_id']#, 'followers_count', 'statuses_count', 'description', 'friends_count', 'location']
    tweet_req_attributes = ['tweet_id', 'topic', 'sentiment', 'tweet']
    path = cwd + "/data/" + filename1;
    tweet_df = pd.read_csv(path, sep='\t');
    
    tweet_df = tweet_df.drop_duplicates(['tweet_id'])
    tweet_df = tweet_df[tweet_req_attributes]
#    print tweet_df['sentiment']

    tweet_df.ix[tweet_df['sentiment']=='positive', 'sentiment'] = 3#1.5
    tweet_df.ix[tweet_df['sentiment']=='neutral', 'sentiment'] = 0
    tweet_df.ix[tweet_df['sentiment']=='negative', 'sentiment'] = -3#1.5

    tweet_df['sentiment'] = pd.to_numeric(tweet_df['sentiment'], errors='coerce')
    tweet_df = tweet_df.dropna(subset=['sentiment'])
    
    tweet_df.ix[tweet_df['sentiment']==1, 'sentiment'] = 3#1.5
    tweet_df.ix[tweet_df['sentiment']==0, 'sentiment'] = 0
    tweet_df.ix[tweet_df['sentiment']==-1, 'sentiment'] = -3#3
    tweet_df.ix[tweet_df['sentiment']==-2, 'sentiment'] = -3
    tweet_df.ix[tweet_df['sentiment']==2, 'sentiment'] = 3
    
#    tweet_df

#    path = cwd + "/data/" + filename2;
#    tweet_df2 = pd.read_csv(path, sep=',')
#    tweet_df2 = tweet_df2.drop_duplicates(['tweet_id'])
#    tweet_df2 = tweet_df2[tweet_req_attributes]
##    tweet_df2['sentiment'] = pd.to_numeric(tweet_df2['sentiment'])

#    tweet_df2 = tweet_df2[tweet_df2['sentiment'] != 'neutral']#.ix[tweet_df2['sentiment']=='neutral', 'sentiment'] = 0
#    tweet_df2.ix[tweet_df2['sentiment']=='positive', 'sentiment'] = 3#1.5
#    tweet_df2.ix[tweet_df2['sentiment']=='negative', 'sentiment'] = -3#1.5
#    tweet_df2 = tweet_df2[tweet_df2['sentiment'] != 'irrelevant']
#    
#    frame = [tweet_df, tweet_df2]
#    data = pd.concat(frame)
   
 #Hack - beware!
    data = tweet_df
    print len(data)
#    data = tweet_df.merge(user_df, left_on="tweet_id", right_on="tweet_id", how="inner")
    data = data.drop_duplicates(['tweet_id'])
#    data = data.dropna(subset=['user_id', 'tweet'])
#    data = data.dropna(subset=['user_id'])
#    op = df['sentiment'] == df['sentiment']
#    df.ix[:, 'sentiment'] *= 2# genre[i]
    
#    Drop random neutrals

    data0 = data.query('sentiment == 0.0').sample(frac=.6, random_state=23)
    data3 = data.query('sentiment == 3.0').sample(frac=.6, random_state=23)
    
    print "l1", len(data)
    data = data.drop(data.query('sentiment == 0.0').sample(frac=.6, random_state=23).index)
    print "l2", len(data)
    data = data.drop(data.query('sentiment == 3.0').sample(frac=.6, random_state=23).index)
    print "l3", len(data)
    print "seed set"
    data = data.sample(frac=1.0, random_state=23)
    
#    print "From user data\n", Counter(list(data["user_id"])).most_common(50)
    #data = data[data["topic"].isin(all_topics)]
    print Counter(data['sentiment'])
    print "indexes reset"
    data = data.reset_index()
    data0 = data0.reset_index()
    data3 = data3.reset_index()
    print len(data)

    return [data, data0, data3]#[req_attributes]


def tokenize_and_stopwords(data_sample):

    print type(data_sample)
    print len(data_sample)
    #Get all english stopwords
    try:
        words = open("common_words.txt", "r").readlines()
        for i in range(len(words)):
            words[i] = words[i].strip()
    except: 
        words = []
    
#    print "words", words
#    abb_dict = pickle.load(open("abbreviations", "r"))
    stop = stopwords.words('english') + words #list(string.punctuation) + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #Use only characters from reviews
    data_sample = data_sample.str.replace("[^a-zA-Z ]", " ")#, " ")
    data_sample = data_sample.str.lower()
                
    return [(" ").join([i for i in sentence.split() if i not in stop]) for sentence in data_sample]

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
    
def load_embeddings(file_name):
 
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in 
f_in])
    wv = np.loadtxt(wv)
    return wv, vocabulary
    
#feature extraction - TFIDF and unigrams
def vectorize(preprocessed_data_sample):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
#    no_features = 1000#500#806#150#800#600#350

    #ngram_range=(1, 1)
#    input=u'content', encoding=u'utf-8', decode_error=u'strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer=u'word', stop_words=None, token_pattern=u'(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<type 'numpy.int64'>, norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False

    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, vocabulary = None, ngram_range=(1,1), strip_accents=None)#, max_features = no_features)#, ngram_range=(2,2))
    #vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(2,2), max_features = no_features)
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(preprocessed_data_sample)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()
    return [train_data_features, vectorizer, no_features]
