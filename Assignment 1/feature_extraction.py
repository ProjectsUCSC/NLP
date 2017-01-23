from preprocess import *
import nltk
from nltk.util import ngrams
from collections import Counter
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import re
from nltk.stem.porter import *
import sys
TFIDF = 1
UNI = 0

#feature extraction - TFIDF and unigrams
def vectorize(preprocessed_data_sample, flag):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    no_features = 1000#500#806#150#800#600#350

    #ngram_range=(1, 1)
#    input=u'content', encoding=u'utf-8', decode_error=u'strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer=u'word', stop_words=None, token_pattern=u'(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<type 'numpy.int64'>, norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False

    if flag:
        vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, vocabulary = None, ngram_range=(1,1), strip_accents=None, max_features = no_features)#, ngram_range=(2,2)) 
    else:
        vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, vocabulary = None, ngram_range=(1,1), strip_accents=None, max_features = no_features)
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

#Stop word removal
def tokenize_and_stopwords(data_sample):
    #data_sample = list(data_sample)
    #Get all english stopwords
    stop = stopwords.words('english')# + list(string.punctuation) + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    #Use only characters from reviews
    data_sample = data_sample.str.replace("[^a-zA-Z ]", " ")#, " ")
    #print data_sample
    #tokenize and remove stop words
    return [[i for i in word_tokenize(sentence) if i not in stop] for sentence in data_sample]
    
def stemmer(preprocessed_data_sample):
    print "stemming "
    #Create a new Porter stemmer.
    stemmer = PorterStemmer()
    #try:
    for i in range(len(preprocessed_data_sample)):
        #Stemming
        try:
            preprocessed_data_sample[i] = " ".join([stemmer.stem(str(word)) for word in preprocessed_data_sample[i]])
        except:
        #No stemming
            preprocessed_data_sample[i] = " ".join([str(word) for word in preprocessed_data_sample[i]])
    return preprocessed_data_sample
    

filename = "clinton-50k.csv"

df = preprocess(filename)
print "length of data", len(df)

data_sample = df['text'].str.lower()

#TFIDF features without text processing
[data_tfidf, vectorizer, no_features] = vectorize(data_sample, TFIDF) 
#Unigram features without text processing
[data_uni, vectorizer, no_features] = vectorize(data_sample, UNI) 

#print data_tfidf.shape
#for i in data_uni:
#    print sum(i)

#Text preprocessing - stopwords, stemming, 
preprocessed_data_sample = tokenize_and_stopwords(data_sample)
preprocessed_data_sample = stemmer(preprocessed_data_sample)

print preprocessed_data_sample[0:5]
#use CMU tagger and remove NNP and NNPS

#----------Dont see below----------------------------------
##All tweets combined together
#tweets = ""
#for index, row in df.iterrows():
#    tweets += " " + row["text"]
#    
#print tweets[0:500]
#print len(tweets)


#token = nltk.word_tokenize(tweets)
#unigrams = [i[0] for i in list(ngrams(token,1))]
#bigrams = [i[0] + " " + i[1] for i in list(ngrams(token,2))]

##trigrams = ngrams(token,3)
##fourgrams = ngrams(token,4)
##fivegrams = ngrams(token,5)
##print len(unigrams)
#print bigrams
##print len(bigrams)
##print bigrams
