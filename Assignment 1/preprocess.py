
# coding: utf-8

# The following file removes
#  1. HTMLs
#  2. Trailing HashTags (still to be implemented)
#  3. Any mentions, special characters
# 


import pandas as pd
import os
import sklearn as skl
import nltk
import re


def readData(filename):    
    cwd = os.getcwd()
    path = cwd + "/" + filename;
    df =pd.read_csv(path);
    return df


def cleanhtml(tweet):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', tweet)
  return cleantext
def cleanUrl(tweet):
    tweet= re.sub(r"http\S+", "",  tweet)
    return tweet; 
def removeMention(tweet):
    tweet = tweet.replace("@","").rstrip() 
    return tweet;
def removeTrailingHash(tweet):
    if len(tweet.split()) ==1:
        return tweet;
    ends_with_hash=tweet.rsplit(' ', 1)[1].startswith("#")
    while(ends_with_hash):
        tweet=tweet.rstrip().rsplit(' ', 1)[0] 
        split_tweet = tweet.rsplit(' ',1)
        ends_with_hash=len(split_tweet) >1
        if(ends_with_hash):
            ends_with_hash = ends_with_hash & split_tweet[1].startswith("#")
    return tweet;

def preprocess(filename):
    df = readData(filename)
    df['text']=df['text'].apply(cleanhtml).apply(cleanUrl).apply(removeMention).apply(removeTrailingHash);
    tweetList = df['text']
    return df

#to test
#filename = "clinton-50k.csv"
#df = preprocess(filename)


