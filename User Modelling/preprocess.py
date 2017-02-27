import pandas as pd
from pandas import DataFrame
import os
from nltk.corpus import stopwords
import re
import enchant
from nltk.stem.porter import *
import numpy as np
import cPickle as pickle
from collections import Counter
from keras.models import model_from_json
import math
import signal
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from scipy.sparse import csr_matrix
K.set_image_dim_ordering('th')


X_train = []
Y_train = []

def handler(signum, frame):
    print 'Ctrl+Z pressed'
    assert False
signal.signal(signal.SIGTSTP, handler)

# In[2]:

def readData(filename1, filename2):
    cwd = os.getcwd()
    req_attributes = ['tweet_id', 'topic', 'sentiment', 'tweet', 'user_id']#, 'followers_count', 'statuses_count', 'description', 'friends_count', 'location']
    user_req_attributes = ['tweet_id', 'user_id']#, 'followers_count', 'statuses_count', 'description', 'friends_count', 'location']
    tweet_req_attributes = ['tweet_id', 'topic', 'sentiment', 'tweet']
    path = cwd + "/data/" + filename1;
    tweet_df = pd.read_csv(path, sep='\t');
    
    tweet_df = tweet_df.drop_duplicates(['tweet_id'])
    tweet_df = tweet_df[tweet_req_attributes]

    path = cwd + "/data/" + filename2;
    user_df = pd.read_csv(path, sep='\t')
    user_df = user_df.dropna(subset=['user_id'])
    user_df = user_df[user_req_attributes]
    
    data = tweet_df.merge(user_df, left_on="tweet_id", right_on="tweet_id", how="inner")
    data = data.drop_duplicates(['tweet_id'])
    data = data.dropna(subset=['user_id', 'tweet'])
    data = data.dropna(subset=['user_id'])
    print len(user_df), len(tweet_df), len(data)
    
    print "From user data\n", Counter(list(data["user_id"])).most_common(50)
    return data[req_attributes]


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
    print "words", words
    abb_dict = pickle.load(open("abbreviations", "r"))
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
    
def preprocess(filename1, filename2):
    #filename = "Homework2_data.csv"
    df = readData(filename1, filename2)
    print "from joined data\n", Counter(list(df["user_id"])).most_common(50)
    indices = []
#    df['tweet'] = df['tweet'].apply(cleanhtml).apply(cleanUrl).apply(removeMention).apply(removeTrailingHash);

    df['tweet'] = df['tweet'].apply(cleanhtml).apply(cleanUrl)#.apply(removeTrailingHash);
    df['tweet'] = tokenize_and_stopwords(df['tweet'])
    data = DataFrame(df.groupby('topic')['tweet'].apply(list)).reset_index()

    for i in range(len(data)):
        data['tweet'][i] = " ".join(data['tweet'][i])
    
    topics = list(data["topic"])
#    Watch out
    topics = topics[0:10]
#    Word topic mapping
    try:
        word_dict = pickle.load(open("word_dict", "r"))
    except:
        tweets = ""
        for index, i in data.iterrows():
#        for i in data['tweet']:
            if i['topic'] in topics:
                tweets += str(i['tweet'])
        
        word_dict = {}
        tweets = tweets.split()
        for word in tweets:
            word_dict[word] = []
            for i in range(len(topics)):
                if word in data["tweet"][i]:
                    word_dict[word].append(topics[i])        
        pickle.dump(word_dict, open("word_dict", "wb"))
    
    print "the word 'election' is present in", (word_dict['register'])
    print len(word_dict)
#    Word model
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        [X_train, Y_train] = pickle.load(open("train_data", "r"))
        
    except:
        [model, X_train, Y_train] = train_cnn(word_dict, topics)
        
        pickle.dump([X_train, Y_train], open("train_data", "wb"))
        try:
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("model.h5")
            print("Saved model to disk")
        except Exception as e:
            print "ex", e
            try:
                pickle.dump(model, open("model", "wb"))
            except:
                print "dumping failed"
    
#    Prediction
    y = model.predict(X_train)#, Y_train, batch_size=32, verbose=1, sample_weight=None)
    diff = abs(y - Y_train)
##    for d in diff:
##        print d
##    Error
    
    
#    Hidden state
    get_last_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[6].output])

# output in train mode = 0
    layer_output = get_last_layer_output([X_train[0:1000], 0])[0]
    print layer_output
    print len(layer_output[0])
    print sum(sum(diff)) / len(X_train) * 1.0


def train_cnn(word_dict, topics):

    try:
        X_train = []#np.array([])
#        Y_train = []#np.array([])
        global X_train, Y_train
        
    #    Encoding X_train and Y_train
        vocab = word_dict.keys()
        print "length of vocab is ", len(vocab)
        print "before"
#        print word_dict
#        print vocab
#        print topics
        for i in range(len(vocab)):
            c = np.zeros(len(topics))
            c_x = np.zeros(len(vocab))
#            for i in range(len(topics)):
#                print topics[i], word_
            locations = np.array([j for j in range(len(topics)) if topics[j] in word_dict[vocab[i]]])
#            print locations
            c_x[i] = 1
            X_train.append(c_x)# = np.append(X_train, c_x)
            try:#buggy, fews words aren't found in any topics, weird, space removed by mistake.'
                c[locations] = 1
#                print len(c_x)
                Y_train.append(c)# = np.append(Y_train, c)
#                print "atleast"
    #            print locations
            except:
                print "couldn't find", vocab[i]
                Y_train.append(np.zeros(len(topics)))
#                print "l is",# locations, word 
             #This could be buggy
        
            
    #    print len(X_train)
    #    print X_train[0]
    #    Main line
    #    X_train = csr_matrix(np.array(X_train))#
        X_train = np.array(X_train).astype('float16')
    #    Y_train = csr_matrix(Y_train)#
        Y_train = np.array(Y_train).astype('float16')
        print "after"
#        print X_train
#        print Y_train
        X_train = X_train.reshape(len(X_train), 1, len(X_train[0]), 1)#(32088, 1, 32088, 1)#
        print X_train.shape
    #    print "first shape", X_train.shape
        print X_train[0]

        print "Shape sir is", Y_train.shape
        cnn = Sequential()
        cnn.add(Convolution2D(64, 3, 1,
            border_mode="same",
            activation="relu",
            input_shape=(1, 3657, 1)))
        cnn.add(Convolution2D(64, 3, 1, border_mode="same", activation="relu"))
        cnn.add(MaxPooling2D(pool_size=(2, 1)))

#        cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
#        cnn.add(Convolution2D(64, 3, 1, border_mode="same", activation="relu"))
##        cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
#        cnn.add(MaxPooling2D(pool_size=(2, 1)))
            
        cnn.add(Convolution2D(32, 3, 1, border_mode="same", activation="relu"))
#        cnn.add(Convolution2D(32, 3, 1, border_mode="same", activation="relu"))
#        cnn.add(Convolution2D(64, 3, 1, border_mode="same", activation="relu"))
        cnn.add(MaxPooling2D(pool_size=(2, 1)))
            
        cnn.add(Flatten())
        cnn.add(Dense(100, activation="linear"))
        cnn.add(Dropout(0.5))
    #    cnn.add(Dense(3, activation="softmax"))
        cnn.add(Dense(len(topics), activation="softmax"))
        # define optimizer and objective, compile cnn

#        cnn.compile(loss="categorical_crossentropy", batch_size=32, optimizer="adam")

##         train

#        cnn.fit(X_train, Y_train, nb_epoch=20, show_accuracy=True)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        cnn.compile(loss='categorical_crossentropy', optimizer=sgd, batch_size=32)
    #    print X_train[0:5]
    #    print Y_train[0:5]
        cnn.fit(X_train, Y_train, batch_size=32, nb_epoch=1)
        print "Done training, returning model"
        return [cnn, X_train, Y_train]
    except KeyboardInterrupt:
        del X_train
        del Y_train
        print "deleted"

    except AssertionError:
        del X_train
        del Y_train
        print "deleted"


def main():
#    import gc
#    gc.collect()
    global X_train, Y_train
    try:
        filename1 = "tweets.txt"#twitter-2016dev-CE-output.txt_semeval_tweets.txt"
        filename2 = "users.txt"#"twitter-2016dev-CE-output.txt_semeval_userinfo.txt"
        preprocess(filename1, filename2)
    except KeyboardInterrupt:
        del X_train
        del Y_train
        print "deleted"
        
    except AssertionError:
        del X_train
        del Y_train
        print "deleted"
    
main()
