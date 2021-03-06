#preprocess
from lib import *
#import pandas as pd
#from pandas import DataFrame
#import os
#from nltk.corpus import stopwords
#import re
##import enchant
#from nltk.stem.porter import *
#import numpy as np
#import cPickle as pickle
#from collections import Counter
#from keras.models import model_from_json
#import math
#import signal
#import h5py
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
#from keras.optimizers import SGD
#from keras.optimizers import Adam
#from keras import backend as K
#from scipy.sparse import csr_matrix
#from sklearn.manifold import TSNE
##import nltk
#import codecs
#import pylab as plot
#import random
#K.set_image_dim_ordering('th')


#X_train = []
#Y_train = []

#genre = np.array(['tech', 'politics', 'music', 'sports'])
#tech = np.array(['@microsoft', 'microsoft', 'twitter', 'nokia', 'amazon', 'amazon prime', 'amazon prime day', 'apple', 'apple watch', 'ipad', 'iphone', 'ipod', 'oracle', 'ibm', 'nintendo', 'moto g', 'google', 'google+', 'ps4', 'netflix'])                        

#politics = np.array(['angela merkel',  'bernie sanders', 'david cameron', 'donald trump', 'hillary', 'joe biden', 'michelle obama', 'obama', 'rahul gandhi', 'tony blair'])

#music = np.array(['bee gees', 'beyonce', 'bob marley', 'chris brown', 'david bowie', 'katy perry',  'ed sheeran', 'foo fighters', 'janet jackson', 'lady gaga', 'michael jackson',  'ac/dc', 'the vamps', 'iron maiden', 'rolling stone', 'jay-z', 'snoop dogg', 'nirvana'])

#sports = np.array(['arsenal', 'barca', 'federer', 'floyd mayweather', 'hulk hogan', 'john cena', 'kris bryant', 'randy orton', 'real madrid', 'serena', 'messi', 'david beckham', 'rousey', 'super eagles', 'kane', 'red sox', 'white sox'])
#genre_topics = [tech, politics, music, sports]


#def handler(signum, frame):
#    print 'Ctrl+Z pressed'
#    assert False
#signal.signal(signal.SIGTSTP, handler)

## In[2]:

#def readData(filename1, filename2):
#    cwd = os.getcwd()
##    req_attributes = ['tweet_id', 'topic', 'sentiment', 'tweet']#, 'user_id']#, 'followers_count', 'statuses_count', 'description', 'friends_count', 'location']
##    user_req_attributes = ['tweet_id', 'user_id']#, 'followers_count', 'statuses_count', 'description', 'friends_count', 'location']
#    tweet_req_attributes = ['tweet_id', 'topic', 'sentiment', 'tweet']
#    path = cwd + "/data/" + filename1;
#    tweet_df = pd.read_csv(path, sep='\t');
#    
#    tweet_df = tweet_df.drop_duplicates(['tweet_id'])
#    tweet_df = tweet_df[tweet_req_attributes]
##    print tweet_df['sentiment']
#    tweet_df['sentiment'] = pd.to_numeric(tweet_df['sentiment'], errors='coerce')
#    tweet_df = tweet_df.dropna(subset=['sentiment'])
#    
#    tweet_df.ix[tweet_df['sentiment']==1, 'sentiment'] = 3#1.5
#    tweet_df.ix[tweet_df['sentiment']==0, 'sentiment'] = 0
#    tweet_df.ix[tweet_df['sentiment']==-1, 'sentiment'] = -3
#    tweet_df.ix[tweet_df['sentiment']==-2, 'sentiment'] = -5
#    tweet_df.ix[tweet_df['sentiment']==2, 'sentiment'] = 5
#    
#    tweet_df.ix[tweet_df['sentiment']=='positive', 'sentiment'] = 5#1.5
#    tweet_df.ix[tweet_df['sentiment']=='neutral', 'sentiment'] = 0
#    tweet_df.ix[tweet_df['sentiment']=='negative', 'sentiment'] = -5#1.5

#    path = cwd + "/data/" + filename2;
#    tweet_df2 = pd.read_csv(path, sep=',')
#    tweet_df2 = tweet_df2.drop_duplicates(['tweet_id'])
#    tweet_df2 = tweet_df2[tweet_req_attributes]
##    tweet_df2['sentiment'] = pd.to_numeric(tweet_df2['sentiment'])
#    tweet_df2.ix[tweet_df2['sentiment']=='positive', 'sentiment'] = 5#1.5
#    tweet_df2 = tweet_df2[tweet_df2['sentiment'] == 'neutral']#.ix[tweet_df2['sentiment']=='neutral', 'sentiment'] = 0
#    tweet_df2.ix[tweet_df2['sentiment']=='negative', 'sentiment'] = -5#1.5
#    tweet_df2 = tweet_df2[tweet_df2['sentiment'] != 'irrelevant']
#    
#    frame = [tweet_df, tweet_df2]
#    data = pd.concat(frame)
#    print len(data)
##    data = tweet_df.merge(user_df, left_on="tweet_id", right_on="tweet_id", how="inner")
#    data = data.drop_duplicates(['tweet_id'])
##    data = data.dropna(subset=['user_id', 'tweet'])
##    data = data.dropna(subset=['user_id'])
##    op = df['sentiment'] == df['sentiment']
##    df.ix[:, 'sentiment'] *= 2# genre[i]
#    
##    Drop random neutrals
#    
#    data = data.drop(data.query('sentiment == "neutral"').sample(frac=.4).index)
#    print len(tweet_df), len(tweet_df2), len(data)
#    
##    print "From user data\n", Counter(list(data["user_id"])).most_common(50)
#    return data#[req_attributes]


#def tokenize_and_stopwords(data_sample):

#    print type(data_sample)
#    print len(data_sample)
#    #Get all english stopwords
#    try:
#        words = open("common_words.txt", "r").readlines()
#        for i in range(len(words)):
#            words[i] = words[i].strip()
#    except: 
#        words = []
#    
#    print "words", words
##    abb_dict = pickle.load(open("abbreviations", "r"))
#    stop = stopwords.words('english') + words #list(string.punctuation) + ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#    #Use only characters from reviews
#    data_sample = data_sample.str.replace("[^a-zA-Z ]", " ")#, " ")
#    data_sample = data_sample.str.lower()
#                
#    return [(" ").join([i for i in sentence.split() if i not in stop]) for sentence in data_sample]

#def cleanhtml(tweet):
#      cleanr = re.compile('<.*?>')
#      cleantext = re.sub(cleanr, '', tweet)
#      return cleantext

#def cleanUrl(tweet):
#    tweet= re.sub(r"http\S+", "",  tweet)
#    return tweet;

#def removeMention(tweet):
#    tweet = tweet.replace("rt@","").rstrip()
#    tweet = tweet.replace("rt ","").rstrip()
#    tweet = tweet.replace("@","").rstrip()
#    return tweet;

#def stemmer(preprocessed_data_sample):
#    print "stemming "
#    #Create a new Porter stemmer.
#    stemmer = PorterStemmer()
#    #try:
#    for i in range(len(preprocessed_data_sample)):
#        #Stemming
#        try:
#            preprocessed_data_sample[i] = preprocessed_data_sample[i].replace(preprocessed_data_sample[i], " ".join([stemmer.stem(str(word)) for word in preprocessed_data_sample[i].split()]))
#        except:
#        #No stemming
#            preprocessed_data_sample[i] = preprocessed_data_sample[i].replace(preprocessed_data_sample[i], " ".join([str(word) for word in preprocessed_data_sample[i].split()]))
#    return preprocessed_data_sample
#    
#def load_embeddings(file_name):
# 
#    with codecs.open(file_name, 'r', 'utf-8') as f_in:
#        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in 
#f_in])
#    wv = np.loadtxt(wv)
#    return wv, vocabulary
#    
##feature extraction - TFIDF and unigrams
#def vectorize(preprocessed_data_sample):
#    from sklearn.feature_extraction.text import TfidfVectorizer

#    # Initialize the "CountVectorizer" object, which is scikit-learn's
#    # bag of words tool.
##    no_features = 1000#500#806#150#800#600#350

#    #ngram_range=(1, 1)
##    input=u'content', encoding=u'utf-8', decode_error=u'strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer=u'word', stop_words=None, token_pattern=u'(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<type 'numpy.int64'>, norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False

#    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, vocabulary = None, ngram_range=(1,1), strip_accents=None)#, max_features = no_features)#, ngram_range=(2,2))
#    #vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(2,2), max_features = no_features)
#    # fit_transform() does two functions: First, it fits the model
#    # and learns the vocabulary; second, it transforms our training data
#    # into feature vectors. The input to fit_transform should be a list of
#    # strings.
#    train_data_features = vectorizer.fit_transform(preprocessed_data_sample)

#    # Numpy arrays are easy to work with, so convert the result to an
#    # array
#    train_data_features = train_data_features.toarray()
#    return [train_data_features, vectorizer, no_features]

    
def get_genre_score(df, word):#df[df["topic"].isin(genre_topics[iter])])

#    op = word in df["tokenized_sents"]
#    df = df.where(op, df)
#    l = list(df["sentiment"])
#    print df["sentiment"].mean()
#    print sum(l) / float(len(l))
    df = df[df["tweet"].str.contains(word, na=False)]    
#    df = df[df["tweet"].str.contains(word, na=False)]["sentiment"].mean()
#    l = list(df["sentiment"])
#    print df["sentiment"].mean()
#    print sum(l) / float(len(l))
#    print "after"
#    assert False
    return df["sentiment"].mean()

def preprocess(filename1, filename2):

    global tech, politics, sports, music, genre
    print "new!!"
    #filename = "Homework2_data.csv"
    [df, df0, df3] = readData(filename1, filename2)
    df = df[0:15000]
    print "length of df is", len(df)
#    print "from joined data\n", Counter(list(df["user_id"])).most_common(50)
    indices = []
#    df['tweet'] = df['tweet'].apply(cleanhtml).apply(cleanUrl).apply(removeMention).apply(removeTrailingHash);

    df['tweet'] = df['tweet'].apply(cleanhtml).apply(cleanUrl)#.apply(removeTrailingHash);
    df['tweet'] = tokenize_and_stopwords(df['tweet'])
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    df = df.dropna(subset=['sentiment'])
    print "final length", len(df)
    l = list(df["sentiment"])
    print df["sentiment"].mean()
    print sum(l) / float(len(l))
#    df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)
    all_topics = list(set(df['topic']))
    
#    Remove topics of no interest

#    df = df[df["topic"].isin(all_topics)]

    
    print "remove unnecessary\n", set(df["topic"])
#    Merging topics for word modelling
    
    topics_array = np.array(([tech, politics, music, sports]))
#    for index, row in df.iterrows():

#        tweet_topic = row['topic']
##        print "tweet_topic", tweet_topic
#        for i in range(len(topics_array)):
#            if tweet_topic in topics_array[i]:
##                print "ta", topics_array[i]
##                df["topic"][index] = genre[i]
#                df.ix[index, 'topic'] = genre[i]
##                print "df", df["topic"][index]
#                break


#    print "After Merge\n", set(df["topic"])
    
    data = DataFrame(df.groupby('topic')['tweet'].apply(list)).reset_index()
            
#    embed_data = 
#    data = shuffle(data)
    topics = set(list(data["topic"]))
    print "after grouping\n", topics
#    Watch out - only ten topics
#    topics = topics[0:10]

    for i in range(len(data)):
 
         data['tweet'][i] = " ".join(data['tweet'][i])
#    print topics
#    data = data[data["topic"] in  ]
#    Word topic mapping
#    try:
#        word_dict = pickle.load(open("word_dict", "r"))
#    except:
    tweets = ""

#    Generating vocabulary
#    for index, i in data.iterrows():
##        for i in data['tweet']:
#       # if i['topic'] in topics:
#        tweets += str(i['tweet'])
#    
    word_dict = {}
#    all_vocab = list(set(tweets.split()))
    [all_words_list, words_with_min_freq, words_del] = get_word_frequency_lists(data, 3)
#    print words_with_min_freq
#    print "\n\n"
#    print words_del
    print "min freq count:", len(words_with_min_freq)
    all_vocab = all_words_list#
    #all_vocab = tweets

#        for word in tweets:
                        
#            word_dict[word] = []
#            for i in range(len(topics)):
#                if word in data["tweet"][i]:
#                    word_dict[word].append(topics[i])        
#        pickle.dump(word_dict, open("word_dict", "wb"))
    
    print data
#    print "the word 'election' is present in", (word_dict['election'])
#    print "the word 'hillary' is present in", (word_dict['hillary'])
#    print len(word_dict)
#    Word model
    try:
        print "loading"
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        [X_train, Y_train] = pickle.load(open("train_data", "r"))
        X_train = X_train.reshape(len(X_train), 1, len(X_train[0]), 1)
        vocab = pickle.load(open("vocab_tsne", "r"))
        print "loaded"
    except:
        print "now entering"
        [model, X_train, Y_train, vocab] = train_cnn(word_dict, topics, df, all_vocab, words_with_min_freq, all_topics)
        pickle.dump(vocab, open("vocab_tsne", "wb"))
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
                print "dumping model failed"
    
#    Prediction
    y = model.predict(X_train, batch_size = 100)#, Y_train, batch_size=32, verbose=1, sample_weight=None)
    try:
#        pickle.dump([all_vocab, y], open("visualize", "wb"))
        print "pickled visualizations"
    except:
        print "couldn't pickle visual"
        
    print "This is the prediction"
#    y[y >= 0.1] = 1
#    y[y < 0.1] = 0
    print y[0:10]
    print "true labels"
    print Y_train[0:10]
#    print sum(sum(y == Y_train))
#    print (len(X_train) * len(X_train[0]))
    print (sum(sum((y - Y_train) ** 2))) / (len(X_train) * 4.0)

##    for d in diff:
##        print d
##    Error
    
#    
#    Hidden state
    get_last_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[7].output])

#    get_last_layer_output = K.function([model.layers[0].input],
#                                  [model.layers[4].output])

# output in train mode = 0
#    layer_output = np.array(get_last_layer_output([X_train[0:1200], 0])[0])
    
# output in train mode = 0
    start = 0
    increment = 100
    flag = 1
    while start+increment <= len(X_train):
        if flag:
            layer_output = get_last_layer_output([X_train[start:start+increment], 0])[0]
            flag = 0
        else:
            layer_output = np.concatenate((layer_output, get_last_layer_output([X_train[start:start+increment], 0])[0]))
        start += increment
    if start != len(X_train):
        layer_output = np.concatenate((layer_output, get_last_layer_output([X_train[start:len(X_train)], 0])[0]))
        
#    Write to file for Tsne or visualize
#    np.set_printoptions(suppress = True)
#    tsne = TSNE(n_components = 2, random_state = 0)
#    Y = tsne.fit_transform(layer_output)
    file_name = "labels.txt"
#    wv, vocabulary = load_embeddings(file_name)
    vocabulary = vocab
    wv = X_train    
#    plot.scatter(Y[:, 0], Y[:, 1])#, Y[:, 2])
#    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
#        plot.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
#    plot.title(topics)
#    plot.show()
#    
#    plot.scatter(tsne_vec[:, 0].astype('float32'), tsne_vec[:, 1].astype('float32'), 20, np.array(range(len(layer_output))))#np.array(vocab))
#    plot.show()
#    f1 = open("vectors.txt", "w")
#    f2 = open("labels.txt", "w")
    
    word2topic = {}
    embedding = {}
    for i in range(len(vocab)):
        word2topic[vocab[i]] = layer_output[i]
        embedding[vocab[i]] = y[i]
    try:
        pickle.dump(word2topic, open("word2topic", "wb"))
        pickle.dump(embedding, open("embedding", "wb"))
    except:
        "pickling word2vec failed"
#    for i in range(len(layer_output)):
#        f1.write(str(layer_output[i]) + str("\n"))
#        f2.write(str(vocab[i]) + str("\n"))
#    f1.close()
#    f2.close()


def train_cnn(word_dict, topics, df, vocab, words_with_min_freq, all_topics):

    global K, genre_topics, tech, politics, music, sports, history, req_pos_tags
    print "not using topics"
    topics = None
#    with K.tf.device('/gpu:1'):
    gpu_options = K.tf.GPUOptions(per_process_gpu_memory_fraction=0.8)#0.2)
    sess = K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options))
#        K._set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)))
    #all_topics = list(set(df['topic']))#np.concatenate((tech, politics, music, sports))
    try:
        X_train = []#np.array([])
        Y_train = []#np.array([])
        global X_train, Y_train
        
    #    Encoding X_train and Y_train
#        vocab = word_dict.keys()
#        killer
        random.shuffle(vocab)
        print "length of vocab is ", len(vocab)
        print "before"
#        print word_dict
#        print vocab
#        print topics
        try:
            [X_train, Y_train] = pickle.load(open("train_data", "r"))
        except:
            for i in range(len(vocab)):
                if i % 100 == 0:
                    print i
                c = np.zeros(len(all_topics))
                c_x = np.zeros(len(vocab))
    #            for i in range(len(topics)):
    #                print topics[i], word_
                
    #            locations = np.array([j for j in range(len(topics)) if topics[j] in word_dict[vocab[i]]])
    #            print locations
                c_x[i] = 1
                # = np.append(X_train, c_x)
    #            if True:
                try:#buggy, fews words aren't found in any topics, weird, space removed by mistake.'
                    #all_topics = list(set(df['topic']))
                    tag = CMUTweetTagger.runtagger_parse(list(vocab[i]))[0][0][1]
#                    print tag 
                    if (vocab[i] in words_with_min_freq):                    
                            
                        if tag in req_pos_tags:
                            for iter in range(len(all_topics)):
                            
            #                    print df["topic"]
            #                    print genre[iter]
            #                    print df["topic"] == genre[iter]
            #                    print df[df["topic"] == genre[iter]] 
            #                    c[iter] = get_genre_score(df[df["topic"] == genre[iter]], vocab[i])
                                op = df[df["topic"] == all_topics[iter]]
            #                    df_temp = op[op["tweet"].str.contains(vocab[i], na=False)]
                                c[iter] = op[op["tweet"].str.contains(vocab[i], na=False)]["sentiment"].mean()#get_genre_score(df[df["topic"] == topics[iter]], vocab[i])
            #                    assert False
            #                    c[iter] = df[df["topic"].isin(genre_topics[iter])]["tweet"].str.contains(word, na=False)["sentiment"].mean()
            #                c[locations] = 1
            #                print len(c_x)
                        Y_train.append(c)# = np.append(Y_train, c)
                        X_train.append(c_x)
        #                print "atleast"
            #            print locations
#                    else:
#                        assert False
    #            else:
    #                e = 1
                except Exception as e:
#                    print "this is the error", e
#                    assert False
                    print "couldn't find", vocab[i]
                    Y_train.append(np.zeros(len(all_topics)))
    #                print "l is",# locations, word 
                 #This could be buggy
            
                
        #    print len(X_train)
        #    print X_train[0]
        #    Main line
        #    X_train = csr_matrix(np.array(X_train))#
            X_train = np.array(X_train).astype('float16')
        #    Y_train = csr_matrix(Y_train)#
            Y_train = np.array(Y_train).astype('float16')
            Y_train = np.nan_to_num(Y_train)
            try:
                pickle.dump([X_train, Y_train], open("train_data", "wb"))
            except:
                "pickling data failed"
        print "after"
#        print X_train
        for counter in range(10):
            print "hello", vocab[counter], Y_train[counter]
            
        X_train = X_train.reshape(len(X_train), 1, len(X_train[0]), 1)#(32088, 1, 32088, 1)#
        print X_train.shape
    #    print "first shape", X_train.shape
        print X_train[0]

        print "Shape sir is", Y_train.shape
        cnn = Sequential()
#        cnn.add(Dense(1000, input_dim=14430))
#        cnn.add(Dense(500, activation="tanh"))
        cnn.add(Convolution2D(16, 100, 1,
            border_mode="same",
            activation="relu",
            input_shape=(1, X_train.shape[0], 1)))
        cnn.add(Dropout(0.2))
        cnn.add(Convolution2D(8, 4, 1, border_mode="same", activation="relu"))
        cnn.add(Dropout(0.2))
        cnn.add(MaxPooling2D(pool_size=(2, 1)))
        
##        cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
##        cnn.add(Convolution2D(64, 3, 1, border_mode="same", activation="relu"))
###        cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
##        cnn.add(MaxPooling2D(pool_size=(2, 1)))
#            
#        cnn.add(Convolution2D(32, 3, 1, border_mode="same", activation="relu"))
##        cnn.add(Convolution2D(32, 3, 1, border_mode="same", activation="relu"))
##        cnn.add(Convolution2D(64, 3, 1, border_mode="same", activation="relu"))
#        cnn.add(MaxPooling2D(pool_size=(2, 1)))
#            
        cnn.add(Flatten())
        cnn.add(Dropout(0.2))
#        cnn.add(Dropout(0.2))
        cnn.add(Dense(100, activation="linear"))
#        cnn.add(Dense(100, activation="tanh"))
        cnn.add(Dropout(0.2))
#    #    cnn.add(Dense(3, activation="softmax"))
        cnn.add(Dense(len(all_topics), activation="linear"))
        # define optimizer and objective, compile cnn

#        cnn.compile(loss="categorical_crossentropy", batch_size=32, optimizer="adam")

##         train

#        cnn.fit(X_train, Y_train, nb_epoch=20, show_accuracy=True)
#        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
        adam = Adam(lr=0.0006, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        cnn.compile(loss='mean_squared_error', optimizer=adam, batch_size=32)
        print history
    #    print X_train[0:5]
    #    print Y_train[0:5]
        cnn.fit(X_train, Y_train, batch_size=32, epochs=10)
        print "Done training, returning model"
        return [cnn, X_train, Y_train, vocab]
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
        filename2 = "full-corpus.csv"#"twitter-2016dev-CE-output.txt_semeval_userinfo.txt"
        preprocess(filename1, filename2)
#        cudaDeviceReset()
        
    except KeyboardInterrupt:
        del X_train
        del Y_train
        print "deleted"
        
    except AssertionError:
        del X_train
        del Y_train
        print "deleted"
#    except:
#        cudaDeviceReset()
    
main()
