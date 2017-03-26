from lib import *
#from keras.layers.merge import Concatenate
from keras.layers import Merge
import copy
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import math

word2topic = pickle.load(open("word2topic", "r"))
embedding = pickle.load(open("word2topic", "r"))

#embedding = pickle.load(open("embedding", "r"))
vocab = word2topic.keys()
max_words = 25#30
depth_embed = 100#370
depth_distance = 100#368#70#100

##def getEmbedding_word2vec(sentence, model):
#    
#    global max_words, depth, no_features, train_length
##    model = model[0]
#    list = np.array([])
#    for word in sentence:
#        if word in model.wv.vocab:
#             list = np.append(list, model.wv[word])
#    
#    #print list.size
#    if(list.size > depth*max_words):
#        list = list[0:depth*max_words]
#    #print sentence
#    pad = np.zeros(depth*max_words - list.size)
#    list = np.append(list, pad)
#    #print list.shape
#    return list
    
def get_topic_rep(topic, word2topic, word2vec):

    global vocab
    topics = str(topic).split(' ')
    v = np.zeros(word2topic['donald'].shape)#(np.concatenate((word2topic['donald'], word2vec.wv['donald'])).shape)
    counter = 0
#    if topics[0] in vocab:
#        v = np.append(v, word#2topic[topics[0]])
##        counter = 0
##        if
    for counter in range(len(topics)):
        if topics[counter] in vocab:
#            print topics[counter]
            try:
                v += word2topic[topics[counter]]#np.concatenate((word2topic[topics[counter]], word2vec.wv[topics[counter]]))
            except:
                v += word2topic[topics[counter]]#np.concatenate((word2topic[topics[counter]], np.zeros(word2vec.wv['donald'].shape)))
#    print counter + 1
    v /= (counter + 1) * 1.0
#    print type(v)
    return v

def custom_loss(y_true, y_pred):

    y = K.argmax(y_true, axis=1)
    print y[0:5]
##    y_true = np.array(y_true).astype('int64')
##
    print y_true[0:5]
##    length = y_true.get_shape()
##    l = tuple([length[i].value for i in range(0, len(length))])[0]

#    for i in range(y_pred.get_shape()[0].value):
#        y_pred[i] = y_pred[i][y[i]]
#         
#    y_pred = K.log(y_pred[:, K.constant(y, dtype='int64')])
    return K.mean(K.categorical_crossentropy(y_pred[np.where(K.eval(K.equal(y, 0)))[0], :], y_true[np.where(K.eval(K.equal(y, 0)))[0], :]), K.categorical_crossentropy(y_pred[np.where(K.eval(K.equal(y, 1)))[0], :], y_true[np.where(K.eval(K.equal(y, 1)))[0], :]), K.categorical_crossentropy(y_pred[np.where(K.eval(K.equal(y, 2)))[0], :], y_true[np.where(K.eval(K.equal(y, 2)))[0], :]))
#    return K.sum(K.mean(K.dot(K.equal(y, 0), y_pred)), K.mean(K.dot(K.equal(y, 1), y_pred)), K.mean(K.dot(K.equal(y, 2), y_pred))) 
    
def evaluate(y_test, thresholded_pred):

    print "accuracy", (sum(abs(y_test == thresholded_pred))) / float(len(thresholded_pred))
    print Counter(y_test)
    print Counter(thresholded_pred)
    print confusion_matrix(y_test, thresholded_pred)
    print "f1 is", f1_score(y_test, thresholded_pred, average='macro')
        
def distance_embed(sentence):

    global max_words, depth_distance, word2topic
    list = np.array([])
    for word in sentence:
        if word in vocab:
             list = np.append(list, word2topic[word])
    
    #print list.size
    if(list.size > max_words * depth_distance):
        list = list[0:max_words * depth_distance]
    #print sentence
    pad = np.zeros(max_words * depth_distance - list.size)
    list = np.append(list, pad)
    #print list.shape
    return list


def getEmbedding(sentence, model):
    
    global max_words, depth_embed, embedding#, depth_distance
    list = np.array([])
    for word in sentence:
        if word in vocab:
            try:
                list = np.append(list, model[word])
#                print "found", word
            except:
                list = np.append(list, np.zeros(model['donald'].shape))
#                print word
    #print list.size
    if(list.size > max_words * depth_embed):
        list = list[0:max_words * depth_embed]
    #print sentence
    pad = np.zeros(max_words * depth_embed - list.size)
    list = np.append(list, pad)
    #print list.shape
    return list

#def getPOS(sentence):
#    
#    global max_words#, depth
#    all_tags = CMUTweetTagger.runtagger_parse(sentence)

#    list = np.array([])
#    for i in range(len(sentence)):
#        if sentence[i] in vocab:
#             list = np.append(list, all_tags[i])
#    
#    #print list.size
#    if(list.size > max_words):
#        list = list[0:max_words]
#    #print sentence
#    pad = np.zeros(max_words - list.size)
#    list = np.append(list, pad)
#    #print list.shape
#    return list
    
#def getEmbedding(sentence):

#    global word2topic, vocab
#    max_words = 30
#    list = []#np.array([])
#    for word in sentence:
#        if word in vocab:
##            list = np.append(list, word2topic[word])
#            list.append(word2topic[word])
##    list = np.array(list)
#    #print list.size
#    if(len(list) > max_words):
#        list = list[0:max_words]
#    #print sentence
#    pad = [0] * 100# * (max_words - len(list))#np.zeros(max_words - list.size)
#    for i in range((max_words - len(list))):
#        list.append(pad)
##    list.append(pad)
#    #print list.shape
#    return list
    
#getEmbedding(df['tokenized_sents'][0])

def readData2(filename1):
    global tech, politics, music, sports
    print "reading data"
    all_topics = np.concatenate((tech, politics, music, sports))
    cwd = os.getcwd()
#    req_attributes = ['tweet_id', 'topic', 'sentiment', 'tweet']#, 'user_id']#, 'followers_count', 'statuses_count', 'description', 'friends_count', 'location']
#    user_req_attributes = ['tweet_id', 'user_id']#, 'followers_count', 'statuses_count', 'description', 'friends_count', 'location']
    tweet_req_attributes = ['tweet', 'topic']
    path = cwd + "/" + filename1;
    tweet_df = pd.read_csv(path, sep=',');
    
    tweet_df = tweet_df[tweet_req_attributes]
#    print tweet_df['sentiment']
    print tweet_df
    return tweet_df#[req_attributes]


def run_model():
    
        global tech, politics, sports, music, genre, max_words, depth_embed, depth_distance, word2topic, vocab, K
    #    with K.tf.device('/gpu:1'):
        gpu_options = K.tf.GPUOptions(per_process_gpu_memory_fraction=1.0)#0.8)#0.2)
        sess = K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options))
        print "in run model"
#        all_topics = np.concatenate((tech, politics, music, sports))
#        print "AAAAAAAAAAAAAAAAAAAAA"
#        print len(all_topics)
#        print all_topics
        

#        try:
#            [X, y, df, d] = pickle.load(open("data_rnn", "r"))
#            print d
##            df = df[df["topic"].isin(all_topics)]

#        except:
            #filename = "Homework2_data.csv"
                #        word2topic = pickle.load(open("word2topic", "r"))
        df = readData2("test_file.csv")

        df['tweet'] = df['tweet'].apply(cleanhtml).apply(cleanUrl)#.apply(removeTrailingHash);
        df['tweet'] = tokenize_and_stopwords(df['tweet'])            
        
        df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)
#        try:
        word2vec = wv.Word2Vec.load("word2vec")
    #model.similarity("this", "is")
#            model.init_sims(replace=True)
        print "loaded"
#        except:
#            print "couldn't find word2vec'"
##                    word2vec = wv.Word2Vec(df["tokenized_sents"], size=depth_embed, window=5, min_count=5, workers=4)
##                    word2vec.save("word2vec")

        #X.shape[0]#7349
        df['embedding'] = df['tokenized_sents'].apply(getEmbedding, args=(word2topic,))
        df['word2vec'] = df['tokenized_sents'].apply(getEmbedding, args=(word2vec.wv,))
        X = list(df['embedding'])
        X_w = list(df['word2vec'])
        X = np.reshape(np.ravel(X), (len(X), max_words, depth_embed))
        X_w = np.reshape(np.ravel(X_w), (len(X_w), max_words, depth_embed))
#            a = copy.deepcopy(X)#np.array(df['embedding'])

        df['tweet_rep'] = df['tokenized_sents'].apply(distance_embed)
####            a = list(df['tweet_rep'])
####            a = np.reshape(np.ravel(a), (len(a), max_words, depth_distance))            
        df['topic_rep'] = df['topic'].apply(get_topic_rep, args=(word2topic, word2vec,))
        d = []


##        LOAD MODEL
        print "loading model"
        try:
            model = load_model('model_rnn_best')
            one_hot = list(df['topic_rep'])#(pd.get_dummies(df['topic']))
            one_hot = np.reshape(np.ravel(np.ravel(one_hot)), (len(one_hot), 1, depth_distance))
        except:
            print "model not found"

        return [model, X, X_w, df, d, one_hot]

def load_model(filename):
    
    json_file = open(filename+ '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(filename + ".h5")
#    [X, y, df, d] = pickle.load(open("data_rnn", "r"))

    return model#, X, y, df, d]

    
def sentiment_classifier():
    
    
    [model, X, X_w, df, d, one_hot] = run_model()
    print "obtained data"    
    print "length of X is", len(X)
    

    pred = model.predict([X, one_hot], batch_size = 64)#, Y_train, batch_size=32, verbose=1, sample_weight=None)
    print pred
    print np.argmax(pred, axis=1)
sentiment_classifier()
