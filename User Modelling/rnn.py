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
    v = np.zeros(np.concatenate((word2topic['donald'], word2vec.wv['donald'])).shape)
    counter = 0
#    if topics[0] in vocab:
#        v = np.append(v, word#2topic[topics[0]])
##        counter = 0
##        if
    for counter in range(len(topics)):
        if topics[counter] in vocab:
#            print topics[counter]
            try:
                v += np.concatenate((word2topic[topics[counter]], word2vec.wv[topics[counter]]))
            except:
                v += np.concatenate((word2topic[topics[counter]], np.zeros(word2vec.wv['donald'].shape)))
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

def run_model():
    
        global tech, politics, sports, music, genre, max_words, depth_embed, depth_distance, word2topic, vocab, K
    #    with K.tf.device('/gpu:1'):
        gpu_options = K.tf.GPUOptions(per_process_gpu_memory_fraction=1.0)#0.8)#0.2)
        sess = K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options))

#        all_topics = np.concatenate((tech, politics, music, sports))
#        print "AAAAAAAAAAAAAAAAAAAAA"
#        print len(all_topics)
#        print all_topics
        

        try:
            [X, y, df, d] = pickle.load(open("data_rnn", "r"))
            print d
#            df = df[df["topic"].isin(all_topics)]

        except:
            #filename = "Homework2_data.csv"
                #        word2topic = pickle.load(open("word2topic", "r"))
            df = readData(filename1, filename2)
            #df = df[df["topic"].isin(all_topics)]
            df['sentiment'] = pd.to_numeric(df['sentiment'])
            
#            topics_array = np.array(([tech, politics, music, sports]))
#            print genre
#            for index, row in df.iterrows():

#                tweet_topic = row['topic']
#        #        print "tweet_topic", tweet_topic
#                for i in range(len(topics_array)):
#                    if tweet_topic in topics_array[i]:
#        #                print "ta", topics_array[i]
#        #                df["topic"][index] = genre[i]
#                        df.ix[index, 'topic'] = genre[i]
#        #                print "df", df["topic"][index]
#                        break

            
            #    Remove topics of no interest

            print "length of df is", len(df)
#            print "from joined data\n", Counter(list(df["user_id"])).most_common(50)
            indices = []
        #    df['tweet'] = df['tweet'].apply(cleanhtml).apply(cleanUrl).apply(removeMention).apply(removeTrailingHash);
#            df['tweet'] = df['tweet'].apply(cleanhtml).apply(cleanUrl).apply(removeTrailingHash);

            df['tweet'] = df['tweet'].apply(cleanhtml).apply(cleanUrl)#.apply(removeTrailingHash);
            df['tweet'] = tokenize_and_stopwords(df['tweet'])            
           # df = df.sample(frac=1).reset_index(drop=True)
#            df = shuffle(df)
            print df.size
            
            df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)
            try:
                word2vec = wv.Word2Vec.load("word2vec")
            #model.similarity("this", "is")
#            model.init_sims(replace=True)
                print "loaded"
            except:
                word2vec = wv.Word2Vec(df["tokenized_sents"], size=depth_embed, window=5, min_count=5, workers=4)
                word2vec.save("word2vec")

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
#            a = np.reshape(a, ())
####            b = list(df['topic_rep'])
####            print b[0]
#            print b
#            print b.shape
####            b = np.reshape(np.ravel(np.ravel(b)), (X.shape[0], 1, depth_distance))

#####            c = (a - b)**2
######            d = c
#####            for i1 in range(len(c)):
#####                for j1 in range(len(c[0])):
#####                    d.append(abs(sum(c[i1][j1])))
#####            d = np.array(d)
#####            d = np.reshape(d, (len(a), max_words))
#####            d[d==0] = 0.1
#####            d = 1.0 / d
#####            print "d[0] is !!!", d[0]


#            df['distance'] = d#1.0 / d#sum(sum(sum(abs(np.array(df['embedding']) - np.array(df['topic_rep'])))))
#            one_hot = 
#            df['pos'] = df['tweet'].apply(getPOS)
#            X = np.column_stack((np.array(df['embedding']), np.array(df['pos'])))
    #        for i in range(len(X)):
    #            X[i] = X[i][0:] 
    #        B = np.array([])
    #        np.dstack((X, B)).shape        
            
            y = np.array(df['sentiment']) 
#            y = np.array(pd.get_dummies(df['sentiment']))
###            No dumping
#            try:
#                pickle.dump([X, y, df, d], open("data_rnn", "wb"))
#            except:
#                "dumping data failed"
        print len(X[0])
        print len(X)

        X_train = X[0:15000]
        X_test = X[15000:]
        X_train_w = X_w[0:15000]
        X_test_w = X_w[15000:]
        y_train = y[0:15000]
        y_test = y[15000:]
        print " Y train!!\n", y_train[0:5]
        print list(df['sentiment'])[0:5]
        print y_test[0:5]


##        LOAD MODEL
        try:
            model = load_model('model_rnn')
            one_hot = list(df['topic_rep'])#(pd.get_dummies(df['topic']))
            one_hot = np.reshape(np.ravel(np.ravel(one_hot)), (len(one_hot), 1, depth_distance))

        except:
#        Word model
            model_word = Sequential()
            model_word.add(Bidirectional(LSTM(3 * max_words, activation='relu', return_sequences=True), input_shape=(max_words, depth_embed)))
            model_word.add(Dropout(0.2))
            model_word.add(Bidirectional(LSTM(max_words, activation='tanh', return_sequences=True)))
            model_word.add(Dropout(0.2))
            
            model_word_w = Sequential()
            model_word_w.add(Bidirectional(LSTM(3 * max_words, activation='relu', return_sequences=True), input_shape=(max_words, depth_embed)))
            model_word_w.add(Dropout(0.2))
            model_word_w.add(Bidirectional(LSTM(max_words, activation='tanh', return_sequences=True)))
            model_word_w.add(Dropout(0.2))
    #        model_word.add(Bidirectional(LSTM(max_words, return_sequences=True)))
    #        model_word.add(Dropout(0.2))
    #        model_word.add(Flatten())
    #        model_word.add(MaxPooling2D(pool_size=(2, 1)))
    #        model_word.add(Dropout(0.2))

    #        model_word.add(Dense((max_words), activation="tanh"))
            
    ##        Reverse
    #        model_word_r = Sequential()
    #        model_word_r.add(LSTM(max_words, input_shape=(max_words, depth), consume_less='gpu', go_backwards=True))
    #        model_word_r.add(Dropout(0.2))
    ##        model_word_r.add(LSTM(max_words, input_shape=(max_words, depth), consume_less='gpu', go_backwards=True))

    #        Topic model

            
            print len(set(df['topic']))
            print "set is", set(df['topic'])
#            print "topic rep!! \n", df['topic_rep']        
            one_hot = list(df['topic_rep'])#(pd.get_dummies(df['topic']))
#            print df['topic'][0:5]
            print "init one hot", one_hot[0:2]
    #        one_hot = one_hot.as_matrix()
            
#            one_hot = d#df['distance']
            print len(one_hot)
    #        print len(one_hot[0])
    #        print one_hot[0]

##            one_hot = np.reshape(one_hot, (one_hot.shape[0], max_words, 1))
#            one_hot = np.reshape(np.ravel(np.ravel(one_hot)), (len(one_hot), depth_distance, 1))
            one_hot = np.reshape(np.ravel(np.ravel(one_hot)), (len(one_hot), 1, 2*depth_distance))
            one_hot_train = one_hot[0:15000]
            one_hot_test = one_hot[15000:]
            print "one hot shape", one_hot.shape
            
            model_topic = Sequential()
    #        , return_sequences=True
            model_topic.add(Bidirectional(LSTM(max_words, activation='tanh', return_sequences=True), input_shape=(1, 2*depth_distance)))
            model_topic.add(Dropout(0.2))
    #        model_topic.add(Bidirectional(LSTM(max_words, return_sequences=True)))
    #        model_topic.add(Flatten())
    #        model_topic.add(MaxPooling2D(pool_size=(2, 1)))
    #        model_topic.add(Dropout(0.2))

    #        model_topic.add(Dense(4, activation="tanh"))
    #        model_topic.add(Dropout(0.2))

    #        Merge forward and backward
    #        merged = Merge([model_word_f, model_word_r], mode='concat')#, concat_axis=1)
    #        model_word = Sequential()
    #        model_word.add(merged)
    #        model_word.add(Dropout(0.2))
    ##        model_word.add(MaxPooling2D(pool_size=(2, 1)))
    ##        model_word.add(Dropout(0.2))
    #        model_word.add(LSTM(max_words, input_shape=(2*max_words, 1)))
    #        model_word.add(Dropout(0.2))
    #        Merge merged and topic info
            merged2 = Merge([model_word, model_word_w, model_topic], mode='concat', concat_axis=1)
    #        merged = Concatenate([model_word, model_topic], axis=-1)
            model = Sequential()
            model.add(merged2)
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(2*max_words, activation='relu', return_sequences=True)))#)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(2*max_words, activation='tanh', return_sequences=True)))

    ##      #  model.add(Flatten())
            model.add(Dropout(0.2))
    #        model.add(Bidirectional(LSTM(max_words), input_shape=(4 + max_words, 1)))
            print "added additional Dense, no flatten"
##            model.add(Dense(max_words, activation='tanh'))
#            model.add(Dropout(0.2))

            #model.add(Dense(1, activation='linear', W_constraint=maxnorm(3)))
            model.add(Bidirectional(LSTM(2*max_words, activation='tanh', return_sequences=True)))#)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(max_words, activation='tanh', return_sequences=True)))#)))
            model.add(Dropout(0.2))
#            model.add(LSTM(3, activation="softmax"))
            model.add(LSTM(1, activation="linear"))
    #        optimizer = RMSprop(lr=0.01)
    #        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            
            adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='mean_squared_error', optimizer=adam)
            print "Custom!!!"
#            model.compile(loss=custom_loss, optimizer=adam)
            print "came here saaaaar!!!!!!\n\n"
        #    print X[0:5]
        #    print Y_train[0:5]
            print "model changedd !!!"
            model.fit([X_train, X_train_w, one_hot_train], y_train, batch_size = 64, epochs=13, validation_split = 0.05, callbacks=[history])
            model_json = model.to_json()
            with open("model_rnn.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("model_rnn.h5")
            print("Saved model to disk")
    #        print(history.History)

        return [model, X, X_w, y, df, d, one_hot]
#        print X.shape
#        print X[0]
#        print X[0]
#        for i in X[0]:
#            print i

def load_model(filename):
    
    json_file = open(filename+ '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(filename + ".h5")
#    [X, y, df, d] = pickle.load(open("data_rnn", "r"))

    return model#, X, y, df, d]

def duplicate_model(filename):

    global tech, politics, sports, music, genre, max_words, depth, word2topic, vocab, K
    
    print "Duplicating!!"
    #        Word model
    model_word = Sequential()
    model_word.add(Bidirectional(LSTM(max_words, return_sequences=True), input_shape=(max_words, depth)))
    model_word.add(Dropout(0.2))
#    model_word.add(Flatten())
#    model_word.add(MaxPooling2D(pool_size=(2, 1)))
#    model_word.add(Dropout(0.2))



    model_topic = Sequential()
    model_topic.add(Bidirectional(LSTM(max_words, return_sequences=True), input_shape=(max_words, 1)))
    model_topic.add(Dropout(0.2))
#    model_topic.add(Flatten())
#    model_topic.add(MaxPooling2D(pool_size=(2, 1)))
#    model_topic.add(Dropout(0.2))

    merged2 = Merge([model_word, model_topic], mode='concat')

    model = Sequential()
    model.add(merged2)
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(max_words, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(max_words))

#        model.add(Flatten())
    model.add(Dropout(0.2))



#    merged = Concatenate([model_word, model_topic], axis=-1)
    model = Sequential()
    model.add(merged2)
    model.add(Dropout(0.2))
    model.add(LSTM(max_words))
#    model.add(Dropout(0.2))
#    print "added additional Dense, no flatten"
#    model.add(Dense(1, activation='linear', W_constraint=maxnorm(5)))

    json_file = open(filename+ '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_true = model_from_json(loaded_model_json)
    # load weights into new model
    model_true.load_weights(filename + ".h5")
    
    model.layers[0].set_weights(model_true.layers[0].get_weights())
    model.layers[1].set_weights(model_true.layers[1].get_weights())
    model.layers[2].set_weights(model_true.layers[2].get_weights())
    model.layers[3].set_weights(model_true.layers[3].get_weights())
    try:
        model.layers[3].set_weights(model_true.layers[3].get_weights())
#        model.layers[3].set_weights(model_true.layers[3].get_weights())
        print "tried"
    except:
        print "excepted"
#        model.add(Dropout(0.2))
#        model.layers[3].set_weights(model_true.layers[3].get_weights())
    return model

#equal weighted categorical cross entropy

    
    
def sentiment_classifier():
    
    global max_words, depth_distance, depth_embed
    print "in senti class, changes, class\n\n"
    try:
        assert False
        print "in try\n\n"
        [model, X, y, df, d] = load_model('model_rnn')
        print "Data found"
        print "done"

    except Exception, e:
        print "Caught an exception\n\n"
        print "Error is", str(e), "\n\n"
        [model, X, X_w, y, df, d, one_hot] = run_model()
        
    print "length of X is", len(X)
    X_test = X[15000:]
    y_test = y[15000:]
    X_test_w = X_w[15000:]
    X_train_w = X_w[0:15000]

    X_train = X[0:15000]
    y_train = y[0:15000]

    topics = list(df['topic'])
    
#    ____________________________________________________________________________________________________________HERE_________________
        
#    one_hot = d#df['distance']        
#    one_hot = pd.get_dummies(df['topic'])
#    one_hot = one_hot.as_matrix()
#    print len(set(df['topic']))
#    print "set is", set(df['topic'])
#    print len(one_hot)
#    print len(one_hot[0])
#    print one_hot[0]
##    print len(all_topics)
##    print all_topics
#    print set(df["topic"])
#    one_hot = np.array(df['topic_rep'])#np.array(pd.get_dummies(df['topic']))
#    one_hot = np.reshape(one_hot, (X.shape[0], 1, depth_distance))
    one_hot_train = one_hot[0:15000]
    one_hot_test = one_hot[15000:]
    

    pred = model.predict([X_test, X_test_w, one_hot_test], batch_size = 64)#, Y_train, batch_size=32, verbose=1, sample_weight=None)
    print pred[0:5]
    print y_test[0:5]
#    pred[:, 0] *= 1.5
#    margin = 0.06
#    indexes = pred[:, 0] + margin >= pred[:, 1]
#    print indexes
#    pred[indexes, 0] = pred[indexes, 1] + 0.01
#    print pred[0:5]
    print "This is the prediction"
#    y[y >= 0.1] = 1
#    y[y < 0.1] = 0
    pred.shape = (pred.shape[0],)
    print pred[0:20]
    print "true labels"
    print y_test[0:20]
#    print sum(sum(y == Y_train))
#    print (len(X_train) * len(X_train[0]))
    print (sum(abs(y_test - pred))) / float(len(pred))
    thresh1 = 1.5#49#1.8#1.5
    thresh2 = 3.9
    thresholded_pred = copy.deepcopy(pred)
    thresholded_pred[(pred > (-thresh1 +  0.0)) & (pred < thresh2)] = 0
    thresholded_pred[(pred >= thresh1) & (pred < thresh2)] = 3#1
    thresholded_pred[pred >= thresh2] = 5#2
    thresholded_pred[(pred > -thresh2) & (pred <= (-thresh1 +  0.0))] = -3#1
    thresholded_pred[pred <= -thresh2] = -5#2
    thresholded_pred = thresholded_pred.astype('int8')
    print "Testing"
    evaluate(y_test, thresholded_pred)
    
    y_test[y_test > 0] = 1
    y_test[y_test < 0] = -1
    
    thresholded_pred[thresholded_pred > 0] = 1
    thresholded_pred[thresholded_pred < 0] = -1
    
#    thresholded_pred = pred.argmax(axis=1)
#    y_test = y_test.argmax(axis=1)
    evaluate(y_test, thresholded_pred)
    
    

    pred = model.predict([X_train, X_train_w, one_hot_train], batch_size = 64)#, Y_train, batch_size=32, verbose=1, sample_weight=None)
    print pred[0:5]
    print y_train[0:5]
    #pred[:,0] *= 1.5
    print "This is the prediction"
    
#    y[y >= 0.1] = 1
#    y[y < 0.1] = 0
    pred.shape = (pred.shape[0],)
    print pred[0:20]
    print "true labels"
    print y_train[0:20]
#    print sum(sum(y == Y_train))
#    print (len(X_train) * len(X_train[0]))
    print (sum(abs(y_train - pred))) / float(len(pred))
    thresh1 = 1.5
    thresh2 = 3.9
    thresholded_pred = copy.deepcopy(pred)
    thresholded_pred[(pred > (-thresh1 +  0.0)) & (pred < thresh2)] = 0
    thresholded_pred[(pred >= thresh1) & (pred < thresh2)] = 3#1
    thresholded_pred[pred >= thresh2] = 5#2
    thresholded_pred[(pred > -thresh2) & (pred <= (-thresh1 +  0))] = -3#1
    thresholded_pred[pred <= -thresh2] = -5#2
    thresholded_pred = thresholded_pred.astype('int8')
    print "Training"
    evaluate(y_train, thresholded_pred)
    y_train[y_train > 0] = 1
    y_train[y_train < 0] = -1
    
    thresholded_pred[thresholded_pred > 0] = 1
    thresholded_pred[thresholded_pred < 0] = -1
    
#    thresholded_pred = pred.argmax(axis=1)
#    y_train = y_train.argmax(axis=1) 
    evaluate(y_train, thresholded_pred)

#    model_dup = duplicate_model('model_rnn')
#    layer_output = model_dup.predict([X_test, one_hot_test], batch_size = 64)
#    
###    get_last_layer_output = K.function([model.layers[0].input],
###                                  [model.layers[2].output])
##    get_last_layer_output = K.function([model.layers[0].input, K.learning_phase()],
##                                  [model.layers[2].output])

### output in train mode = 0
###    layer_output = np.array(get_last_layer_output([X_train[0:1200], 0])[0])
##    
### output in train mode = 0
##    
###    X = [X_test, one_hot_test]
##    print X_test.shape
##    print one_hot_test.shape
##    print len(X_test)
##    print len(one_hot_test)
##    
##    
##    X_2 = np.concatenate((X_test, one_hot_test), axis=2)

##    start = 0
##    increment = 100
##    flag = 1
##    print len(X_test)
##    print "now!!"
##    while start+increment <= len(X_test):
###        X = [[X_test[start:start+increment], 1], [one_hot_test[start:start+increment], 1]]
##        if flag:
##            layer_output = get_last_layer_output([X_2[start:start+increment], 0])[0]#get_last_layer_output([[X_test[start:start+increment], 0], [one_hot_test[:, start:start+increment], 0]])[0]
##            flag = 0
##        else:
##            layer_output = np.concatenate((layer_output, get_last_layer_output([X_2[start:start+increment], 0])[0]))
##        start += increment
##    if start != len(X_test):
###        X = [X_test[start:start+increment], one_hot_test[start:start+increment]]
##        layer_output = np.concatenate((layer_output, get_last_layer_output([X_2[start:start+increment], 0])[0]))
#    print "length of hidden", len(layer_output[0])    
#    for iter in range(10):
#        print df["tweet"][iter], layer_output[iter] 
        
sentiment_classifier()
