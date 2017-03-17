from lib import *
#from keras.layers.merge import Concatenate
from keras.layers import Merge
import copy
from collections import Counter
from sklearn.metrics import confusion_matrix


word2topic = pickle.load(open("word2topic", "r"))
vocab = word2topic.keys()
max_words = 30
depth = 100

def get_topic_rep(topic):

    global vocab
    topics = str(topic).split(' ')
    v = np.zeros(word2topic['donald'].shape)
    counter = 0
#    if topics[0] in vocab:
#        v = np.append(v, word2topic[topics[0]])
##        counter = 0
##        if
    for counter in range(len(topics)):
        if topics[counter] in vocab:
#            print topics[counter]
            v += word2topic[topics[counter]]
    
#    print counter + 1
    v /= (counter + 1) * 1.0
#    print type(v)
    return v

def evaluate(y_test, thresholded_pred):

    print "accuracy", (sum(abs(y_test == thresholded_pred))) / float(len(thresholded_pred))
    print Counter(y_test)
    print Counter(thresholded_pred)
    print confusion_matrix(y_test, thresholded_pred)
        
def getEmbedding(sentence):
    
    global max_words, depth
    list = np.array([])
    for word in sentence:
        if word in vocab:
             list = np.append(list, word2topic[word])
    
    #print list.size
    if(list.size > max_words * depth):
        list = list[0:max_words * depth]
    #print sentence
    pad = np.zeros(max_words * depth - list.size)
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
    
        global tech, politics, sports, music, genre, max_words, depth, word2topic, vocab, K
    #    with K.tf.device('/gpu:1'):
        gpu_options = K.tf.GPUOptions(per_process_gpu_memory_fraction=0.8)#0.2)
        sess = K.tf.Session(config=K.tf.ConfigProto(gpu_options=gpu_options))

        all_topics = np.concatenate((tech, politics, music, sports))
        print "AAAAAAAAAAAAAAAAAAAAA"
        print len(all_topics)
        print all_topics
        

        try:
            [X, y, df, d] = pickle.load(open("data_rnn", "r"))
#            df = df[df["topic"].isin(all_topics)]

        except:
            #filename = "Homework2_data.csv"
                #        word2topic = pickle.load(open("word2topic", "r"))
            df = readData(filename1, filename2)
            df = df[df["topic"].isin(all_topics)]
            df['sentiment'] = pd.to_numeric(df['sentiment'])
            
            topics_array = np.array(([tech, politics, music, sports]))
            print genre
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
            df = shuffle(df)
            print df.size
            
            df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['tweet']), axis=1)
            #X.shape[0]#7349
            df['embedding'] = df['tokenized_sents'].apply(getEmbedding)
            X = list(df['embedding'])
            X = np.reshape(np.ravel(X), (len(X), max_words, depth))
            
            df['topic_rep'] = df['topic'].apply(get_topic_rep)
            d = []
            a = copy.deepcopy(X)#np.array(df['embedding'])
#            a = np.reshape(a, ())
            b = list(df['topic_rep'])
            print b[0]
#            print b
#            print b.shape
            b = np.reshape(np.ravel(np.ravel(b)), (X.shape[0], 1, depth))

            c = (a - b)**2
            for i1 in range(len(c)):
                for j1 in range(len(c[0])):
                    d.append(abs(sum(c[i1][j1])))
            d = np.array(d)
            d = np.reshape(d, (len(a), max_words))
            d[d==0] = 0.1 
#            df['distance'] = 1.0 / d#sum(sum(sum(abs(np.array(df['embedding']) - np.array(df['topic_rep'])))))
#            one_hot = 
#            df['pos'] = df['tweet'].apply(getPOS)
#            X = np.column_stack((np.array(df['embedding']), np.array(df['pos'])))
    #        for i in range(len(X)):
    #            X[i] = X[i][0:] 
    #        B = np.array([])
    #        np.dstack((X, B)).shape        
            
            y = np.array(df['sentiment']) 
            pickle.dump([X, y, df, d], open("data_rnn", "wb"))
            
        print len(X[0])
        print len(X)

        X_train = X[0:6000]
        X_test = X[6000:]
        y_train = y[0:6000]
        y_test = y[6000:]
#        Word model
        model_word = Sequential()
        model_word.add(Bidirectional(LSTM(max_words, return_sequences=True), input_shape=(max_words, depth)))
        model_word.add(Dropout(0.2))
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

        
        
#        one_hot = pd.get_dummies(df['topic'])
#        print len(set(df['topic']))
#        print "set is", set(df['topic'])
#        one_hot = one_hot.as_matrix()
        
        one_hot = d#df['distance']
        print len(one_hot)
        print len(one_hot[0])
        print one_hot[0]

        one_hot = np.reshape(one_hot, (X.shape[0], one_hot.shape[1], 1))
        one_hot_train = one_hot[0:6000]
        one_hot_test = one_hot[6000:]
        print "one hot shape", one_hot.shape
        
        model_topic = Sequential()
        model_topic.add(Bidirectional(LSTM(max_words, return_sequences=True), input_shape=(max_words, 1)))
        model_topic.add(Dropout(0.2))
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
        merged2 = Merge([model_word, model_topic], mode='concat')
#        merged = Concatenate([model_word, model_topic], axis=-1)
        model = Sequential()
        model.add(merged2)
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(max_words, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(LSTM(max_words))

#        model.add(Flatten())
        model.add(Dropout(0.2))
#        model.add(Bidirectional(LSTM(max_words), input_shape=(4 + max_words, 1)))
        print "added additional Dense, no flatten"
#        model.add(Dense(max_words, activation='tanh'))
#        model.add(Dropout(0.2))

        model.add(Dense(1, activation='linear', W_constraint=maxnorm(5)))
#        model.add(Dense(5, activation="softmax"))
#        optimizer = RMSprop(lr=0.01)
#        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        
        adam = Adam(lr=0.00055, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=3e-6)
        model.compile(loss='mean_squared_error', optimizer=adam, batch_size=32)
        print "came here saaaaar!!!!!!\n\n"
    #    print X[0:5]
    #    print Y_train[0:5]
        model.fit([X_train, one_hot_train], y_train, batch_size=32, epochs=100)
        model_json = model.to_json()
        with open("model_rnn.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model_rnn.h5")
        print("Saved model to disk")
        return [model, X, y, df, d]
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
    [X, y, df, d] = pickle.load(open("data_rnn", "r"))

    return [model, X, y, df, d]

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

def sentiment_classifier():
    
    print "in senti class, hello\n\n"
    try:
        print "in try\n\n"
        [model, X, y, df, d] = load_model('model_rnn')
        print "Data found"
        print "done"

    except Exception, e:
        print "Caught an exception\n\n"
        print "Error is", str(e), "\n\n"
        [model, X, y, df, d] = run_model()
        
    X_test = X[6000:]
    y_test = y[6000:]
    X_train = X[0:6000]
    y_train = y[0:6000]

    topics = list(df['topic'])
    
#    ____________________________________________________________________________________________________________HERE_________________
        
    one_hot = d#df['distance']        
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
    one_hot = np.reshape(one_hot, (X.shape[0], one_hot.shape[1], 1))
    one_hot_train = one_hot[0:6000]
    one_hot_test = one_hot[6000:]
    

    pred = model.predict([X_test, one_hot_test], batch_size = 100)#, Y_train, batch_size=32, verbose=1, sample_weight=None)
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
    thresh1 = 2.5#1.8#1.5
    thresh2 = 3.65
    thresholded_pred = copy.deepcopy(pred)
    thresholded_pred[(pred > (-thresh1 +  0.2)) & (pred < thresh2)] = 0
    thresholded_pred[(pred >= thresh1) & (pred < thresh2)] = 3#1
    thresholded_pred[pred >= thresh2] = 5#2
    thresholded_pred[(pred > -thresh2) & (pred <= (-thresh1 +  0.2))] = -3#1
    thresholded_pred[pred <= -thresh2] = -5#2
    thresholded_pred = thresholded_pred.astype('int8')
    print "Testing"
    evaluate(y_test, thresholded_pred)
    
    y_test[y_test > 0] = 1
    y_test[y_test < 0] = -1
    
    thresholded_pred[thresholded_pred > 0] = 1
    thresholded_pred[thresholded_pred < 0] = -1
    
    evaluate(y_test, thresholded_pred)
    
    

    pred = model.predict([X_train, one_hot_train], batch_size = 100)#, Y_train, batch_size=32, verbose=1, sample_weight=None)
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
#    thresh1 = 1.0
#    thresh2 = 3.7
    thresholded_pred = copy.deepcopy(pred)
    thresholded_pred[(pred > (-thresh1 +  0.2)) & (pred < thresh2)] = 0
    thresholded_pred[(pred >= thresh1) & (pred < thresh2)] = 3#1
    thresholded_pred[pred >= thresh2] = 5#2
    thresholded_pred[(pred > -thresh2) & (pred <= (-thresh1 +  0.2))] = -3#1
    thresholded_pred[pred <= -thresh2] = -5#2
    thresholded_pred = thresholded_pred.astype('int8')
    print "Training"
    evaluate(y_train, thresholded_pred)
    y_train[y_train > 0] = 1
    y_train[y_train < 0] = -1
    
    thresholded_pred[thresholded_pred > 0] = 1
    thresholded_pred[thresholded_pred < 0] = -1
    
    evaluate(y_train, thresholded_pred)

    model_dup = duplicate_model('model_rnn')
    layer_output = model_dup.predict([X_test, one_hot_test], batch_size = 100)
    
##    get_last_layer_output = K.function([model.layers[0].input],
##                                  [model.layers[2].output])
#    get_last_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                  [model.layers[2].output])

## output in train mode = 0
##    layer_output = np.array(get_last_layer_output([X_train[0:1200], 0])[0])
#    
## output in train mode = 0
#    
##    X = [X_test, one_hot_test]
#    print X_test.shape
#    print one_hot_test.shape
#    print len(X_test)
#    print len(one_hot_test)
#    
#    
#    X_2 = np.concatenate((X_test, one_hot_test), axis=2)

#    start = 0
#    increment = 100
#    flag = 1
#    print len(X_test)
#    print "now!!"
#    while start+increment <= len(X_test):
##        X = [[X_test[start:start+increment], 1], [one_hot_test[start:start+increment], 1]]
#        if flag:
#            layer_output = get_last_layer_output([X_2[start:start+increment], 0])[0]#get_last_layer_output([[X_test[start:start+increment], 0], [one_hot_test[:, start:start+increment], 0]])[0]
#            flag = 0
#        else:
#            layer_output = np.concatenate((layer_output, get_last_layer_output([X_2[start:start+increment], 0])[0]))
#        start += increment
#    if start != len(X_test):
##        X = [X_test[start:start+increment], one_hot_test[start:start+increment]]
#        layer_output = np.concatenate((layer_output, get_last_layer_output([X_2[start:start+increment], 0])[0]))
    print "length of hidden", len(layer_output[0])    
    for iter in range(10):
        print df["tweet"][iter], layer_output[iter] 
        
sentiment_classifier()
