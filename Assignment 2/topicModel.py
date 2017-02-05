
from lda import lda_tm
import numpy as np
import pandas as pd
import btm

import gensim
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from time import gmtime, strftime
import numpy as np
    
    
#usage
#csv_name = "Homework2_data.csv"
#df = pp.preprocess(csv_name)
def run_btm_tm(csv_name, df, num_topics , top_topics ): 
    print  "================BTM===================="
    print "running topic modelling using btm"
    btm_dir = "OnlineBTM" #directory where btm code is .
    btm.run_btm(btm_dir, csv_name, df, num_topics, top_topics)
    print "topic modelling using btm complete."
    
def run_lda_tm(texts, no_topics, top_words):


    print  "================LDA===================="
    print "running topic modelling using LDA"
    lda_output = "lda_output.txt"
#    texts = ["dog cat banana", "fruit apple bird wolf lion", "orange tomato"]
    texts = [[word for word in text.split()] for text in texts]
#    print texts

    dictionary = corpora.Dictionary(texts)
#    for d in dictionary.keys():
#        print d, dictionary[d]

    corpus = [dictionary.doc2bow(text) for text in texts]
#    print corpus
    model = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=no_topics, update_every=1, passes=1)
    print "number of topics = ", no_topics
#    topics = model.show_topics(num_topics=no_topics, num_words=top_words)
#    print "These are our topics\n", topics
    
    f = open(lda_output, "w")
    for i in range(no_topics):
        words =  model.get_topic_terms(i, top_words);
        f.write("Topic " + str(i) + ":\n")
        for pair in words:
            f.write(str(dictionary[pair[0]]) + " " + str(pair[1]) + ",")
        f.write("\n")
    f.close()

    return
    
    #get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
    #Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.

    #Ignore topics with very low probability (below minimum_probability).

    #get_term_topics(word_id, minimum_probability=None)
    #Returns most likely topics for a particular word in vocab.

    #get_topic_terms(topicid, topn=10)
    #Return a list of (word_id, probability) 2-tuples for the most probable words in topic topicid.

    #Only return 2-tuples for the topn most probable words (ignore the rest).
