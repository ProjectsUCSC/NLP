def lda_tm(texts, no_topics):

    import gensim
    from gensim import corpora, models
    from nltk.corpus import stopwords
    from nltk.tokenize import RegexpTokenizer
    from time import gmtime, strftime
    import numpy as np


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
    topics = model.show_topics(num_topics=no_topics, num_words=50)
    print "These are our topics\n", topics
    
    return [dictionary, model]

    #get_document_topics(bow, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
    #Return topic distribution for the given document bow, as a list of (topic_id, topic_probability) 2-tuples.

    #Ignore topics with very low probability (below minimum_probability).

    #get_term_topics(word_id, minimum_probability=None)
    #Returns most likely topics for a particular word in vocab.

    #get_topic_terms(topicid, topn=10)
    #Return a list of (word_id, probability) 2-tuples for the most probable words in topic topicid.

    #Only return 2-tuples for the topn most probable words (ignore the rest).
