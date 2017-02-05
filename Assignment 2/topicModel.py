
from lda import lda_tm
import numpy as np
import pandas as pd
import btm


def run_lda_tm(df):
    [dictionary, model] = lda_tm(df["text"])
    print "Top 50 words of topic 0"
    words =  model.get_topic_terms(0, 30);
    for pair in words:
        print dictionary[pair[0]], pair[1]

    print "Most likely topics"
#    few_words = np.array(range(1,11))
    words = model.get_term_topics(19)
    print words
    print dictionary[19]
    for pair in words:
        print "word - > ", dictionary[pair[0]], pair
    
#usage
#csv_name = "Homework2_data.csv"
#df = pp.preprocess(csv_name)
def run_btm_tm(csv_name, df, num_topics , top_topics ): 
    print  "================BTM===================="
    print "running topic modelling using btm"
    btm_dir = "OnlineBTM" #directory where btm code is .
    btm.run_btm(btm_dir, csv_name, df, num_topics, top_topics)
    print "topic modelling using btm complete."


