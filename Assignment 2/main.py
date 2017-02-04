from preprocess import preprocess
from lda import lda_tm
import numpy as np
import pandas as pd

def main():
    filename = "Homework2_data.csv"
    df = preprocess(filename)
    [dictionary, model] = lda_tm(df["text"])
    
    print "Top 50 words of topic 0"
    words =  model.get_topic_terms(0, 50);
    for pair in words:
        print dictionary[pair[0]], pair[1]
    
    print "Most likely topics"
#    few_words = np.array(range(1,11))
    words = model.get_term_topics(19)
    print words
    print dictionary[19]
    for pair in words:
        print "word - > ", dictionary[pair[0]], pair
    

main()
