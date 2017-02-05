from preprocess import preprocess
from lda import lda_tm
import numpy as np
import pandas as pd

def main():
    filename = "Homework2_data.csv"
    no_topics = 5
    no_words = 50
    lda_output = "lda_output.txt"
    f = open(lda_output, "w")
    df = preprocess(filename)
    [dictionary, model] = lda_tm(df["text"], no_topics)
    
    print "Top 50 words of topic 0"
    for i in range(no_topics):
        words =  model.get_topic_terms(i, no_words);
        f.write("Topic " + str(i) + ":\n")
        for pair in words:
            f.write(str(dictionary[pair[0]]) + " " + str(pair[1]) + ",")
        f.write("\n")
        
#    print "Most likely topics"
##    few_words = np.array(range(1,11))
#    words = model.get_term_topics(19)
#    print words
    print dictionary[19]
#    for pair in words:
#        print "word - > ", dictionary[pair[0]], pair
    f.close()

main()
