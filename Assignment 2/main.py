from preprocess import preprocess
import topicModel as tm
import sys

def main(num_topics = 5, top_words = 35):
    filename = "Homework2_data.csv"
    df = preprocess(filename)

    tm.run_lda_tm(df["text"], num_topics, top_words)
    tm.run_btm_tm(filename, df, num_topics, top_words)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Running Topic Modelling with default values k = 5, n =35"
        main()
    elif len(sys.argv) !=3 :
        print "please give either 2 or 0 arguments ."
        print "usage :  python main.py num_topics top_words"
        exit(1)
    else:
        try:
            num_topics = int(sys.argv[1])
            top_words = int(sys.argv[2])
            main(num_topics, top_words)
        except:
            print("please enter numerical values for number of topics and top topics")
            exit(1)
            
#from preprocess import preprocess
#from lda import lda_tm
#import numpy as np
#import pandas as pd

#def main():
#    filename = "Homework2_data.csv"
#    no_topics = 5
#    no_words = 50
#    lda_output = "lda_output.txt"
#    f = open(lda_output, "w")
#    df = preprocess(filename)
#    [dictionary, model] = lda_tm(df["text"], no_topics)
#    
#    print "Top 50 words of topic 0"
#    for i in range(no_topics):
#        words =  model.get_topic_terms(i, no_words);
#        f.write("Topic " + str(i) + ":\n")
#        for pair in words:
#            f.write(str(dictionary[pair[0]]) + " " + str(pair[1]) + ",")
#        f.write("\n")
#        
##    print "Most likely topics"
###    few_words = np.array(range(1,11))
##    words = model.get_term_topics(19)
##    print words
#    print dictionary[19]
##    for pair in words:
##        print "word - > ", dictionary[pair[0]], pair
#    f.close()

#main()
