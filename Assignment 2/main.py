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
