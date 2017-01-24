from preprocess import *
from feature_extraction import *
from feature_selection import *
from kmeans import *

def main():
    filename = "clinton-50k.csv"

    df = preprocess(filename)
    print "length of data", len(df)

    data_sample = df['text'].str.lower()

    #TFIDF features without text processing
    [data_tfidf, vectorizer, no_features] = vectorize(data_sample, TFIDF) #feature set 2
    #Unigram features without text processing
    [data_uni, vectorizer, no_features] = vectorize(data_sample, UNI) #Feature set 1

    #Text preprocessing - stopwords, stemming, lowercase
    preprocessed_data_sample = tokenize_and_stopwords(data_sample)
    preprocessed_data_sample = stemmer(preprocessed_data_sample)
    print preprocessed_data_sample[0:5]

    #use CMU tagger and remove NNP and NNPS
    
#    -----To be done------
#------------------------------------------     

    #TFIDF features with text processing
    [data_tfidf_plus, vectorizer, no_features] = vectorize(data_sample, TFIDF) #Feature set 3
#    #Unigram features with text processing
#    [data_uni_plus, vectorizer, no_features] = vectorize(data_sample, UNI)
   
    #Feature selection - LSA
    X = lsa(data_tfidf_plus) #feature set 4
    print X.shape
    
#    Run kmeans
    no_clusters = 5
    labels = run_kmeans(X, no_clusters)
    print labels
main()
