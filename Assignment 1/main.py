
from preprocess import *
from feature_extraction import *
from lsi import *
from kmeans import *
import CMUTweetTagger


def get_feature_sets(filename):

    df = preprocess(filename)
    print "length of data", len(df)

    data_sample = df['text'].str.lower()

    #TFIDF features without text processing
    [data_fs2, vectorizer, no_features] = vectorize(data_sample, TFIDF) #feature set 2
    #Unigram features without text processing
    [data_fs1, vectorizer, no_features] = vectorize(data_sample, UNI) #Feature set 1

    #Text preprocessing - stopwords, stemming, lowercase
    data_fs3 = tokenize_and_stopwords(data_sample)
    data_fs3 = stemmer(data_fs3)
    #use CMU tagger and remove NNP and NNPS

    print "CMU tagger"
    all_tags = CMUTweetTagger.runtagger_parse(data_fs3)
    for i in range(len(all_tags)):
        for tag in all_tags[i]:
#            print tag[1]
            if tag[1] == 'NNP' or tag[1] == 'NNPS':
                data_fs3[i] = data_fs3[i].replace(tag[0], '')

    [data_fs3, vectorizer, no_features] = vectorize(data_fs3, TFIDF) #Feature set 3
    data_fs4 = lsa(data_fs3) #feature set 4
    print data_fs1.shape
    print data_fs2.shape
    print data_fs3.shape
    print data_fs4.shape
    return [data_fs1, data_fs2, data_fs3, data_fs4, df]

def kmeans_analysis(filename):
    no_clusters = 5
    filename = "clinton-50k.csv"
    [data_fs1, data_fs2, data_fs3, data_fs4, df1] = get_feature_sets(filename)
    df  = df1[['tweet_id','text', 'retweets']].copy()
    df['cluster_fs1'] = run_kmeans(data_fs1, 7)
    df['cluster_fs2'] = run_kmeans(data_fs2, 7)
    df['cluster_fs3'] = run_kmeans(data_fs3, 7)
    df['cluster_fs4'] = run_kmeans(data_fs4, 7)
    result_filename = filename.replace(".csv", "") +"-result.csv"
    df.to_csv(result_filename)
    return df


def perform_analysis(df):
    print df.groupby(['cluster_fs1']).describe()
    print df.groupby(['cluster_fs2']).describe()
    print df.groupby(['cluster_fs3']).describe()
    print df.groupby(['cluster_fs4']).describe()
    df.corr()
    result1 = df.sort(['cluster_fs1'])
    result2 = df.sort(['cluster_fs2'])
    result3 = df.sort(['cluster_fs3'])
    result4 = df.sort(['cluster_fs4'])
    result4



def main():
    f1 = "trump-50k.csv"
    f2 = "clinton-50k.csv"
    df1 = kmeans_analysis(f1)
    perform_analysis(df1)
    df2 = kmeans_analysis(f2)
    perform_analysis(df2)



main()
