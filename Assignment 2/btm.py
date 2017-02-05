import os
import subprocess

def create_input_file(df, data_dir, fname):
    cwd = os.getcwd()
    path = cwd + data_dir + fname
    #print path;
    f = open(path, "w")
    for item in df["text"]:
        item = item.strip(" ")
        if(item):
            print>>f, item
    f.close()
    return path


def obtm(sub_dir, num_topics , top_topics):
    cwd = os.getcwd()
    try:
        scr =  './runExample.sh'
        os.chdir(cwd + sub_dir)
        subprocess.call([scr, str(num_topics), str(top_topics)])
        os.chdir(cwd)
    except:
        print "error while executing the shell script "
        os.chdir(cwd)

def run_btm(btm_dir, csv_name, df, num_topics, top_topics):
    input_fname ="0.txt"
    data_dir= "/" + btm_dir + "/sample-data/" 
    path = create_input_file(df, data_dir, input_fname)
    sub_dir="/" +btm_dir + "/script"
    obtm(sub_dir, num_topics, top_topics)


