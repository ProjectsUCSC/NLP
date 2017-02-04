

import preprocess as pp
import sys
import os
import subprocess


def create_input_file(df, data_dir, fname):
    cwd = os.getcwd()
    path = cwd + data_dir + fname
    #print path;
    f = open(path, "w")
    for item in df["text"]:
        print>>f, item
    f.close()
    return path

def obtm(sub_dir):
    cwd = os.getcwd()
    try:
        scr =  './runExample.sh'
        os.chdir(cwd + sub_dir)
        subprocess.call([scr])
        os.chdir(cwd)
    except:
        os.chdir(cwd)

def run_btm(btm_dir, csv_name):
    
    df = pp.preprocess(csv_name)
    input_fname ="0.txt"
    data_dir= "/" + btm_dir + "/sample-data/" 
    path = create_input_file(df, data_dir, input_fname) #create input text file in sample-data folder of OnlineBTM module
    sub_dir="/" +btm_dir + "/script"
    obtm(sub_dir)#runs the obtm script

#Usage
#csv_name = "Homework2_data.csv"
#btm_dir = "OnlineBTM"
#run_btm(btm_dir, csv_name)



#sub_dir="/" +btm_dir + "/script"
#obtm(sub_dir)





