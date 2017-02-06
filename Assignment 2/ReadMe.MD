Topic Modelling for Twitter 
========================================

Pre-requisite python libraries :
================================
pip install pyenchant
pip install gensim

Python Files : 
========================
1. preprocess.py
2. topicModel.py
3. main.py
 
Supporting Files (for topic modelling):
=========================
1. btm.py
2. lda.py
 
Output files:
===========================
1. btm.txt (in OnlineBTM/output)
2. lda_output.txt 
 
Dependencies :
=========================
1. NLTK
2. Gensim
3. Numpy
4. OBTM code (use the code provided, not the one from github)
  
How to run :
==========================
1.  Without changing parameters :   python main.py
2.  With different number of topics :  python main.py number_of_topics number_of_top_words
  
  Currently the code expects the OnlineBTM code to present as a directory at head level .  
  To change the location/name of the directory , modify btm.py
