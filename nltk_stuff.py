
# Make vocabulary and frequencies per token:

from sklearn.feature_extraction.text import CountVectorizer

train_set = ("The sky is blue.", "The sun is bright.")
test_set = ("The sun in the sky is bright.", "We can see the shining sun, the bright sun.")

vectorizer = CountVectorizer(stop_words=None)

vectorizer.fit_transform(train_set)

print(vectorizer.vocabulary_)


from nltk import FreqDist

word_dist = FreqDist()

for z, s in enumerate(train_set):
    print('{} {}'.format(z, s))
    #print(s.split())                   # split each sentence by words
    word_dist.update(s.split())

print(dict(word_dist))


# https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/

import nltk  
import numpy as np  
import random  
import string

import bs4 as bs  
import urllib.request  
import re  


raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/Natural_language_processing')  
raw_html = raw_html.read()

article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('p')    #filter the text within the paragraph text

#create a complete corpus by concatenating all the paragraphs.
article_text = ''
for para in article_paragraphs:  
    article_text += para.text

#split the corpus into individual sentences
corpus = nltk.sent_tokenize(article_text)

# Preprocess:
for i in range(len(corpus )):
    corpus [i] = corpus [i].lower()
    corpus [i] = re.sub(r'\W',' ',corpus [i])
    corpus [i] = re.sub(r'\s+',' ',corpus [i])

# number of sentences / documents in corpus
print(len(corpus))

print(corpus[30])

#okenize the sentences in the corpus and create a dictionary 
#that contains words and their corresponding frequencies in the corpus
wordfreq = {}
for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1


import heapq
most_freq = heapq.nlargest(200, wordfreq, key=wordfreq.get)

# BAG OF WORDS representation (=vector representation per sentence/document)
sentence_vectors = []
for sentence in corpus:
    sentence_tokens = nltk.word_tokenize(sentence)
    sent_vec = []
    for token in most_freq:
        if token in sentence_tokens:
            sent_vec.append(1)
        else:
            sent_vec.append(0)
    sentence_vectors.append(sent_vec)

sentence_vectors = np.asarray(sentence_vectors)


##############################################################################################
#############################################################################################


import numpy as np                # import numpy module
#import matplotlib.pyplot as plt
#from matplotlib import rc               # for LaTex
from os import chdir
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#import re
import spacy
#nltk.download('punkt') 

default_path = "C:\\Users\\Alexander\\Documents\\Arbeit\\Allianz\\"

chdir(default_path)

# Print out the topics from LDA:
#----------------------------------------------------------------------------------
def print_topics(topics, feature_names, sorting, topics_per_chunk=6, n_words=20):
    for i in range(0, len(topics), topics_per_chunk):
        # for each chunk:
        these_topics = topics[i: i + topics_per_chunk]
        # maybe we have less than topics_per_chunk left
        len_this_chunk = len(these_topics)
        # print topic headers
        print(("topic {:<8}" * len_this_chunk).format(*these_topics))
        print(("-------- {0:<5}" * len_this_chunk).format(""))
        # print top n_words frequent words
        for i in range(n_words):
            try:
                print(("{:<14}" * len_this_chunk).format(
                    *feature_names[sorting[these_topics, i]]))
            except:
                pass
        print("\n")
#-------------------------------------------------------------------------------

# Import data
#----------------------------------------------------------
with open("col.txt",mode = "r") as f:  
  data = pd.read_table(f, header=0)
#----------------------------------------------------------

data.head(100)

any(data.isna())

data_nona = data.dropna()

causes = data_nona.iloc[:,0].tolist()
causes

#my_stopwords = {'and','the'}

print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

orig_sw = ENGLISH_STOP_WORDS
#sw = list(orig_sw)
my_incl = {'strong'}
my_except = {'third', 'tp', 'by', 'involving'}   # set
new_stop_w = (orig_sw | my_incl) - my_except        # difference set A \ B

# set minimum number of documents/strings 
# a token needs to appear in
# not include english stopwords
#------------------------------------------
new_regex = '(?u)(?:(?!\d)\w)+\\w+'
#new_regex = '(?u)\\b\\w\\w+\\b'

vect = CountVectorizer(max_features=100, max_df=0.1,
                       lowercase=True, token_pattern = new_regex, 
                       ngram_range=(2, 3), tokenizer = None,     # use stemmer or lemmatization
                       stop_words = new_stop_w).fit(causes)    

#ps = PorterStemmer()
#sentence = "gaming, the gamers play games"
#words = word_tokenize(sentence)
#for word in words:
#    print(word + ":" + ps.stem(word))


# Fine tune regular expression:
#---------------------------------
#regex1 = re.compile('(?u)\\b\\w\\w+\\b')      # original
# Removes digits!
#regex2 = re.compile('(?u)(?:(?!\d)\w)+\\w+')    # modified, works better
# (?!\d)\w: A position that is not followed by a digit, and then \w. 
# to match multiple of these enclose in brackets and +

#i=0
#i+=1
#st = causes[i] ; print(st)
#regex1.findall(st)
#regex2.findall(st)
#---------------------------------------------

#help(CountVectorizer)

print("Vocabulary size: {}".format(len(vect.vocabulary_)))

causes

# Vocabulary
vect.vocabulary_

print("Vocabulary:\n{}".format(vect.get_feature_names()))

# Construct bag of words collection:
# How often does each word/token appear in each text 
# in the corpus 
#---------------------------------------------------
bag_of_words = vect.transform(causes)         

# Show attributes of that object:
#dir(vect)

print("bag_of_words: {}".format(repr(bag_of_words)))   # Return a string containing a printable representation of an object

#print("Dense representation of bag_of_words:\n{}".format(bag_of_words.toarray()))

# load spacy's English-language models
#en_nlp = spacy.load('en')
#nlp = spacy.load('en_core_web_sm')


# instantiate nltk's Porter stemmer
#stemmer = nltk.stem.PorterStemmer()

#dir(stemmer)

#stemmer.stem(token.norm_.lower())

# define function to compare lemmatization in spacy with stemming in nltk
#def compare_normalization(doc):
    # tokenize document in spacy
    #doc_spacy = en_nlp(doc)
    # print lemmas found by spacy
    #print("Lemmatization:")
    #print([token.lemma_ for token in doc_spacy])
    # print tokens found by Porter stemmer
    #print("Stemming:")
    #print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])
    
nof_topics = 10

# Train the topic model:
#---------------------------
lda = LDA(n_components=nof_topics, learning_method="batch", max_iter=25, random_state=33)

# We build the model and transform the data in one step
# Computing transform takes some time,
# and we can save time by doing both at once
X = vect.fit_transform(causes)

document_topics = lda.fit_transform(X)

print("lda.components_.shape: {}".format(lda.components_.shape))


# for each topic (a row in the components_), sort the features (ascending).
# Invert rows with [:, ::-1] to make sorting descending
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# get the feature names from the vectorizer:
feature_names = np.array(vect.get_feature_names())

# Call:
#---------
print_topics(topics = range(nof_topics), feature_names=feature_names,
              sorting=sorting, topics_per_chunk=1, n_words=10)


    
#tf = TfidfVectorizer(min_df=5)
#pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None),
 #                    LogisticRegression())
#param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}
#grid = GridSearchCV(pipe, param_grid, cv=5)
#grid.fit(text_train, y_train)
#print("Best cross-validation score: {:.2f}".format(grid.best_score_))
 

### Adapted version!!!!!!!!!!!!!!!!!!!!!!!

def build_freqs(text_list, ys):
    """Build frequencies.
    Input:
        text: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    text = text_list.tolist()
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for z, (y, sentence) in enumerate(zip(yslist, text)):
        print(z)
        print(sentence)
        tokens = nltk.word_tokenize(sentence)
        for word in tokens:
            #print(word)
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs

train_set = causes        # see below

dat = pd.DataFrame(train_set, columns=['text'])
dat['label'] = pd.DataFrame(np.random.randint(0,9,len(train_set)),index=dat.index)
dat

freqs = build_freqs(dat.text, dat.label)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))


yslist = np.squeeze(dat.label).tolist()
yslist[0]


for z, (y, sentence) in enumerate(zip(yslist, text)):
    print(z)
    print(sentence)
    tokens = nltk.word_tokenize(sentence)
    print(tokens)


def extract_features(single_sentence, freqs):
    '''
    Input: 
        single_sentence: a list of words for one sentence
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,class +1)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    word_l = process_tweet(single_sentence)              # change!!!

    nof_class = 2

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, nof_class + 1))     # + bias term 
    
    #bias term is set to 1
    x[0,0] = 1 
        
    # loop through each word in the list of words
    for word in word_l:
        
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word, 1.0), 0.)                   # set default dict value 0. if key does not exist
        
        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word, 0.0), 0.)
        
    assert(x.shape == (1, nof_class + 1))
    return x