
# Make vocabulary and frequencies per token:

from sklearn.feature_extraction.text import CountVectorizer

train_set = ("The sky is blue.", "The sun is bright.")
test_set = ("The sun in the sky is bright.", "We can see the shining sun, the bright sun.")

vectorizer = CountVectorizer(stop_words=None)

vectorizer.fit_transform(train_set)

print(vectorizer.vocabulary_)



from nltk import FreqDist

word_dist = FreqDist()
for s in train_set:
    print(s)
    #print(s.split())
    word_dist.update(s.split())

print(dict(word_dist))


# https://stackabuse.com/python-for-nlp-creating-bag-of-words-model-from-scratch/
#----------------------------------------------------------------------------------

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
