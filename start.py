#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:57:58 2018

@author: sahil.maheshwari
"""

import os
os.chdir('D:\Personal\Interviews\Racetrack')

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from word_movers_knn import WordMoversKNNCV

if not os.path.exists("data/embed.dat"):
    print("Caching word embeddings in memmapped format...")
    from gensim.models import KeyedVectors
    wv = KeyedVectors.load_word2vec_format("data/GoogleNews-vectors-negative300.bin.gz", binary=True)
    wv.init_sims(replace=True)
    print("Loaded word2vec format...")
    
    fp = np.memmap("data/embed.dat", dtype=np.double, mode='w+', shape=wv.vectors.shape)
    fp[:] = wv.vectors[:]
    print("Mapped.")
    
    with open("data/embed.vocab", "w") as f:
        for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
            print(w, file=f)
    del fp, wv

W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with open("data/embed.vocab") as f:
    vocab_list = map(str.strip, f.readlines())
    
vocab_dict = {w: k for k, w in enumerate(vocab_list)}



# getting data
newsgroups = fetch_20newsgroups()
docs, y = newsgroups.data, newsgroups.target

docs_train, docs_test, y_train, y_test = train_test_split(docs, y,
                                                          train_size=100,
                                                          test_size=300,
                                                          random_state=0)


# Since the W embedding array is pretty huge, we might as well restrict it to just the words that actually occur in the dataset
vect = CountVectorizer(stop_words="english").fit(docs_train + docs_test)
common = [word for word in vect.get_feature_names() if word in vocab_dict]
W_common = W[[vocab_dict[w] for w in common]]

# We can then create a fixed-vocabulary vectorizer using only the words we have embeddings for
vect = CountVectorizer(vocabulary=common, dtype=np.double)
X_train = vect.fit_transform(docs_train)
X_test = vect.transform(docs_test)


# model
knn_cv = WordMoversKNNCV(cv=3,
                         n_neighbors_try=range(1, 20),
                         W_embed=W_common, verbose=5, n_jobs=3)
knn_cv.fit(X_train, y_train)


# evaluation
print("CV score: {:.2f}".format(knn_cv.cv_scores_.mean(axis=0).max()))
print("Test score: {:.2f}".format(knn_cv.score(X_test, y_test)))