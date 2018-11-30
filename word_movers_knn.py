#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:02:20 2018

@author: sahil.maheshwari
"""

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.externals.joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
from sklearn.cross_validation import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.preprocessing import normalize

from pyemd import emd


class WordMoversKNN(KNeighborsClassifier):
    """K nearest neighbors classifier using the Word Mover's Distance.

    Parameters
    ----------
    
    W_embed : array, shape: (vocab_size, embed_size)
        Precomputed word embeddings between vocabulary items.
        Row indices should correspond to the columns in the bag-of-words input.

    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for Word Mover's Distance computation.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
    
    verbose : int, optional
        Controls the verbosity; the higher, the more messages. Defaults to 0.
        
    References
    ----------
    
    Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger
    From Word Embeddings To Document Distances
    The International Conference on Machine Learning (ICML), 2015
    http://mkusner.github.io/publications/WMD.pdf
    
    """
    _pairwise = False

    def __init__(self, W_embed, n_neighbors=1, n_jobs=1, verbose=False):
        self.W_embed = W_embed
        self.verbose = verbose
        super(WordMoversKNN, self).__init__(n_neighbors=n_neighbors, n_jobs=n_jobs,
                                            metric='precomputed', algorithm='brute')

    def _wmd(self, i, row, X_train):
        """Compute the WMD between training sample i and given test row.
        
        Assumes that `row` and train samples are sparse BOW vectors summing to 1.
        """
        union_idx = np.union1d(X_train[i].indices, row.indices)
        W_minimal = self.W_embed[union_idx]
        W_dist = euclidean_distances(W_minimal)
        bow_i = X_train[i, union_idx].A.ravel()
        bow_j = row[:, union_idx].A.ravel()
        return emd(bow_i, bow_j, W_dist)
    
    def _wmd_row(self, row, X_train):
        """Wrapper to compute the WMD of a row with all training samples.
        
        Assumes that `row` and train samples are sparse BOW vectors summing to 1.
        Useful for parallelization.
        """
        n_samples_train = X_train.shape[0]
        return [self._wmd(i, row, X_train) for i in range(n_samples_train)]

    def _pairwise_wmd(self, X_test, X_train=None):
        """Computes the word mover's distance between all train and test points.
        
        Parallelized over rows of X_test.
        
        Assumes that train and test samples are sparse BOW vectors summing to 1.
        
        Parameters
        ----------
        X_test: scipy.sparse matrix, shape: (n_test_samples, vocab_size)
            Test samples.
        
        X_train: scipy.sparse matrix, shape: (n_train_samples, vocab_size)
            Training samples. If `None`, uses the samples the estimator was fit with.
        
        Returns
        -------
        dist : array, shape: (n_test_samples, n_train_samples)
            Distances between all test samples and all train samples.
        
        """
        n_samples_test = X_test.shape[0]
        
        if X_train is None:
            X_train = self._fit_X

        dist = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self._wmd_row)(test_sample, X_train)
            for test_sample in X_test)

        return np.array(dist)

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values

        Parameters
        ----------
        X : scipy sparse matrix, shape: (n_samples, n_features)
            Training data. 

        y : {array-like, sparse matrix}
            Target values of shape = [n_samples] or [n_samples, n_outputs]

        """
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        return super(WordMoversKNN, self).fit(X, y)

    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X : scipy.sparse matrix, shape (n_test_samples, vocab_size)
            Test samples.

        Returns
        -------
        y : array of shape [n_samples]
            Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)
        dist = self._pairwise_wmd(X)
        return super(WordMoversKNN, self).predict(dist)
    
    
class WordMoversKNNCV(WordMoversKNN):
    """Cross-validated KNN classifier using the Word Mover's Distance.

    Parameters
    ----------
    W_embed : array, shape: (vocab_size, embed_size)
        Precomputed word embeddings between vocabulary items.
        Row indices should correspond to the columns in the bag-of-words input.

    n_neighbors_try : sequence, optional
        List of ``n_neighbors`` values to try.
        If None, tries 1-5 neighbors.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, StratifiedKFold is used.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for Word Mover's Distance computation.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    verbose : int, optional
        Controls the verbosity; the higher, the more messages. Defaults to 0.

    Attributes
    ----------
    cv_scores_ : array, shape (n_folds, len(n_neighbors_try))
        Test set scores for each fold.

    n_neighbors_ : int,
        The best `n_neighbors` value found.

    References
    ----------

    Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger
    From Word Embeddings To Document Distances
    The International Conference on Machine Learning (ICML), 2015
    http://mkusner.github.io/publications/WMD.pdf
    
    """
    def __init__(self, W_embed, n_neighbors_try=None, scoring=None, cv=3,
                 n_jobs=1, verbose=False):
        self.cv = cv
        self.n_neighbors_try = n_neighbors_try
        self.scoring = scoring
        super(WordMoversKNNCV, self).__init__(W_embed,
                                              n_neighbors=None,
                                              n_jobs=n_jobs,
                                              verbose=verbose)

    def fit(self, X, y):
        """Fit KNN model by choosing the best `n_neighbors`.
        
        Parameters
        -----------
        X : scipy.sparse matrix, (n_samples, vocab_size)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target
        """
        if self.n_neighbors_try is None:
            n_neighbors_try = range(1, 6)
        else:
            n_neighbors_try = self.n_neighbors_try

        X = check_array(X, accept_sparse='csr', copy=True)
        X = normalize(X, norm='l1', copy=False)

        cv = check_cv(self.cv, X, y)
        knn = KNeighborsClassifier(metric='precomputed', algorithm='brute')
        scorer = check_scoring(knn, scoring=self.scoring)

        scores = []
        for train_ix, test_ix in cv:
            dist = self._pairwise_wmd(X[test_ix], X[train_ix])
            knn.fit(X[train_ix], y[train_ix])
            scores.append([
                scorer(knn.set_params(n_neighbors=k), dist, y[test_ix])
                for k in n_neighbors_try
            ])
        scores = np.array(scores)
        self.cv_scores_ = scores

        best_k_ix = np.argmax(np.mean(scores, axis=0))
        best_k = n_neighbors_try[best_k_ix]
        self.n_neighbors = self.n_neighbors_ = best_k

        return super(WordMoversKNNCV, self).fit(X, y)