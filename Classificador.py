#!/usr/bin/python
# coding: utf-8
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB

class PipeLineFactory(object):

    def getSVC(self,config):
        map = {0: Pipeline([
                         ('vect', CountVectorizer(ngram_range=(1,2))),
                         ('tfidf', TfidfTransformer()),
                         ('reduction', TruncatedSVD(n_components=1000, n_iter=80, random_state=42)),
                         ('clf',SVC(kernel='linear', C=80,probability = True))
                        ]),
                1: Pipeline([
                        ('vect', CountVectorizer(ngram_range=(1, 2))),
                        ('tfidf', TfidfTransformer()),
                        ('reduction', TruncatedSVD(n_components=1000, n_iter=80, random_state=42)),
                        ('clf', BaggingClassifier(SVC(kernel='linear', C=80, probability=True)))
                        ]),
                2: Pipeline([
                        ('vect', CountVectorizer(ngram_range=(1, 2))),
                        ('tfidf', TfidfTransformer()),
                        ('reduction', TruncatedSVD(n_components=1000, n_iter=80, random_state=42)),
                        ('clf', AdaBoostClassifier(SVC(kernel='linear', C=80, probability=True)))
                    ])
               }
        return map[config]

    def getNaiveBayes(self,config):
        map = {0: Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('reduction', TruncatedSVD(n_components=1000, n_iter=80, random_state=42)),
            ('clf', MultinomialNB(probability=True))
        ])}