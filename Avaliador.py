#!/usr/bin/python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss
import numpy as np

class AvaliadorFactory(object):

    def divisaoTrainTest(self,text_clf,targets, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0)
        text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_test)
        print(metrics.classification_report(y_test, predicted,
                                            target_names=targets))
        confusion_mat = metrics.confusion_matrix(y_test, predicted)
        print('matrix de confusão:')
        print(confusion_mat)