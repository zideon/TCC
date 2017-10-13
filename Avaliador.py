#!/usr/bin/python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_score

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
    def comparacaoBases(self,text_clf,targets,X_train,y_train, X_test,  y_test):
        text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_test)
        print(metrics.classification_report(y_test, predicted,
                                            target_names=targets))
        confusion_mat = metrics.confusion_matrix(y_test, predicted)
        print('matrix de confusão:')
        print(confusion_mat)
    def crossValidationTest(self,text_clf, X, y, kfounds):
        accuracy = cross_val_score(text_clf,
                                   X, y, scoring='accuracy', cv=kfounds)
        print "Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%"

        # f1 = cross_val_score(text_clf,
        #                      X, y, scoring='f1_weighted', cv=kfounds)
        # print "F1: " + str(round(100 * f1.mean(), 2)) + "%"
        #
        # precision = cross_val_score(text_clf,
        #                             X, y, scoring='precision_weighted', cv=kfounds)
        # print "Precision: " + str(round(100 * precision.mean(), 2)) + "%"
        #
        # recall = cross_val_score(text_clf,
        #                          X, y, scoring='recall_weighted', cv=kfounds)
        # print "Recall: " + str(round(100 * recall.mean(), 2)) + "%"
