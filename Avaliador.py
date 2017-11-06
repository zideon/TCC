#!/usr/bin/python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,HashingVectorizer
from sklearn.preprocessing import Normalizer, OneHotEncoder,LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score,accuracy_score,f1_score,precision_score
import re




import numpy as np
from sklearn import svm, datasets, feature_selection
from sklearn.model_selection import cross_val_score,StratifiedKFold,StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

class AvaliadorFactory(object):

    def divisaoTrainTest(self,text_clf,targets, X, y):
        X_train, X_test, y_train, y_test= train_test_split(
            X, y, test_size=0.4, random_state=0)
        text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_test)
        print(metrics.classification_report(y_test, predicted,
                                            target_names=targets))
        # confusion_mat = metrics.confusion_matrix(y_test, predicted)
        # print('matrix de confusão:')
        # print(confusion_mat)
        # gg =GerenciadorGraficos()
        # gg.plot_confusion_matriz(y_test, predicted)
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
                                   X, y, scoring='accuracy', cv=StratifiedKFold(n_splits=kfounds))
        print "Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%"

        f1 = cross_val_score(text_clf,
                             X, y, scoring='f1_weighted', cv=StratifiedKFold(n_splits=kfounds))
        print "F1: " + str(round(100 * f1.mean(), 2)) + "%"

        precision = cross_val_score(text_clf,
                                    X, y, scoring='precision_weighted', cv=StratifiedKFold(n_splits=kfounds))
        print "Precision: " + str(round(100 * precision.mean(), 2)) + "%"

        recall = cross_val_score(text_clf,
                                 X, y, scoring='recall_weighted', cv=StratifiedKFold(n_splits=kfounds))
        print "Recall: " + str(round(100 * recall.mean(), 2)) + "%"
    def StratifyCrossValidationTest(self,text_clf,targets, X, y, kfounds):
        X = np.array(X)
        y = np.array(y)
        skf = StratifiedShuffleSplit(n_splits=kfounds)
        skf.get_n_splits(X, y)
        targetsResults = {}
        newTargets = targets[:]
        newTargets.append('avg')
        for i in newTargets:
            targetsResults[i] = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            text_clf.fit(X_train, y_train)
            predicted = text_clf.predict(X_test)

            m = metrics.classification_report(y_test, predicted,
                                                target_names=targets)
            m = m.replace("\n\n","\n")
            array = m.split("\n")
            for i in range(len(newTargets)):
                targetsResults[newTargets[i]].append(re.findall(r"[-+]?\d*\.\d+|\d+",array[i+1]))
        for i in newTargets:
            print i
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            for j in targetsResults[i]:
                precision = precision +float(j[0])
                recall = recall +float(j[1])
                f1 = f1 +float(j[2])
            print 'precision:'+str(precision/kfounds)
            print 'recall:'+str(recall/kfounds)
            print 'f1:'+str(f1/kfounds)

class GerenciadorGraficos(object):


    def plot_ROC_curve(self,classifier, X, y, pos_label=1, n_folds=5):
        X = np.array(X)
        y = np.array(y)
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        for i, (train, test) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
            train = np.array(train)
            test = np.array(test)
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area under the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
        mean_tpr /= n_folds
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--',
                 label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def plot_PR_curve(self,classifier, X, y, n_folds=5):
        X = np.array(X)
        y = np.array(y)
        """
        Plot a basic precision/recall curve.
        """
        for i, (train, test) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
            train = np.array(train)
            test = np.array(test)
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1],
                                                                   pos_label=1)
            plt.plot(recall, precision, lw=1, label='PR fold %d' % (i,))
            #  clf_name = str(type(classifier))
            # clf_name = clf_name[clf_name.rindex('.')+1:]
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-recall curve')
        plt.legend(loc="lower right")
        plt.show()
    def plot_confusion_matriz(self,y_true,y_pred):
        confusion_mat = confusion_matrix(y_true, y_pred)
        plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Paired)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def getAnova(self, X, y):

        # y = y[:200]
        # X = X[:200]
        X = LabelEncoder().fit_transform(X.ravel()).reshape(*X.shape)
        # transform to binary
        # X = OneHotEncoder().fit_transform(X_int).toarray()

        n_samples = len(y)
        X = X.reshape((n_samples, -1))
        # add 200 non-informative features
        X = np.hstack((X, 2 * np.random.random((n_samples, 200))))

        transform = feature_selection.SelectPercentile(feature_selection.f_classif)



        clf = Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])

        # #############################################################################
        # Plot the cross-validation score as a function of percentile of features
        score_means = list()
        score_stds = list()
        percentiles = (5, 10, 20, 40, 60, 80, 100)

        for percentile in percentiles:
            clf.set_params(anova__percentile=percentile)
            # Compute cross-validation score using 1 CPU
            this_scores = cross_val_score(clf, X, y, n_jobs=1,verbose=10,cv=3)
            score_means.append(this_scores.mean())
            score_stds.append(this_scores.std())

        plt.errorbar(percentiles, score_means, np.array(score_stds))

        plt.title(
            'Performance of the SVM-Anova varying the percentile of features selected')
        plt.xlabel('Percentile')
        plt.ylabel('Prediction rate')

        plt.axis('tight')
        plt.show()