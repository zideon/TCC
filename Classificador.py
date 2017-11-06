#!/usr/bin/python
# coding: utf-8
import StringIO
from sklearn.base import TransformerMixin

from sklearn.feature_selection import SelectKBest, chi2,f_classif, mutual_info_classif

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import brier_score_loss

from mlxtend.preprocessing import DenseTransformer

from scipy import sparse
import pandas as pd
from sklearn.preprocessing import Normalizer, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, KernelPCA, LatentDirichletAllocation
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.model_selection import cross_val_score, train_test_split
from Modelo import EntradaCluster, Cluster
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel,RFE
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


class Constantes(object):
    SVC = 0
    NAIVE_BAYES = 1
    RANDOM_FLOREST = 2
    VOTING_CLASSIFIER = 3
    RIDGE_CLASSIFIER = 4
    PASSIVE_AGRESSIVE_CLASSIFIER = 5
    LDA = 6
    QDA = 7
    LINEAR_SVC = 8

    NORMAL = 0
    BAGGING = 1
    BOOSTING = 2

    COUNT_VECTORIZER = 0
    TFIDF_VECTORIZER = 1
    WEIGHT_COUNT = 2
    WEIGHT_TFIDF = 3
    COUNT_VECTORIZER_BIGRAM = 4
    TFIDF_VECTORIZER_BIGRAM = 5

    LSA = 0
    LATENT = 1

    SELECTK_CHIE2 = 0
    SELECTK_F_CLASSIF = 1
    SELECTK_MULTI_F_CLASSIF = 2
    SELECT_SVC = 3
    SELECT_BAYES = 4
    RF_SVC = 5


    NORMALIZING = 0
    ONEHOT = 1
    DENSE = 2


class PipeLineFactory(object):
    def getDecomposition(self, config):
        map = {
            0: TruncatedSVD(n_components=1000, n_iter=80, random_state=42),
            1: LatentDirichletAllocation(n_topics=300, max_iter=50,
                                         learning_method='online',
                                         learning_offset=50.,
                                         random_state=42)
        }
        return ('decomposion_' + str(config), map[config])

    def getDecompositionArray(self, array):
        features = []
        for i in array:
            features.append(self.getDecomposition(i))
        return ('decomposion_union', FeatureUnion(features))

    def getFeatureSelection(self, config):
        map = {
            0: SelectKBest(chi2, k=1500),
            1: SelectKBest(f_classif, k=1500),
            2: SelectKBest(mutual_info_classif, k=1500),
            3: SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                         tol=1e-3, class_weight='balanced', C=50)),
            4: SelectFromModel(MultinomialNB()),
            5: RFE(estimator=LinearSVC(penalty="l1", dual=False,
                                         tol=1e-3, class_weight='balanced', C=50), n_features_to_select=1500, step=1),
            6: SelectFromModel(RandomForestClassifier(max_depth=300, random_state=42, n_jobs=1))
        }
        return ('selection_' + str(config), map[config])

    def getFeatureSelectionArray(self, array):
        features = []
        for i in array:
            features.append(self.getFeatureSelection(i))
        return ('select_union', FeatureUnion(features))

    def getFeaturesExtraction(self, config):
        map = {
            0: CountVectorizer(ngram_range=(1, 1)),
            1: TfidfVectorizer(ngram_range=(1, 1)),
            2: WeightCountVectorizerTransformer(tag="_c",weight=10),
            3: WeightTFIDFVectorizerTransformer(tag="_c", weight=10),
            4: CountVectorizer(ngram_range=(1, 2)),
            5: TfidfVectorizer(ngram_range=(1, 2))
        }
        return ('extraction_' + str(config), map[config])

    def getLinearSVC(self, config):
        map = {
            0: LinearSVC(penalty="l2", dual=False,
                         tol=1e-3, class_weight='balanced', C=50),
            1: BaggingClassifier(n_estimators=200, random_state=42, n_jobs=3,
                                 base_estimator=LinearSVC(penalty="l2", dual=False,
                                                          tol=1e-3, class_weight='balanced', C=50)),
            2: AdaBoostClassifier(
                LinearSVC(penalty="l1", dual=False,
                          tol=1e-3, class_weight='balanced', C=50),
                algorithm="SAMME",
                n_estimators=200)
        }
        return ('clf_' + str(config), map[config])

    def getSVC(self, config):
        map = {
            0: SVC(kernel='linear', C=80, random_state=42, probability=True, class_weight='balanced'),

            1: BaggingClassifier(n_estimators=100, random_state=42, n_jobs=3,
                                 base_estimator=SVC(kernel='linear', C=80, class_weight='balanced', random_state=42,
                                                    probability=True)),
            # 1: BaggingClassifier(n_estimators=50, random_state=42, n_jobs=3,
            #                      base_estimator=SVC(kernel='rbf',gamma=0.01, C=50, class_weight='balanced', random_state=42,
            #                                         probability=True)),
            2: AdaBoostClassifier(
                SVC(kernel='linear', C=80, class_weight='balanced', random_state=42, probability=True),
                algorithm="SAMME",
                n_estimators=200)
        }
        return ('clf_' + str(config), map[config])

    def getRandomFlorest(self, config):
        map = {
            0: RandomForestClassifier(max_depth=300, random_state=42, n_jobs=1),
            1: BaggingClassifier(n_estimators=200, random_state=42, n_jobs=3,
                                 base_estimator=RandomForestClassifier(max_depth=1000, random_state=42)),
            2: AdaBoostClassifier(RandomForestClassifier(max_depth=1000, random_state=42), algorithm="SAMME",
                                  n_estimators=200)
        }
        return ('clf_' + str(config), map[config])

    def getNaiveBayes(self, config):
        map = {
            0: MultinomialNB(),
            1: BaggingClassifier(n_estimators=200, random_state=42, n_jobs=3,
                                 base_estimator=MultinomialNB(class_prior=[2, 1])),
            2: AdaBoostClassifier(MultinomialNB(), algorithm="SAMME", n_estimators=200)
        }
        return ('clf_' + str(config), map[config])

    def getLDA(self, config):
        map = {
            0: LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            1: BaggingClassifier(n_estimators=200, random_state=42, n_jobs=3,
                                 base_estimator=LinearDiscriminantAnalysis(n_components=1000, solver='lsqr',
                                                                           shrinkage='auto')),
            2: AdaBoostClassifier(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'), algorithm="SAMME",
                                  n_estimators=200)
        }
        return ('clf_' + str(config), map[config])

    def getQDA(self, config):
        map = {
            0: QuadraticDiscriminantAnalysis(),
            1: BaggingClassifier(n_estimators=200, random_state=42, n_jobs=3,
                                 base_estimator=QuadraticDiscriminantAnalysis()),
            2: AdaBoostClassifier(QuadraticDiscriminantAnalysis())
        }
        return ('clf_' + str(config), map[config])

    def getRidgeClassifier(self, config):
        map = {
            0: RidgeClassifier(tol=1e-2, solver="auto"),
            1: BaggingClassifier(n_estimators=200, random_state=42, n_jobs=3,
                                 base_estimator=RidgeClassifier(class_weight="balanced",tol=1e-3, solver="auto")),
            2: AdaBoostClassifier(RidgeClassifier(tol=1e-2, solver="lsqr"), algorithm="SAMME", n_estimators=200)
        }
        return ("clf_" + str(config), map[config])

    def getPassiveAggressiveClassifier(self, config):
        map = {
            0: PassiveAggressiveClassifier(n_iter=50),
            1: BaggingClassifier(n_estimators=200, random_state=42, n_jobs=3,
                                 base_estimator=PassiveAggressiveClassifier(n_iter=50)),
            2: AdaBoostClassifier(PassiveAggressiveClassifier(n_iter=50), algorithm="SAMME", n_estimators=200)
        }
        return ("clf_" + str(config), map[config])

    def getVoting(self, config):
        map = {
            0: VotingClassifier(estimators=[self.getSVC(0), self.getNaiveBayes(1)]
                                , voting='soft', weights=[1, 2, 2]),
            1: BaggingClassifier(n_estimators=200, random_state=42, n_jobs=1,
                                 base_estimator=VotingClassifier(estimators=[self.getSVC(1), self.getNaiveBayes(1)]
                                                                 , voting='soft', weights=[2, 1])),
            2: AdaBoostClassifier(n_estimators=200,
                                  base_estimator=VotingClassifier(
                                      estimators=[self.getSVC(0), self.getNaiveBayes(1)]
                                      , voting='soft', weights=[1, 2, 2]))

        }
        return ('vote_' + str(config), map[config])

    def getCLF(self, clf_tipo, clf_config):
        map = {
            0: self.getSVC(clf_config),
            1: self.getNaiveBayes(clf_config),
            2: self.getRandomFlorest(clf_config),
            3: self.getVoting(clf_config),
            4: self.getPassiveAggressiveClassifier(clf_config),
            5: self.getRidgeClassifier(clf_config),
            6: self.getLDA(clf_config),
            7: self.getQDA(clf_config),
            8: self.getLinearSVC(clf_config)
        }
        return map[clf_tipo]

    def getProcessing(self, config):
        map = {
            0: Normalizer(),
            1: OneHotEncoder(),
            2: DenseTransformer()
        }
        return ('pre_' + str(config), map[config])

    def getPipeline(self, pr=None, fe=None, fs=None, dc=None, clf_tipo=None, clf_config=None):

        pipe = []
        if fe is not None:
            pipe.append(self.getFeaturesExtraction(fe))
        if dc is not None:
            if type(dc) is list:
                pipe.append(self.getDecompositionArray(dc))
            else:
                pipe.append(self.getDecomposition(dc))
        if pr is not None:
            pipe.append(self.getProcessing(pr))
        if fs is not None:
            if type(fs) is list:
                pipe.append(self.getFeatureSelectionArray(fs))
            else:
                pipe.append(self.getFeatureSelection(fs))
        if (clf_tipo is not None) and (clf_config is not None):
            pipe.append(self.getCLF(clf_tipo, clf_config))
        return Pipeline(pipe)

    def getKnn(self, config):
        knn = KNeighborsClassifier(n_neighbors=2)
        sfs = SFS(knn,
                  k_features=(3, 10),
                  forward=True,
                  floating=False,
                  scoring='accuracy',
                  cv=4,
                  n_jobs=-1)

        sbs = SFS(knn,
                  k_features=3,
                  forward=False,
                  floating=False,
                  scoring='accuracy',
                  cv=4,
                  n_jobs=-1)

        sffs = SFS(knn,
                   k_features=3,
                   forward=True,
                   floating=True,
                   scoring='accuracy',
                   cv=4,
                   n_jobs=-1)

        sfbs = SFS(knn,
                   k_features=(3, 10),
                   forward=False,
                   floating=True,
                   scoring='accuracy',
                   cv=4,
                   n_jobs=-1)

        map = {
            0: Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer()),
                ('reduction', sfs),
                ('clf', knn)]),
            1: Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer()),
                ('reduction', sbs),
                ('clf', knn)]),
            2: Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer()),
                ('reduction', sffs),
                ('clf', knn)]),
            3: Pipeline([
                ('vect', CountVectorizer(ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer()),
                ('reduction', sfbs),
                ('clf', knn)]),
        }
        return map[config]


class GerenciadorCluster(object):
    def kmeansBagOfWords(self, ids, titulos, corpus, n_clusters):
        vectorizer = CountVectorizer(binary=True)
        counts = vectorizer.fit_transform(corpus)

        kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)

        predicoes = kmeans.fit_predict(counts)
        dic = {}
        for i in range(len(predicoes)):
            if predicoes[i] in dic:
                dic[predicoes[i]].append(EntradaCluster(id=ids[i], titulo=titulos[i], texto=corpus[i]))
            else:
                dic[predicoes[i]] = [EntradaCluster(id=ids[i], titulo=titulos[i], texto=corpus[i])]

        output = StringIO.StringIO()
        saida = []
        clusters = []
        for i in range(np.shape(kmeans.cluster_centers_)[0]):
            data_cluster = pd.concat(
                [pd.Series(vectorizer.get_feature_names()), pd.DataFrame(kmeans.cluster_centers_[i])], axis=1)
            data_cluster.columns = ['keywords', 'weights']
            data_cluster = data_cluster.sort_values(by=['weights'], ascending=False)
            data_clust = data_cluster.head(n=10)['keywords'].tolist()
            clusters.append(data_clust)
            head = data_cluster.head(n=10)['keywords']
            saida.append(Cluster(id=i, entradas=dic[i], keywords=[h for h in head]))
        # data_CLUSTERS.to_csv('output_full.csv',index=False)
        # pd.DataFrame(clusters).to_csv('keywords_.csv')
        return saida

    def kmeansTDIDF(self, ids, titulos, corpus, n_clusters, classes=None):
        vectorizer = TfidfVectorizer(min_df=1,
                                     norm='l2',
                                     smooth_idf=True,
                                     use_idf=True
                                     )
        counts = vectorizer.fit_transform(corpus)

        kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)

        predicoes = kmeans.fit_predict(counts)
        print len(predicoes)
        dic = {}
        for i in range(len(predicoes)):
            if predicoes[i] in dic:
                if classes is None:
                    dic[predicoes[i]].append(EntradaCluster(id=ids[i], titulo=titulos[i], texto=corpus[i]))
                else:
                    dic[predicoes[i]].append(
                        EntradaCluster(id=ids[i], titulo=titulos[i], texto=corpus[i], classe=classes[i]))
            else:
                if classes is None:
                    dic[predicoes[i]] = [EntradaCluster(id=ids[i], titulo=titulos[i], texto=corpus[i])]
                else:
                    dic[predicoes[i]] = [
                        EntradaCluster(id=ids[i], titulo=titulos[i], texto=corpus[i], classe=classes[i])]

        output = StringIO.StringIO()
        saida = []
        clusters = []
        for i in range(np.shape(kmeans.cluster_centers_)[0]):
            data_cluster = pd.concat(
                [pd.Series(vectorizer.get_feature_names()), pd.DataFrame(kmeans.cluster_centers_[i])], axis=1)
            data_cluster.columns = ['keywords', 'weights']
            data_cluster = data_cluster.sort_values(by=['weights'], ascending=False)
            data_clust = data_cluster.head(n=10)['keywords'].tolist()
            clusters.append(data_clust)
            head = data_cluster.head(n=10)['keywords']
            saida.append(Cluster(id=i, entradas=dic[i], keywords=[h for h in head]))
        # data_CLUSTERS.to_csv('output_full.csv',index=False)
        # pd.DataFrame(clusters).to_csv('keywords_.csv')

        return saida

    def classificaClusters(self, targets, clusters, model):
        classificacoes = []
        for i in clusters:
            count = {}
            for t in targets:
                count[t] = 0

            entradasClusters = i.getEntradas()
            entradas = [(i.getTitulo() + " " + i.getTexto()) for i in entradasClusters]
            ids = [i.getId() for i in entradasClusters]
            predicoes = model.predict(entradas)
            for p in predicoes:
                count[p] = count[p] + 1
            maior = None
            for t in targets:
                # print "cluster:"+ str(i)
                # print str(t) +" "+str(count[t])
                # print ""
                if maior is None:
                    maior = t
                elif count[t] > count[maior]:
                    maior = t
            classificacoes.append(maior)
        return classificacoes

    def getDivergenciasCluster(self, targets, clusters):
        divergentes = {}
        for i in range(len(clusters)):
            count = {}
            for t in targets:
                count[t] = 0.0

            entradasClusters = clusters[i].getEntradas()
            predicoes = [j.getClasse() for j in entradasClusters]
            for p in predicoes:
                count[p] = count[p] + 1.0
            maior = None
            for t in targets:
                if maior is None:
                    maior = t
                elif count[t] > count[maior]:
                    maior = t
            if (count[maior] / len(predicoes)) >= 0.6:
                for p in range(len(predicoes)):
                    if predicoes[p] != maior:
                        if i in divergentes:
                            divergentes[i]['entradas'].append(entradasClusters[p])
                        else:
                            divergentes[i] = {'classe': maior, 'entradas': [entradasClusters[p]]}
        return divergentes


class Otimizador(object):
    def getFeatureSelection(self, pipeline, X, y):
        param_grid = dict(
            vote_1__n_estimators=[50, 100, 200, 300, 500],
            # vote_1__base_estimator__svc_0__C = [50, 100, 200],
            # vote_1__base_estimator__svc_0__class_weight = [{0: 10}, {1: 10}, {0: 5}],
            # vote_1__base_estimator__rf_0__max_depth = [300, 500, 1000, 1500],
            # vote_1__base_estimator__weights = [[2, 1, 1], [1, 2, 2], [1, 1, 2], [2, 2, 1]]
        )
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, n_jobs=-1)
        grid_search.fit(X, y)
        print grid_search.best_estimator_
        print grid_search.best_params_
        print grid_search.best_score_

    def getEstimators(self, pipeline, X, y):
        param_grid = dict(
                          clf_1__n_estimators=[50, 80, 100, 120],

                          # selection_0__score_func=[chi2,f_classif,mutual_info_classif],
                          # selection_0__k=[500,800,1000,1200,1500],
            # selection_5__n_features_to_select=[500, 800, 1000, 1200, 1500,2000],
                          # extraction_1__ngram_range=[(1,1),(1,2),(2,2),(1,3),(2,3),(1,4)]
        )
        # param_grid = [{'clf_0__kernel': ['linear'], 'clf_0__C': [1, 10, 50, 600]},
        #                   {'clf_0__kernel': ['poly'], 'clf_0__degree': [2, 3]},
        #                   {'clf_0__kernel': ['rbf'], 'clf_0__gamma': [0.01, 0.001], 'clf_0__C': [1, 10, 50, 600]},
        #                   ]
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10, n_jobs=-1)
        grid_search.fit(X, y)
        print grid_search.best_estimator_
        print grid_search.best_params_
        print grid_search.best_score_


class ConversorDados(object):
    def bow_extractor(self, corpus, ngram_range=(1, 1)):

        vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
        features = vectorizer.fit_transform(corpus)
        return vectorizer, features

    from sklearn.feature_extraction.text import TfidfTransformer

    def tfidf_transformer(self, bow_matrix):

        transformer = TfidfTransformer(norm='l2',
                                       smooth_idf=True,
                                       use_idf=True)
        tfidf_matrix = transformer.fit_transform(bow_matrix)
        return transformer, tfidf_matrix

    from sklearn.feature_extraction.text import TfidfVectorizer

    def tfidf_extractor(self, corpus, ngram_range=(1, 1)):
        vectorizer = TfidfVectorizer(min_df=1,
                                     norm='l2',
                                     smooth_idf=True,
                                     use_idf=True,
                                     ngram_range=ngram_range)
        features = vectorizer.fit_transform(corpus)
        return vectorizer, features

    def average_word_vectors(self, words, model, vocabulary, num_features):

        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.

        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])

        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    def averaged_word_vectorizer(self, corpus, model, num_features):
        vocabulary = set(model.wv.index2word)
        features = [self.average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
        return np.array(features)

    def tfidf_wtd_avg_word_vectors(self, words, tfidf_vector, tfidf_vocabulary, model, num_features):

        word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)]
                       if tfidf_vocabulary.get(word)
                       else 0 for word in words]
        word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}

        feature_vector = np.zeros((num_features,), dtype="float64")
        vocabulary = set(model.wv.index2word)
        wts = 0.
        for word in words:
            if word in vocabulary:
                word_vector = model[word]
                weighted_word_vector = word_tfidf_map[word] * word_vector
                wts = wts + word_tfidf_map[word]
                feature_vector = np.add(feature_vector, weighted_word_vector)
        if wts:
            feature_vector = np.divide(feature_vector, wts)

        return feature_vector

    def tfidf_weighted_averaged_word_vectorizer(self, corpus, tfidf_vectors,
                                                tfidf_vocabulary, model, num_features):

        docs_tfidfs = [(doc, doc_tfidf)
                       for doc, doc_tfidf
                       in zip(corpus, tfidf_vectors)]
        features = [self.tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
                                                    model, num_features)
                    for tokenized_sentence, tfidf in docs_tfidfs]
        return np.array(features)

    def ldaFeatures(self, n_classes, X, y):
        np.set_printoptions(precision=4)

        mean_vectors = []
        for cl in range(1, n_classes):
            mean_vectors.append(np.mean(X[y == cl], axis=0))
            print('Mean Vector class %s: %s\n' % (cl, mean_vectors[cl - 1]))

    def calibracao(self, clf, X, y):
        sample_weight = np.random.RandomState(42).rand(y.shape[0])
        X_train, X_test, y_train, y_test, sw_train, sw_test = \
            train_test_split(X, y, sample_weight, test_size=0.9, random_state=42)

        # Gaussian Naive-Bayes with no calibration
        clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
        prob_pos_clf = clf.predict_proba(X_test)[:, 1]

        # Gaussian Naive-Bayes with isotonic calibration
        clf_isotonic = CalibratedClassifierCV(clf, cv=10, method='isotonic')
        clf_isotonic.fit(X_train, y_train, sw_train)
        prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

        # Gaussian Naive-Bayes with sigmoid calibration
        clf_sigmoid = CalibratedClassifierCV(clf, cv=10, method='sigmoid')
        clf_sigmoid.fit(X_train, y_train, sw_train)
        prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

        print("Brier scores: (the smaller the better)")

        clf_score = brier_score_loss(y_test, prob_pos_clf, sw_test)
        print("No calibration: %1.3f" % clf_score)

        clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic, sw_test)
        print("With isotonic calibration: %1.3f" % clf_isotonic_score)

        clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid, sw_test)
        print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

        # #############################################################################
        # Plot the data and the predicted probabilities
        plt.figure()
        y_unique = np.unique(y)
        colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
        for this_y, color in zip(y_unique, colors):
            this_X = X_train[y_train == this_y]
            this_sw = sw_train[y_train == this_y]
            plt.scatter(this_X[:, 0], this_X[:, 1], s=this_sw * 50, c=color,
                        alpha=0.5, edgecolor='k',
                        label="Class %s" % this_y)
        plt.legend(loc="best")
        plt.title("Data")

        plt.figure()
        order = np.lexsort((prob_pos_clf,))
        plt.plot(prob_pos_clf[order], 'r', label='No calibration (%1.3f)' % clf_score)
        plt.plot(prob_pos_isotonic[order], 'g', linewidth=3,
                 label='Isotonic calibration (%1.3f)' % clf_isotonic_score)
        plt.plot(prob_pos_sigmoid[order], 'b', linewidth=3,
                 label='Sigmoid calibration (%1.3f)' % clf_sigmoid_score)
        plt.plot(np.linspace(0, y_test.size, 51)[1::2],
                 y_test[order].reshape(25, -1).mean(1),
                 'k', linewidth=3, label=r'Empirical')
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Instances sorted according to predicted probability "
                   "(uncalibrated GNB)")
        plt.ylabel("P(y=1)")
        plt.legend(loc="upper left")
        plt.title("Gaussian naive Bayes probabilities")

        plt.show()

class WeightCountVectorizerTransformer(TransformerMixin):
    c = None
    tag = None
    weight = None
    def __init__(self,tag,weight):
        self.tag = tag
        self.weight = weight
        self.c = CountVectorizer()
    def transform(self, X, y=None, **fit_params):
        dtm = self.c.transform(X)
        names = self.c.get_feature_names()
        arrays = dtm.toarray()
        for array in arrays:
            for i in range(len(names)):
                if self.tag in names[i]:
                    if(array[i]!=0):
                        array[i] = array[i] * self.weight
        return sparse.csr_matrix(arrays)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self.c.fit(X,y,**fit_params)

class WeightTFIDFVectorizerTransformer(TransformerMixin):
    c = None
    tag = None
    weight = None
    def __init__(self,tag,weight):
        self.tag = tag
        self.weight = weight
        self.c = TfidfVectorizer()
    def transform(self, X, y=None, **fit_params):
        dtm = self.c.transform(X)
        names = self.c.get_feature_names()
        arrays = dtm.toarray()
        for array in arrays:
            for i in range(len(names)):
                if self.tag in names[i]:
                    if(array[i]!=0):
                        array[i] = array[i] * self.weight
        return sparse.csr_matrix(arrays)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self.c.fit(X,y,**fit_params)
