#!/usr/bin/python
# coding: utf-8
import StringIO

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from Modelo import EntradaCluster
from Modelo import Cluster



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

class GerenciadorCluster(object):
    def kmeansBagOfWords(self,ids,titulos,corpus,n_clusters):
        vectorizer = CountVectorizer( binary=True)
        counts = vectorizer.fit_transform(corpus)

        kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)

        predicoes = kmeans.fit_predict(counts)
        dic = {}
        for i in range(len(predicoes)):
            if predicoes[i] in dic:
                dic[predicoes[i]].append(EntradaCluster(id=ids[i],titulo=titulos[i], texto=corpus[i]))
            else:
                dic[predicoes[i]] = [EntradaCluster(id=ids[i], titulo=titulos[i],texto=corpus[i])]


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
            head =  data_cluster.head(n=10)['keywords']
            saida.append(Cluster(id=i, entradas=dic[i],keywords=[h for h in head]))
        # data_CLUSTERS.to_csv('output_full.csv',index=False)
        # pd.DataFrame(clusters).to_csv('keywords_.csv')
        return saida

    def kmeansTDIDF(self,ids,titulos,corpus,n_clusters):
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
            if predicoes[i] in  dic:
                dic[predicoes[i]].append(EntradaCluster(id=ids[i],titulo=titulos[i],texto=corpus[i]))
            else:
                dic[predicoes[i]] = [EntradaCluster(id=ids[i],titulo=titulos[i], texto=corpus[i])]

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
    def classificaClusters(self,targets,clusters,model):
        classificacoes = []
        for i in clusters:
            count = {}
            for t in targets:
                count[t] = 0

            entradasClusters = i.getEntradasPorClassificacao()
            entradas = [ (i.getTitulo() +" "+ i.getTexto()) for i in entradasClusters]
            ids = [ i.getId() for i in entradasClusters]
            predicoes = model.predict(entradas)
            for p in predicoes:
                count[p] = count[p]+1
            maior = None
            for t in targets:
                # print "cluster:"+ str(i)
                # print str(t) +" "+str(count[t])
                # print ""
                if maior is None:
                    maior = t
                elif count[t]>count[maior]:
                    maior = t
            classificacoes.append(maior)
        return classificacoes

