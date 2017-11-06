#!/usr/bin/python
# coding: utf-8
from sklearn.preprocessing import Normalizer

from Classificador import PipeLineFactory, GerenciadorCluster, Otimizador
from Persistencia import GeradorEntradas
from Persistencia import GerenciadorMongoDB
from Avaliador import AvaliadorFactory, GerenciadorGraficos
from Classificador import ConversorDados
from Persistencia import AtualizadorBase
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import sparse

from ProcessadorTexto import Processador
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ProcessadorTexto import Processador
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_selection import SelectFromModel, RFE

from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
import pandas.tools.plotting as plotting
from sklearn.pipeline import Pipeline, FeatureUnion
from Classificador import Constantes

truncate = TruncatedSVD(algorithm="arpack", n_components=2)
lda = LinearDiscriminantAnalysis(n_components=2,
                                 solver='svd')

# pr = Processador()
# X = [pr.processar(u"Três pratos de trigo para três tigres tristes."),
#      pr.processar(
#          u"O doce respondeu pro doce, Que o doce mais doce, Do que o doce de batata doce, É o doce de batata doce."),
#      pr.processar(u"cachorro de brinquedo.")
#     , pr.processar(u"brinquedo de cachorro.")]
# count = CountVectorizer()
# tfidf = TfidfVectorizer()
# norm = Normalizer(copy=False)
# dtm = count.fit_transform(X)
# dtm_lsa = truncate.fit_transform(dtm)
# dtm_lsa = norm.fit_transform(dtm_lsa)
# df = DataFrame(truncate.components_, index=["componente1","componente2"] ,columns=count.get_feature_names()).head(10)
# print df
# df = DataFrame(dtm_lsa, index=["A","B","C","D"] ,columns=["componente1","componente2"]).head(10)
# print df

from sklearn import metrics
import tf_glove

pf = PipeLineFactory()
ge = GeradorEntradas()
gmdb = GerenciadorMongoDB()
cd = ConversorDados()
pr = Processador()
comentario = u'o    $%#%%questionamento é o h o m o s s e x u a l i s m o. H e t e r o s s e x u a l i d a d e não deve! jamais! se! comparar ao h o m o s s e x u a l i s m o. Então nem pergunte uma besteira destas. que h e t er  o s s e x u a l i d a d e sim é estável, diferente do h o m o s s e x u a l i s m o que varia e sempre em idades mais'
# comentario = u'mor.r.e vi/a.do'
print pr.processar(comentario, "_T")
# gmdb.converteG1paraNoticias('AllData','NovaFullBase')
# gmdb.converteHerokuParaClassificacao('ManyUsers','ManyClassificacoes')
# for i in range(10):

gg = GerenciadorGraficos()
# model = pf.getNaiveBayes(0)
gc = GerenciadorCluster()
af = AvaliadorFactory()
ot = Otimizador()
ab = AtualizadorBase()
bases = ['classificacoes', 'uniao', 'plus3']
pipelines = []
pipelines.append(pf.getPipeline(fe=Constantes.COUNT_VECTORIZER, fs=None, dc=None,
                                clf_tipo=Constantes.NAIVE_BAYES, clf_config=Constantes.NORMAL))
pipelines.append(pf.getPipeline(fe=Constantes.COUNT_VECTORIZER, fs=None, dc=None,
                                clf_tipo=Constantes.SVC, clf_config=Constantes.NORMAL))
pipelines.append(pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=None, dc=None,
                                clf_tipo=Constantes.NAIVE_BAYES, clf_config=Constantes.NORMAL))
pipelines.append(pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=None, dc=None,
                                clf_tipo=Constantes.SVC, clf_config=Constantes.NORMAL))
pipelines.append(pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER_BIGRAM, fs=None, dc=None,
                                clf_tipo=Constantes.SVC, clf_config=Constantes.NORMAL))
pipelines.append(pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=Constantes.SELECTK_CHIE2, dc=None,
                                clf_tipo=Constantes.SVC, clf_config=Constantes.NORMAL))
pipelines.append(pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=Constantes.SELECT_SVC, dc=None,
                                clf_tipo=Constantes.SVC, clf_config=Constantes.NORMAL))
pipelines.append(pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=None, dc=Constantes.LSA,
                                clf_tipo=Constantes.SVC, clf_config=Constantes.NORMAL))
pipelines.append(pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=None, dc=None,
                                clf_tipo=Constantes.SVC, clf_config=Constantes.BAGGING))
pipelines.append(pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=Constantes.SELECT_SVC, dc=None,
                                clf_tipo=Constantes.SVC, clf_config=Constantes.BAGGING))
print 'neutro/odio/ofensivo'
for b in range(len(bases)):
    for p in range(len(pipelines)):
        print 'B'+str(b+1)
        print 'P'+str(p+1)
        X,y = ge.getXy({0: 0, 1: 1, 2: 2},'join',bases[b])
        # af.crossValidationTest(pipelines[p],X,y,5)
        af.StratifyCrossValidationTest(pipelines[p], ['neutro', 'odio','ofensivo'], X, y, 5)

# X, y = ge.getXy({0: 1, 1: 0}, 'join', "plus11")
# pipe = pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=Constantes.SELECT_SVC, dc=None,
#                       clf_tipo=Constantes.SVC, clf_config=Constantes.BAGGING)
# X = np.array(X)
# y = np.array(y)
# af.StratifyCrossValidationTest(pipe, ['odio', 'outro'], X, y, 5)
# af.crossValidationTest(pipe, X, y, 5)
# for p in range(len(pipelines)):
#     print 'B4'
#     print 'P' + str(p + 1)
#     X, y = ge.getXy({0: 1, 1: 0}, 'join', 'plus11')
#     af.StratifyCrossValidationTest(pipelines[p], ['odio', 'outro'], X, y, 5)

# modelSVC.fit(X, y)
# af.divisaoTrainTest(modelSVC, ["odio", "outro"], X, y)

# gg.plot_ROC_curve(modelSVC,X,y)

# ab.atualizar({'SVC':modelSVC},"join","plus3")

# ab.analisar({'SVC':modelSVC},"join","plus3")
# ot.getEstimators(modelSVC,X,y)

# modelRIDGE = pf.getPipeline(fe=Constantes.TFIDF_VECTORIZER, fs=Constantes.SELECT_SVC,
#                        clf_tipo=Constantes.RIDGE_CLASSIFIER,clf_config=Constantes.BAGGING)
# modelRIDGE.fit(X,y)
# gg.plot_ROC_curve(model,X,y)

# af.divisaoTrainTest(modelRIDGE,["odio","outro"],X,y)
# af.crossValidationTest(modelRIDGE,X,y,5)



# X,y = ge.getXy({0:0,1:1,2:2},'join','plus2')
# print len(X)
# X = np.array(X)
# y = np.array(y)
# #
# model = pf.getPipeline(fe=1,fs=1,clf_tipo=3,clf_config=1,dc=None)
# af.divisaoTrainTest(model,["neutro","odio","ofensivo"],X,y)


# ot.getEstimators(model,X,y)

# cd.calibracao(model,X,y)



# ids,titulos,corpus = ge.getDiferencaClassificacaoBase("FullBase",'uniao')
#
# print len(corpus)
# saida = gc.kmeansBagOfWords(ids,corpus,10)
# saida = gc.kmeansTDIDF(ids,titulos,corpus,500)
#
# for i in saida:
#     gmdb.salvarModelo("clusterTDIDF",i)
#
# clusters = gmdb.getClusters("clusterTDIDF")
# classificacoes = gc.classificaClusters([0,1],clusters,model)
# for i in range(len(classificacoes)):
#     print "cluster numero "+str(i)
#     print "classe majoritaria:"+ str(classificacoes[i])


# ids,titulos,corpus,classes = ge.getDiferencaBaseClassificacao("join",'plus2')
#
# print len(corpus)
# # saida = gc.kmeansBagOfWords(ids,corpus,10)
# saida = gc.kmeansTDIDF(ids,titulos,corpus,100,classes)
#
# for i in saida:
#     gmdb.salvarModelo("classificacaoClusterTDIDF",i)

# clusters = gmdb.getClusters("classificacaoClusterTDIDF")
# entradas =gc.getDivergenciasCluster(["odio","neutro","ofensivo"],clusters)
#
# for i in entradas:
#     print "cluster numero "+str(i)
#     print "classe majoritaria:"+ entradas[i]['classe']
#     print ""
#     list = entradas[i]['entradas']
#     for l in list:
#         noticia = gmdb.getNoticiasPorIdComentario("FullBase",l.getId())
#         gmdb.updateClasses('uniao2',l.getId(),entradas[i]['classe'])
#         print l.getId()
#         print noticia.getTitulo()
#         print noticia.getComentarios()[0].getTexto()
#         print l .getClasse()
#         print ""
#
# model = pf.getSVC(1)
# X,y = ge.getXy({0:1,1:0,2:1},'FullBase','uniao2')
# af.divisaoTrainTest(model,["odio","outro"],X,y)
# af.crossValidationTest(model,X,y,10)
# model = pf.getSVC(1)
# X,y = ge.getXy({0:1,1:0,2:1},'FullBase','uniao2')
# af.divisaoTrainTest(model,["odio","outro"],X,y)
# af.crossValidationTest(model,X,y,10)
# model.fit(X,y)
# classificacoes = gc.classificaClusters([0,1],clusters,model)
# for c in range(len(classificacoes)):
#     print "cluster numero:"+str(c) +" foi classificado como "+ str(classificacoes[c])

# tags = ['svm','bagging','ada']
# for i in range(3):
#     model = pf.getSVC(i)
#     print tags[i]
#     print 'uniao'
#     X,y = ge.getXy({0:1,1:0,2:1},'FullBase','uniao')
#     # af.divisaoTrainTest(model,["neutro","odio","ofensivo"],X,y)
#     af.crossValidationTest(model,X,y,10)
#     print 'classificacoes'
#     X,y = ge.getXy({0:1,1:0,2:1},'FullBase','classificacoes')
#     # af.divisaoTrainTest(model,["odio","outro"],X,y)
#     af.crossValidationTest(model,X,y,10)
#     print 'plus'
#     X,y = ge.getXy({0:1,1:0,2:1},'join','plus')
#     # af.divisaoTrainTest(model,["odio","outro"],X,y)
#     af.crossValidationTest(model,X,y,10)


# entradas2 = ge.getEntradas({0:0,1:1,2:2},'FullBase','ManyClassificacoes')
# print len(entradas2)
# X2 = [i.getInput() for i in entradas2]
# y2 = [i.getClasse() for i in entradas2]

# af.divisaoTrainTest(model,["neutro","odio","ofensivo"],X,y)
# af.divisaoTrainTest(model,["neutro","odio","ofensivo"],X2,y2)
# af.comparacaoBases(model,["neutro","odio","ofensivo"],X,y,X2,y2)
# af.comparacaoBases(model,["neutro","odio","ofensivo"],X2,y2,X,y)
# af.crossValidationTest(model,X,y,10)

# Sequential Forward Selection
