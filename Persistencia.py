#!/usr/bin/python
# coding: utf-8
from __future__ import division

from pymongo import MongoClient
from Modelo import Noticia
from Modelo import Classificacao
from Modelo import Comentario
from Modelo import ClassificacaoNoticia
from Modelo import EntradaClassificador
from Modelo import Cluster
from ProcessadorTexto import Processador
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
import numpy as np


class GerenciadorMongoDB(object):
    cliente = None
    banco = None

    def __init__(self, host='localhost', porta=27017, nomeBanco='fabioTCC'):
        self.cliente = MongoClient(host, porta)
        self.banco = self.cliente[nomeBanco]

    def getColecao(self, nome):
        return self.banco[nome]

    def createColecao(self, nome):
        return self.banco.create_collection(name=nome)

    def getNoticias(self, nomeBanco):
        colecao = self.banco[nomeBanco]
        noticias = []
        for i in colecao.find({}):
            noticias.append(Noticia(doc=i))
        return noticias

    def getClusters(self, nomeBanco):
        colecao = self.banco[nomeBanco]
        clusters = []
        for i in colecao.find({}):
            clusters.append(Cluster(doc=i))
        return clusters

    def salvarModelo(self, nomeBanco, modelo):
        colecao = self.banco[nomeBanco]
        colecao.insert_one(modelo.documento)

    def updateClasses(self, nomeBanco, idComentario, classe):
        colecao = self.banco[nomeBanco]
        colecao.update({'idComentario': str(idComentario)}, {'$set': {'classe': str(classe)}})

    def getNoticiasPorIdComentario(self, nomeBanco, idComentario):
        colecao = self.banco[nomeBanco]
        for i in colecao.find(
                {'comentarios': {'$elemMatch': {'idComentario': int(idComentario)}}}
                , {'titulo': 1, 'comentarios': {'$elemMatch': {'idComentario': int(idComentario)}}}):
            return Noticia(doc=i)

    def getClustersPorIdComentario(self, nomeBanco, idComentario):
        colecao = self.banco[nomeBanco]
        clusters = []
        for i in colecao.find({'entradas': {'$elemMatch': {'id': int(idComentario)}}}):
            clusters.append(Cluster(doc=i))
        return clusters

    def getClustersPorTexto(self, nomeBanco, texto):
        colecao = self.banco[nomeBanco]
        clusters = []
        for w in word_tokenize(texto):
            for i in colecao.find({'keywords': {'$elemMatch': {'$eq': w}}}):
                clusters.append(Cluster(doc=i))
        return clusters

    def getClassificacoes(self, nomeBanco):
        colecao = self.banco[nomeBanco]
        classificacoes = []
        for i in colecao.find({}):
            classificacoes.append(Classificacao(doc=i))
        return classificacoes

    def getCount(self, nomeBanco, comentario, classe):
        colecao = self.banco[nomeBanco]
        classificacoes = []
        return colecao.count({'classe': classe, 'idComentario': int(comentario)})

    def getClasses(self, nomeBanco):
        colecao = self.banco[nomeBanco]
        return colecao.distinct('classe')

    def converteHerokuParaClassificacao(self, nomeHeroku, nomeBase):
        baseHeroku = self.getColecao(nomeHeroku)
        self.createColecao(nomeBase)
        baseMongoDB = self.getColecao(nomeBase)
        for classes in baseHeroku.find({}, {"values": 1}):
            for c in classes["values"]:
                classificacao = Classificacao(id=str(c[0]) + '#' + str(c[1]), noticia=c[3], comentario=c[1], email=c[0],
                                              classe=c[2])
                baseMongoDB.insert_one(classificacao.documento)

    def converteG1paraNoticias(self, nomeG1, nomeBase):
        baseG1 = self.getColecao(nomeG1)
        self.createColecao(nomeBase)
        baseMongoDB = self.getColecao(nomeBase)
        total = 0
        dicionario = {}
        for post in baseG1.find({}, {"itens": 1}):
            for x in post["itens"]:
                total = total + len(x)
                topico = x["topico"]["titulo"]
                textoOriginal = x["texto"]
                idComentario = x["idComentario"]
                if topico in dicionario:
                    dicionario[topico].append({"idComentario": idComentario, "texto": textoOriginal})
                else:
                    dicionario[topico] = [{"idComentario": idComentario, "texto": textoOriginal}]
                for y in x["replies"]:
                    total = total + len(y)
                    topico = y["topico"]["titulo"]
                    textoOriginal = y["texto"]
                    idComentario = y["idComentario"]
                    if topico in dicionario:
                        dicionario[topico].append({"idComentario": idComentario, "texto": textoOriginal})
                    else:
                        dicionario[topico] = [{"idComentario": idComentario, "texto": textoOriginal}]

        print("total:" + str(total))
        for dit in dicionario:
            comentarios = []
            for json in dicionario[dit]:
                comentarios.append(Comentario(doc=json))
            noticia = Noticia(titulo=dit, comentarios=comentarios)
            baseMongoDB.insert_one(noticia.documento)


class GeradorEntradas(object):
    def getMaior(self, conjunto):
        maior = None
        for i in conjunto:
            if conjunto[i] > conjunto[maior]:
                maior = i
        return maior

    def getDiferencaClassificacaoBase(self, nomeBaseNoticias, nomeBaseClassificoes):
        gmdb = GerenciadorMongoDB()
        classificacoes = gmdb.getClassificacoes(nomeBaseClassificoes)
        usados = [int(i.getIdComentario()) for i in classificacoes]
        countFora = 0
        ids, titulos, textos = self.getTodasEntradas(nomeBaseNoticias)
        print 'total de entradas da base ' + str(len(ids))
        idsNovo = []
        titulosNovo = []
        textosNovo = []
        for i in range(len(ids)):
            if ids[i] not in usados:
                idsNovo.append(ids[i])
                titulosNovo.append(titulos[i])
                textosNovo.append(textos[i])
            else:
                countFora = countFora + 1
        print 'quantidade de elementos removidos =' + str(countFora)
        return idsNovo, titulosNovo, textosNovo

    def getDiferencaBaseClassificacao(self, nomeBaseNoticias, nomeBaseClassificoes):
        gmdb = GerenciadorMongoDB()
        classificacoes = gmdb.getClassificacoes(nomeBaseClassificoes)
        usados = [int(i.getIdComentario()) for i in classificacoes]

        ids, titulos, textos = self.getTodasEntradas(nomeBaseNoticias)
        idsNovo = []
        titulosNovo = []
        textosNovo = []
        classes = []
        for i in range(len(ids)):
            if ids[i] in usados:
                indice = usados.index(ids[i])
                idsNovo.append(ids[i])
                classes.append(classificacoes[indice].getClasse())
                titulosNovo.append(titulos[i])
                textosNovo.append(textos[i])
        return idsNovo, titulosNovo, textosNovo, classes

    def getClassificacoesNoticias(self, nomeBaseNoticias, nomeBaseClassificoes):
        gmdb = GerenciadorMongoDB()
        classificacoes = gmdb.getClassificacoes(nomeBaseClassificoes)
        # noticias = gmdb.getNoticias(nomeBaseNoticias)
        classes = gmdb.getClasses(nomeBaseClassificoes)

        label_encoder = preprocessing.LabelEncoder()
        input_classes = classes
        label_encoder.fit(input_classes)
        for i, item in enumerate(label_encoder.classes_):
            print item, '-->', i
        conjunto = {}
        for classificacao in classificacoes:
            noticia = gmdb.getNoticiasPorIdComentario(nomeBaseNoticias, classificacao.getIdComentario())
            if noticia is not None:
                comentarios = noticia.getComentarios()
                if comentarios[0].getId() not in conjunto:
                    titulo = noticia.getTitulo()
                    texto = comentarios[0].getTexto()
                    words = word_tokenize(texto)
                    if len(words)>2:
                        classe = None
                        dict = {}
                        sum = 0
                        for c in classes:
                            dict[c] = gmdb.getCount(nomeBaseClassificoes, comentarios[0].getId(), c)
                            sum = sum + dict[c]
                        if sum > 1:
                            classe = self.getMaior(dict)
                        else:
                            classe = classificacao.getClasse()
                        classeTrans = label_encoder.transform([classe])
                        conjunto[comentarios[0].getId()] = ClassificacaoNoticia(titulo, texto, classeTrans[0])
        entradas = [conjunto[i] for i in conjunto]
        return entradas

    def getXy(self, targets, nomeBaseNoticias, nomeBaseClassificoes):
        entradas = self.getEntradasPorClassificacao(targets, nomeBaseNoticias, nomeBaseClassificoes)
        print len(entradas)
        X, y = self.RemoveRepetidosListaEntradaClassificador(entradas)
        return X, y

    def RemoveRepetidosListaEntradaClassificador(self, lista):
        X = []
        y = []
        for i in range(len(lista)):
            if not X.count(lista[i].getInput()):
                X.append(lista[i].getInput())
                y.append(lista[i].getClasse())
        return X, y

    def getEntradasPorClassificacao(self, targets, nomeBaseNoticias, nomeBaseClassificoes):

        classificacoesNoticias = self.getClassificacoesNoticias(nomeBaseNoticias, nomeBaseClassificoes)
        entradas = []
        p = Processador()
        for cn in classificacoesNoticias:
            tituloProcessado = p.processar(cn.getTitulo(),"_T")
            textoProcessado = p.processar(cn.getTexto(),"_C")

            if (tituloProcessado and textoProcessado):
                entradas.append(EntradaClassificador(tituloProcessado + " " + textoProcessado, targets[cn.getClasse()]))
                # entradas.append(EntradaClassificador( textoProcessado, targets[cn.getClasse()]))
                # entradas.append(EntradaClassificador( tituloProcessado, 1))
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(entradas)))
        entradas = np.asarray(entradas)
        entradas = entradas[shuffle_indices]
        return entradas

    def getTodasEntradas(self, nomeBanco):
        gmdb = GerenciadorMongoDB()
        noticias = gmdb.getNoticias(nomeBanco)
        print len(noticias)
        corpus = []
        Ids = []
        titulos = []
        inputs = []
        p = Processador()
        for i in noticias:
            for j in i.getComentarios():
                try:
                    tituloProcessado = p.processar(i.getTitulo(),"_T")
                    textoProcessado = p.processar(j.getTexto(),"_C")
                    if (textoProcessado is not None) and (tituloProcessado is not None) \
                            and textoProcessado and tituloProcessado and not inputs.count(
                                        tituloProcessado + " " + textoProcessado):
                        Ids.append(j.getId())
                        titulos.append(tituloProcessado)
                        corpus.append(textoProcessado)
                        inputs.append(tituloProcessado + " " + textoProcessado)
                except Exception as e:
                    print "erro ao processar"
        return Ids, titulos, corpus


class AtualizadorBase(object):
    def atualizar(self, modelDict, baseNoticias, baseClassificacoes):
        ge = GeradorEntradas()
        gmdb = GerenciadorMongoDB()
        ids, titulos, corpus = ge.getDiferencaClassificacaoBase(baseNoticias, baseClassificacoes)
        print len(ids)
        entradas = [[ids[i], (titulos[i] + " " + corpus[i])] for i in range(len(ids))]
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(entradas)))
        entradas = np.asarray(entradas)
        entradas = entradas[shuffle_indices]

        predicts = {}
        for nome in modelDict:
            print modelDict[nome].classes_
            predicts[nome] = modelDict[nome].predict([i[1] for i in entradas])

        for i in range(len(entradas)):
            id = entradas[i][0]
            noticia = gmdb.getNoticiasPorIdComentario(baseNoticias, id)
            print noticia.getTitulo()
            print noticia.getComentarios()[0].getTexto()
            for p in predicts:
                print p + " " + str(predicts[p][i])
            input = raw_input("Digite a:odio b:ofensivo c:neutro d:ignora")
            if input == "a":
                c = Classificacao(id="system#" + str(id), comentario=id, noticia=noticia.getId(),
                                  email="system", classe='odio')
                gmdb.salvarModelo(baseClassificacoes, c)
            elif input == "b":
                c = Classificacao(id="system#" + str(id), comentario=id, noticia=noticia.getId(),
                                  email="system", classe='ofensivo')
                gmdb.salvarModelo(baseClassificacoes, c)
            elif input == "c":
                c = Classificacao(id="system#" + str(id), comentario=id, noticia=noticia.getId(),
                                  email="system", classe='neutro')
                gmdb.salvarModelo(baseClassificacoes, c)

    class AtualizadorBase(object):
        def atualizar(self, modelDict, baseNoticias, baseClassificacoes):
            ge = GeradorEntradas()
            gmdb = GerenciadorMongoDB()
            ids, titulos, corpus = ge.getDiferencaClassificacaoBase(baseNoticias, baseClassificacoes)
            print len(ids)
            entradas = [[ids[i], (titulos[i] + " " + corpus[i])] for i in range(len(ids))]
            np.random.seed(10)
            shuffle_indices = np.random.permutation(np.arange(len(entradas)))
            entradas = np.asarray(entradas)
            entradas = entradas[shuffle_indices]

            predicts = {}
            for nome in modelDict:
                print modelDict[nome].classes_
                predicts[nome] = modelDict[nome].predict([i[1] for i in entradas])

            for i in range(len(entradas)):
                id = entradas[i][0]
                noticia = gmdb.getNoticiasPorIdComentario(baseNoticias, id)
                print noticia.getTitulo()
                print noticia.getComentarios()[0].getTexto()
                for p in predicts:
                    print p + " " + str(predicts[p][i])
                input = raw_input("Digite a:odio b:ofensivo c:neutro d:ignora")
                if input == "a":
                    c = Classificacao(id="system#" + str(id), comentario=id, noticia=noticia.getId(),
                                      email="system", classe='odio')
                    gmdb.salvarModelo(baseClassificacoes, c)
                elif input == "b":
                    c = Classificacao(id="system#" + str(id), comentario=id, noticia=noticia.getId(),
                                      email="system", classe='ofensivo')
                    gmdb.salvarModelo(baseClassificacoes, c)
                elif input == "c":
                    c = Classificacao(id="system#" + str(id), comentario=id, noticia=noticia.getId(),
                                      email="system", classe='neutro')
                    gmdb.salvarModelo(baseClassificacoes, c)

    def analisar(self, modelDict, baseNoticias, baseClassificacoes):
        ge = GeradorEntradas()
        gmdb = GerenciadorMongoDB()
        ids, titulos, corpus = ge.getDiferencaClassificacaoBase(baseNoticias, baseClassificacoes)
        print len(ids)
        entradas = [[ids[i], (titulos[i] + " " + corpus[i])] for i in range(len(ids))]
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(entradas)))
        entradas = np.asarray(entradas)
        entradas = entradas[shuffle_indices]

        predicts = {}
        for nome in modelDict:
            print modelDict[nome].classes_
            predicts[nome] = modelDict[nome].predict_proba([i[1] for i in entradas])
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        for i in range(len(entradas)):

            for p in predicts:
                if(predicts[p][i][0]>=0.70):

                    id = entradas[i][0]
                    noticia = gmdb.getNoticiasPorIdComentario(baseNoticias, id)
                    print noticia.getTitulo()
                    print noticia.getComentarios()[0].getTexto()
                    print p + " " + str(predicts[p][i])

                    # input = raw_input("Digite a:TP b:FP c:TN d:FN e:exit")
                    # if input == "a":
                    #     TP = TP + 1
                    #     print "TP"
                    # elif input == "b":
                    #     FP = FP + 1
                    #     print "FP"
                    # elif input == "c":
                    #     TN = TN + 1
                    #     print "TN"
                    # elif input == "d":
                    #     FN = FN + 1
                    #     print "FN"
                    # elif input == "e":
                    #     print "TP:" + str(TP)
                    #     print "FP:" + str(FP)
                    #     print "TN:" + str(TN)
                    #     print "FN:" + str(FN)
                    #     try:
                    #         print "accuracy:" + str((TP + TN) / (TP + TN + FP + FN))
                    #     except ZeroDivisionError:
                    #         print "accuracy erro"
                    #     try:
                    #         print "precision:" + str(TP / (TP + FP))
                    #     except ZeroDivisionError:
                    #         print "precision erro"
                    #     try:
                    #         print "recall:" + str(TP / (TP + FN))
                    #     except ZeroDivisionError:
                    #         print "recall erro"
                    #
                    #     exit()



