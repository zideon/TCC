#!/usr/bin/python
# coding: utf-8

from pymongo import MongoClient
from Modelo import Noticia
from Modelo import Classificacao
from Modelo import Comentario
from Modelo import ClassificacaoNoticia
from Modelo import EntradaClassificador
from Modelo import Cluster
from ProcessadorTexto import Processador
from sklearn import preprocessing
import numpy as np

class GerenciadorMongoDB(object):


    cliente = None
    banco = None

    def __init__(self,host='localhost',porta=27017,nomeBanco='fabioTCC'):
        self.cliente = MongoClient(host, porta)
        self.banco = self.cliente[nomeBanco]
    def getColecao(self,nome):
        return self.banco[nome]
    def createColecao(self,nome):
        return self.banco.create_collection(name=nome)
    def getNoticias(self,nomeBanco):
        colecao = self.banco[nomeBanco]
        noticias = []
        for i in colecao.find({}):
            noticias.append(Noticia(doc=i))
        return noticias


    def getClusters(self,nomeBanco):
        colecao = self.banco[nomeBanco]
        clusters = []
        for i in colecao.find({}):
            clusters.append(Cluster(doc=i))
        return clusters

    def salvarModelo(self,nomeBanco,modelo):
        colecao = self.banco[nomeBanco]
        colecao.insert_one(modelo.documento)

    def getNoticiasPorIdComentario(self,nomeBanco,idComentario):
        colecao = self.banco[nomeBanco]
        for i in colecao.find(
                {'comentarios': {'$elemMatch': {'idComentario': int(idComentario)}}}
                ,{'titulo':1,'comentarios': {'$elemMatch': {'idComentario': int(idComentario)}}}):
            return Noticia(doc= i)
    def getClassificacoes(self,nomeBanco):
        colecao = self.banco[nomeBanco]
        classificacoes = []
        for i in colecao.find({}):
            classificacoes.append(Classificacao(doc=i))
        return classificacoes
    def getCount(self,nomeBanco,comentario,classe):
        colecao = self.banco[nomeBanco]
        classificacoes = []
        return colecao.count({'classe':classe,'idComentario':int(comentario)})
    def getClasses(self,nomeBanco):
        colecao = self.banco[nomeBanco]
        return colecao.distinct('classe')
    def converteHerokuParaClassificacao(self,nomeHeroku,nomeBase):
        baseHeroku = self.getColecao(nomeHeroku)
        self.createColecao(nomeBase)
        baseMongoDB = self.getColecao(nomeBase)
        for classes in baseHeroku.find({}, {"values": 1}):
            for c in classes["values"]:
                classificacao = Classificacao(id=str(c[0]) + '#' + str(c[1]),noticia=c[3],comentario=c[1],email=c[0],classe=c[2])
                baseMongoDB.insert_one(classificacao.documento)

    def converteG1paraNoticias(self,nomeG1,nomeBase):
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
            noticia = Noticia(titulo=dit,comentarios=comentarios)
            baseMongoDB.insert_one(noticia.documento)

class GeradorEntradas(object):

    def getMaior(self,conjunto):
        maior = None
        for i in conjunto:
            if conjunto[i] > conjunto[maior]:
                maior = i
        return maior

    def getDiferencaClassificacaoBase(self,nomeBaseNoticias, nomeBaseClassificoes):
        gmdb = GerenciadorMongoDB()
        classificacoes = gmdb.getClassificacoes( nomeBaseClassificoes)
        usados = [int(i.getIdComentario()) for i in classificacoes]

        ids,titulos,textos = self.getTodasEntradas( nomeBaseNoticias)
        idsNovo =[]
        titulosNovo = []
        textosNovo = []
        for i in range(len(ids)):
            if ids[i] not in usados:
                idsNovo.append(ids[i])
                titulosNovo.append(titulos[i])
                textosNovo.append(textos[i])
        return idsNovo,titulosNovo,textosNovo

    def getClassificacoesNoticias(self,nomeBaseNoticias, nomeBaseClassificoes):
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
            noticia = gmdb.getNoticiasPorIdComentario(nomeBaseNoticias,classificacao.getIdComentario())
            if noticia is not None:
                comentarios = noticia.getComentarios()
                if comentarios[0].getId() not in conjunto:
                    titulo =noticia.getTitulo()
                    texto = comentarios[0].getTexto()
                    classe = None
                    dict = {}
                    sum = 0
                    for c in classes:
                        dict[c]= gmdb.getCount(nomeBaseClassificoes,comentarios[0].getId(),c)
                        sum = sum + dict[c]
                    if sum >1:
                        classe = self.getMaior(dict)
                    else:
                        classe = classificacao.getClasse()
                    classeTrans = label_encoder.transform([classe])
                    conjunto[comentarios[0].getId()] = ClassificacaoNoticia(titulo,texto,classeTrans[0])
        entradas = [conjunto[i] for i in conjunto]
        return entradas

    def getXy(self,targets,nomeBaseNoticias, nomeBaseClassificoes):
        entradas = self.getEntradasPorClassificacao(targets, nomeBaseNoticias, nomeBaseClassificoes)
        X = [i.getInput() for i in entradas]
        y = [i.getClasse() for i in entradas]
        print len(entradas)
        return X,y

    def getEntradasPorClassificacao(self, targets, nomeBaseNoticias, nomeBaseClassificoes):

        classificacoesNoticias = self.getClassificacoesNoticias(nomeBaseNoticias, nomeBaseClassificoes)
        entradas = []
        p = Processador()
        for cn in classificacoesNoticias:
            tituloProcessado =p.processar(cn.getTitulo())
            textoProcessado =p.processar(cn.getTexto())
            if(tituloProcessado and textoProcessado):
                entradas.append(EntradaClassificador(tituloProcessado + " " + textoProcessado, targets[cn.getClasse()]))
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(entradas)))
        entradas =  np.asarray(entradas)
        entradas = entradas[shuffle_indices]
        return entradas

    def getTodasEntradas(self,nomeBanco):
        gmdb = GerenciadorMongoDB()
        noticias = gmdb.getNoticias(nomeBanco)
        print len(noticias)
        corpus = []
        Ids = []
        titulos = []
        p = Processador()
        for i in noticias:
            for j in i.getComentarios():
                try:
                    tituloProcessado = p.processar(i.getTitulo())
                    textoProcessado = p.processar(j.getTexto())
                    if (textoProcessado is not None) and (tituloProcessado is not None):
                        Ids.append(j.getId())
                        titulos.append(tituloProcessado)
                        corpus.append(textoProcessado)
                except Exception as e:
                    print "erro ao processar"
        return Ids,titulos,corpus
