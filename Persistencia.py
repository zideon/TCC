#!/usr/bin/python
# coding: utf-8

from pymongo import MongoClient
from Modelo import Noticia
from Modelo import Classificacao
from Modelo import Comentario
from Modelo import ClassificacaoNoticia
from Modelo import Entrada
from ProcessadorTexto import Processador
from sklearn import preprocessing

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
    def getNoticiasPorIdComentario(self,nomeBanco,idComentario):
        colecao = self.banco[nomeBanco]
        for i in colecao.find({ 'comentarios': { '$elemMatch': { 'idComentario': idComentario } } },{'titulo':1,'comentarios': { '$elemMatch': { 'idComentario': idComentario } }}):
            return Noticia(doc= i)
    def getClassificacoes(self,nomeBanco):
        colecao = self.banco[nomeBanco]
        classificacoes = []
        for i in colecao.find({}):
            classificacoes.append(Classificacao(doc=i))
        return classificacoes
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
            noticia = Noticia(dit,comentarios)
            baseMongoDB.insert_one(noticia.documento)

class GeradorEntradas(object):

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

        entradas = []
        for classificacao in classificacoes:
            noticia = gmdb.getNoticiasPorIdComentario(nomeBaseNoticias,classificacao.getIdComentario())
            if noticia is not None:
                comentarios = noticia.getComentarios()
                titulo =noticia.getTitulo()
                texto = comentarios[0].getTexto()
                classe = classificacao.getClasse()
                classeTrans = label_encoder.transform([classe])
                entradas.append(ClassificacaoNoticia(titulo,texto,classeTrans[0]))
        return entradas

    def getEntradas(self,targets,nomeBaseNoticias, nomeBaseClassificoes):

        classificacoesNoticias = self.getClassificacoesNoticias(nomeBaseNoticias, nomeBaseClassificoes)
        entradas = []
        p = Processador()
        for cn in classificacoesNoticias:
            tituloProcessado =p.processar(cn.getTitulo())
            textoProcessado =p.processar(cn.getTexto())
            if(tituloProcessado and textoProcessado):
                entradas.append(Entrada(tituloProcessado+" "+textoProcessado,targets[cn.getClasse()]))
        return entradas


