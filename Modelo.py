#!/usr/bin/python
# coding: utf-8


class Cluster(object):
    documento = None

    def __init__(self,id=None,entradas=None,keywords=None,doc=None):
        if doc is not None:
            self.documento = doc
        elif (id is not None) and (entradas is not None) and (keywords is not None):
            entradasJson = []
            for doc in entradas:
                entradasJson.append(doc.documento)
            self.documento = {'id':id,'entradas':entradasJson,"keywords":keywords}
    def getId(self):
        return self.documento['id']
    def getEntradas(self):
        entradasJson = self.documento['entradas']
        entradas = []
        for c in entradasJson:
            entradas.append(EntradaCluster(doc=c))
        return entradas
    def getKeyWords(self):
        return self.documento['keywords']

class EntradaCluster(object):

    documento = None

    def __init__(self,id=None,titulo=None,texto=None,classe=None,doc=None):
        if doc is not None:
            self.documento = doc
        elif (id is not None) and (titulo is not None) and (texto is not None) :
            if classe is None:
                self.documento = {'id':id,'titulo':titulo,'texto':texto}
            else:
                self.documento = {'id': id, 'titulo': titulo, 'texto': texto,'classe':classe}
    def getId(self):
        return self.documento['id']
    def getTitulo(self):
        return self.documento['titulo']
    def getTexto(self):
        return self.documento['texto']
    def getClasse(self):
        return self.documento['classe']

class Comentario(object):

    documento = None

    def __init__(self,id=None,texto=None,doc=None):
        if doc is not None:
            self.documento = doc
        elif ((id is not None) and (texto is not None)):
            self.documento = {'idComentario':id,'texto':texto}
    def getId(self):
        return self.documento['idComentario']
    def getTexto(self):
        return self.documento['texto']

class Noticia(object):

    documento = None

    def __init__(self,id=None,titulo=None,comentarios=None,doc=None):
        if doc is not None:
            self.documento = doc
        elif ((id is None) and (titulo is not None)  and (comentarios is not None)):
            comentariosJson = []
            for doc in comentarios:
                comentariosJson.append(doc.documento)
            self.documento = {'titulo': titulo, 'comentarios': comentariosJson}
        elif ((id is not None) and (titulo is not None) and (comentarios is not None)):
            comentariosJson = []
            for doc in comentarios:
                comentariosJson.append(doc.documento)
            self.documento = {'_id': id, 'titulo': titulo, 'comentarios': comentariosJson}
    def getId(self):
        return self.documento['_id']
    def getTitulo(self):
        return self.documento['titulo']
    def getComentarios(self):
        comentariosJson = self.documento['comentarios']
        comentarios = []
        for c in comentariosJson:
            comentarios.append(Comentario(doc=c))
        return comentarios


class Classificacao(object):

    documento = None

    def __init__(self,id=None,noticia=None,comentario=None,email=None,classe=None,doc=None):
        if doc is not None:
            self.documento = doc
        elif ((id is not None)  and (comentario is not None) and (classe is not None)):
            self.documento = {'_id':id,'idNoticia':noticia,'idComentario':comentario,'email':email,'classe':classe}
    def getId(self):
        return self.documento['_id']
    def getIdNoticia(self):
        return self.documento['idNoticia']
    def getIdComentario(self):
        return self.documento['idComentario']
    def getEmail(self):
        return self.documento['email']
    def getClasse(self):
        return self.documento['classe']

class ClassificacaoNoticia(object):
    documento = None

    def __init__(self, titulo=None,texto=None, classe=None,doc=None):
        if doc is not None:
            self.documento = doc
        elif ((titulo is not None) and (texto is not None)  and (classe is not None)):
            self.documento = {'titulo': titulo, 'texto': texto, 'classe': classe}

    def getTitulo(self):
        return self.documento['titulo']
    def getTexto(self):
        return self.documento['texto']
    def getClasse(self):
        return self.documento['classe']

class EntradaClassificador(object):

    documento = None

    def __init__(self, input=None, classe=None,doc=None):
        if doc is not None:
            self.documento = doc
        elif ((input is not None)  and (classe is not None)):
            self.documento = {'input': input, 'classe': classe}


    def getInput(self):
        return self.documento['input']

    def getClasse(self):
        return self.documento['classe']

    def getArray(self):
        return [self.getInput(),self.getClasse()]