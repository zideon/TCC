#!/usr/bin/python
# coding: utf-8
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
import re
import unicodedata


class RemovedorRadicais(object):

    def remover( self,texto, linguagem = 'portuguese'):
        stemmer_snowball = SnowballStemmer(linguagem)
        words = []
        tokens = word_tokenize(texto)
        for word in tokens:
            w = stemmer_snowball.stem(word)
            if (word):
                words.append(word)
        saida = ""
        for word in words:
            saida = saida + " " + word
        saida = saida.strip()
        return saida

class RemovedorStopWord(object):

    def remover(self,texto,linguagem = 'portuguese'):
        stop_words = nltk.corpus.stopwords.words(linguagem)
        words = []
        tokens = word_tokenize(texto)
        for word in tokens:
            if (word):
                if (word not in stop_words):
                    words.append(word)
        saida = ""
        for word in words:
            saida = saida + " " + word
        saida = saida.strip()
        return saida

class RemovedorCaracters(object):

    def remover(self,texto):
        palavras = word_tokenize(texto)
        # Unicode normalize transforma um caracter em seu equivalente em latin.
        saida = ""
        for palavra in palavras:
            nfkd = unicodedata.normalize('NFKD', palavra)
            palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
            palavraSemAcento = re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento)
            saida = saida +" "+ palavraSemAcento
        saida = saida.strip()
        return saida

class Processador(object):

    def processar(self,texto):
        texto = texto.strip()
        texto = texto.lower()
        rc =RemovedorCaracters()
        rs = RemovedorStopWord()
        rr =RemovedorRadicais()
        texto = rc.remover(texto)
        texto = rs.remover(texto)
        texto = rr.remover(texto)

        return texto