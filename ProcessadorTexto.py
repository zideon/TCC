#!/usr/bin/python
# coding: utf-8
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize,wordpunct_tokenize
import nltk
import re
import unicodedata
import string

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
            palavraSemAcento = re.sub('[^a-zA-Z _.\\\]', '', palavraSemAcento)
            saida = saida +" "+ palavraSemAcento
        saida = saida.strip()
        return saida
    def adicionarTag(self,texto,tag):
        palavras = word_tokenize(texto)
        # Unicode normalize transforma um caracter em seu equivalente em latin.
        saida = ""
        for palavra in palavras:
            saida = saida + " " + palavra+tag
        saida = saida.strip()
        return saida
    def adicionarLabels(self, line):
        line = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EM', line)
        line = re.sub(r'\w+:\/\/\S+', r'_U', line)

        # First off we format white space
        line = line.replace('"', ' ')
        line = line.replace('\'', ' ')
        line = line.replace('_', ' ')
        line = line.replace('-', ' ')
        line = line.replace('\n', ' ')
        line = line.replace('\\n', ' ')
        line = line.replace('\'', ' ')
        line = re.sub(' +', ' ', line)
        line = line.replace('\'', ' ')


        # next we kill off any punctuation issues, employing tags (such as _Q, _X) where we feel the punctuation might lead to useful features.
        line = re.sub(r'([^!\?])(\?{2,})(\Z|[^!\?])', r'\1 _BQ\n\3', line)
        line = re.sub(r'([^\.])(\.{2,})', r'\1 _SS\n', line)
        line = re.sub(r'([^!\?])(\?|!){2,}(\Z|[^!\?])', r'\1 _BX\n\3', line)
        line = re.sub(r'([^!\?])\?(\Z|[^!\?])', r'\1 _Q\n\2', line)
        line = re.sub(r'([^!\?])!(\Z|[^!\?])', r'\1 _X\n\2', line)
        line = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', line)
        line = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', line)
        line = re.sub(r'(\w+)\.(\w+)', r'\1\2', line)

        # more encoding. This time we're encoding things like swearing (_SW). Internet trolls can be pretty sweary.

        line = re.sub(r'([#%&\*\$]{2,})(\w*)', r'\1\2 _SW', line)
        line = re.sub('[1|2|3|4|5|6|7|8|9|0]', '', line)

        line = re.sub(r' [8x;:=]-?(?:\)|\}|\]|>){2,}', r' _BS', line)
        line = re.sub(r' (?:[;:=]-?[\)\}\]d>])|(?:<3)', r' _S', line)
        line = re.sub(r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' _BF', line)
        line = re.sub(r' [x:=]-?[\(\[\|\\/\{<]', r' _F', line)
        line = re.sub('[%]', '', line)

        return line
class AjustadorLetras(object):

    def ajustar(self,texto):
        palavras = word_tokenize(texto)
        saida = ""
        # letras = ['a','e','i','o','u','k']
        letras = list(string.ascii_lowercase)
        letras.remove('r')
        letras.remove('s')
        for palavra in palavras:
            for letra in letras:
                ll = letra+letra
                while ll in palavra:
                    palavra = palavra.replace(ll,letra)
            saida = saida + " " + palavra
        return saida
class DetectorPalavras(object):

    def ajustar(self,texto,linguagem = 'portuguese'):
        palavras = wordpunct_tokenize(texto)
        saida = ""
        palavra =""
        stop_words = nltk.corpus.stopwords.words(linguagem)
        for i in range(len(palavras)):
            if (len(palavras[i])==1 or len(palavras[i])==2) and len(palavra)==0 and (palavras[i] not in stop_words)\
                    and re.match('[a-zA-Z0-9 \\\]',palavras[i]):
                palavra = palavra + palavras[i]
            elif (len(palavras[i])==1 or len(palavras[i])==2) and len(palavra)>0 and re.match('[a-zA-Z0-9 \\\]',palavras[i]):
                palavra = palavra + palavras[i]
            else:
                if len(palavra)>0:
                    saida = saida +" "+ palavra
                    palavra =""
                saida = saida + " " + palavras[i]
        return saida
class AjustadorTamanho(object):
    def ajustarMin(self,texto,min):
        palavras = word_tokenize(texto)
        saida = ""
        for palavra in palavras:
            if len(palavra) >min:
                saida = saida + " " + palavra
        return saida

    def ajustarMax(self, texto, max):
        palavras = word_tokenize(texto)
        saida = ""
        for palavra in palavras:
            if len(palavra) <= max:
                saida = saida + " " + palavra
        return saida


class Processador(object):

    def processar(self,texto,tag=None):
        texto = texto.strip()
        texto = texto.lower()
        aj = AjustadorLetras()
        rc =RemovedorCaracters()
        rs = RemovedorStopWord()
        rr =RemovedorRadicais()
        at = AjustadorTamanho()
        dp = DetectorPalavras()
        texto = rc.adicionarLabels(texto)
        texto = rc.remover(texto)
        texto = dp.ajustar(texto)
        texto = aj.ajustar(texto)
        texto = rs.remover(texto)
        texto = at.ajustarMin(texto,1)
        texto = at.ajustarMax(texto,18)
        texto = rr.remover(texto)
        if(tag is not None):
            texto = rc.adicionarTag(texto,tag)
        return texto