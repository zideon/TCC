#!/usr/bin/python
# coding: utf-8

import requests
import json
from bs4 import BeautifulSoup
import urllib
import time
from Persistencia import GerenciadorMongoDB

class coletor(object):

    def __init__(self):
        pass

    def getElement(text, element):
        i = text.index(element)
        j = text[i:].index(",")
        j = j + i
        i = i + len(element)
        saida = text[i:j].strip()
        saida = saida.replace("\"", "")
        # print(element + saida)
        return saida

    def coletarNoticias(self,nomeBase,tag):
        album = GerenciadorMongoDB.getColecao(nomeBase)
        while indice <= 10:
            urlPai = "http://g1.globo.com/busca/?q=" + tag + "&ps=on&species=not%C3%ADcias&page=" + str(indice)
            print urlPai
            dataPaginaBusca = requests.get(urlPai)
            paginaBusca = BeautifulSoup(dataPaginaBusca.content, "html.parser")

            for row in paginaBusca.find_all('a', attrs={"class": "busca-titulo"}):
                url = row['href']
                i = url.index('http')
                j = url[i:].index("&")
                url = url[i:j + i]
                url = url.replace("%2F", "/")
                url = url.replace("%3A", ":")
                print url
                try:
                    dataPaginaNoticia = requests.get(url)
                    paginaNoticia = BeautifulSoup(dataPaginaNoticia.content, "html.parser")

                    script = paginaNoticia.find('script', {"id": "SETTINGS"})

                    text = script.text;
                    v1 = self.getElement(text, "HOST_COMENTARIOS:")
                    v2 = self.getElement(text, "COMENTARIOS_URI:")
                    v3 = self.getElement(text, "COMENTARIOS_IDEXTERNO:")
                    v4 = self.getElement(text, "CANONICAL_URL:")
                    v5 = self.getElement(text, "TITLE:")
                    x = 1
                    v2 = v2.replace("/", "@@")
                    v2 = urllib.quote(v2.encode("utf-8"))
                    v3 = v3.replace("/", "@@")
                    v3 = urllib.quote(v3.encode("utf-8"))
                    v4 = v4.replace("/", "@@")
                    v4 = urllib.quote(v4.encode("utf-8"))
                    v5 = urllib.quote(v5.encode("utf-8"))
                    bol = True
                    sum = 0
                    while bol:
                        try:
                            time.sleep(1)
                            urlComentarios = v1 + "/comentarios/" + v2.replace("/", "@@") + "/" + v3.replace("/",
                                                                                                             "@@") + "/" + v4.replace(
                                "/", "@@") + "/shorturl/" + v5 + "/" + str(x) + ".json"
                            print urlComentarios
                            header = {'Host': 'comentarios.globo.com', 'User-Agent': 'godagent',
                                      'Content-Type': 'application/json'}
                            t = requests.get(urlComentarios, headers=header)
                            requestJson = BeautifulSoup(t.content, "html.parser")
                            text = requestJson.text
                            text = text.replace('__callback_listacomentarios(', '')
                            text = text[:-1]
                            newDictionary = json.loads(text)
                            itens = newDictionary["itens"]
                            if isinstance(itens, list) and len(itens) > 1:
                                # print len(itens)
                                sum = sum + len(itens)
                                newDictionary['_id'] = newDictionary['agregador']
                                album.insert_one(newDictionary)
                                # print (newDictionary)
                            else:
                                bol = False
                            x = x + 1
                        except requests.exceptions.RequestException as e:  # This is the correct syntax
                            print e
                            bol = False
                            # print sum
                except Exception as e:
                    print e
            indice = indice + 1

import re
import editdistance
from nltk.tokenize import word_tokenize


class filtro(object):
    chaves = ["imoral", "queimar", "isla", "pecado", "nojo", "cancer", "sodoma", "indecente", "dst", "aids", "praga",
              "arabia", "nojento", "dizimar", "lixo", "excretor", "espancar", "morrer", "matar", "matando", "baitola",
              "boiola", "biba", "bicha", "bichinha", "viado", "sapatao", "traveco", "homo", "aberracao", "gay",
              "doente", "doenca", "perversao"]
    def __init__(self):
        pass
    def __init__(self,chaves):
        self.chaves = chaves
    def ativou(self,texto):
        for chave in self.chaves:
            if re.match(".*" + chave + ".*", texto):
                return True
        return False

    def ativou(self, texto,distance):
        palavras  = word_tokenize(texto)
        for chave in self.chaves:
            for w in palavras:
                if editdistance.eval(chave, w) <= distance and editdistance.eval(chave, w) >= (distance*-1):
                    return True
        return False