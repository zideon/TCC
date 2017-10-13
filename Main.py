from Classificador import PipeLineFactory,GerenciadorCluster
from Persistencia import GeradorEntradas
from Persistencia import GerenciadorMongoDB
from Avaliador import AvaliadorFactory
from sklearn import metrics
import tf_glove

pf = PipeLineFactory()
ge = GeradorEntradas()
gmdb = GerenciadorMongoDB()
# gmdb.converteG1paraNoticias('AllData','NovaFullBase')
# gmdb.converteHerokuParaClassificacao('ManyUsers','ManyClassificacoes')

model = pf.getSVC(0)
gc = GerenciadorCluster()
af = AvaliadorFactory()
X,y = ge.getXy({0:1,1:0,2:1},'FullBase','uniao')
ids,titulos,corpus = ge.getDiferencaClassificacaoBase("FullBase",'uniao')

model.fit(X,y)
for i in range(len(ids)):
    if
    print titulos[i]+" "+corpus[i]
    print model.predict(titulos[i]+" "+corpus[i])
    print model.predict_proba(titulos[i]+" "+corpus[i])

# print len(corpus)
# saida = gc.kmeansBagOfWords(ids,corpus,10)
# saida = gc.kmeansTDIDF(ids,titulos,corpus,100)
#
# for i in saida:
#     gmdb.salvarModelo("clusterTDIDF",i)

# clusters = gmdb.getClusters("clusterTDIDF")
# model = pf.getSVC(0)
# X,y = ge.getXy({0:1,1:0,2:1},'FullBase','ManyClassificacoes')
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

