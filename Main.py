from Classificador import PipeLineFactory
from Persistencia import GeradorEntradas
from Avaliador import AvaliadorFactory
pf = PipeLineFactory()
ge = GeradorEntradas()
model = pf.getSVC(0)
af = AvaliadorFactory()
entradas = ge.getEntradas({0:1,1:0,2:1},'join','OnlyUserFocusHate')
X = [i.getInput() for i in entradas]
y = [i.getClasse() for i in entradas]
af.divisaoTrainTest(model,["odio","neutro"],X,y)