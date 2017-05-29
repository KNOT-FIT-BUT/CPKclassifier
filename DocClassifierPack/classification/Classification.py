# -*- coding: UTF-8 -*-
"""
Obsahuje nástroje pro trénování klasifikátoru a klasifikování.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV

import math
import logging
from collections import Counter

from ..utils.DataSet import DataTypeSelector
from ..utils.Targets import TargetsTranslator

class Classification(object):
    """
    Třída pro trénování klasifikátoru a klasifikaci.
    """
    

    #Názvy klasifikátorů. Vše malé znaky.
    classifiersNames=["svc", "linearsvc", "multinomialnb", "kneighborsclassifier", "sgdclassifier"]
    
    multinomialNBName="multinomialnb"
    linearSVCName="linearsvc"
    SVCName="svc"
    KNeighborsClassifierName="kneighborsclassifier"
    SGDClassifierName="sgdclassifier"
    

    def __init__(self, classifiersNames, classifierParams={}):
        """
        Inicializace klasifikace.
        
        :param classifiersNames:  list -- trojic (název dat, název klasifikátoru, váha)
        :param classifierParams: dict -- klíč název klasifikátoru a hodnota jsou parametry klasifikátoru v podobě dict.
        """
        
        self.__clsNames=classifiersNames
        
        self.__classifiers=self.__makeClassifiers(classifiersNames, classifierParams)
        self.targets=[]
        
    def train(self, data, targets, balancers=[]):
        """
        Natrénuje klasifikátor.
        
        :param data: dict -- Obsahující data pro trénování. Klíč je název druhu dat.
        :param targets: Cíle pro trénování.
        :param balancers: list -- Obsahující metody vyvažování.
            Budou použity v pořadí v jakém jsou v listu uvedeny.
            Metodu reprezentuje (název metody, parametry)
                
        """
        
        for dataName, classifierName, classifier, w in self.__classifiers:
            logging.info("začátek trénování klasifikátoru "+ classifierName+" s váhou "+str(w)+" pro "+dataName)
            classifier.fit(data, targets)
            logging.info("konec trénování klasifikátoru "+ classifierName+" s váhou "+str(w)+" pro "+dataName)
        
        self.targets=classifier.classes_.tolist()
        
        
        
    def couldGetNBest(self):
        """
        Zjistí jeslti lze získat n nejlepších.
        
        :returns: bool -- True => lze získat
        """

        for _,_,classifier,_ in self.__classifiers:
            if not callable(getattr(classifier, "predict_proba", None)):
                return False
        
        return True
        
        
    def predictAuto(self, data, splitIntoPartsOfMaxSize=None):
        """
        Predikuje cíle pro data. Pokud je možné použít 
        predictProba, tak jej použije pokud ne zvolí predict.
        
        :param data: dict -- Obsahující data pro predikci. Klíč je název druhu dat.
        :param splitIntoPartsOfMaxSize: Pokud je uveden rozdělí množinu dokumentů do částí s maximálním počtem dokumentů definovaných v tomto parametru.
        :returns: list -- Cíle pro data | list --  S jakou jistotou patří data do natrénovaných cílů.
        """
        
        if self.couldGetNBest():
            return self.predictProba(data, splitIntoPartsOfMaxSize)
        
        return self.predict(data, splitIntoPartsOfMaxSize)

        
    def predict(self, data, splitIntoPartsOfMaxSize=None):    
        """
        Predikuje cíle pro data.
        
        :param data: dict -- Obsahující data pro predikci. Klíč je název druhu dat.
        :param splitIntoPartsOfMaxSize: Pokud je uveden rozdělí množinu dokumentů do částí s maximálním počtem dokumentů definovaných v tomto parametru.
        :returns:  list -- Cíle pro data
        """
        logging.info("začátek predikování cílů")
        predictedAll=dict([ (dataName, {}) for dataName, _, _,_ in self.__classifiers])
        
        #pro setreni pameti
        translator=TargetsTranslator()
        
        if hasattr(data[next(iter(data))], "shape"):
            docNum=data[next(iter(data))].shape[0]
        else:
            docNum=len([next(iter(data))])
        
        for dataName, classifierName, classifier, weight in self.__classifiers:
            logging.info("začátek predikování cílů pro "+dataName+" pomocí "+classifierName+" s váhou "+str(weight))
            
            if classifierName not in predictedAll[dataName]:
                predictedAll[dataName][classifierName]={}
                
            if splitIntoPartsOfMaxSize:
                predictedAll[dataName][classifierName][weight]=[]
                
                for i in range(math.ceil(docNum/splitIntoPartsOfMaxSize)):
                    part={}
                    endOfPart=(i+1)*splitIntoPartsOfMaxSize if (i+1)*splitIntoPartsOfMaxSize<docNum else docNum
                    for x in data.keys():
                        part[x]=data[x][i*splitIntoPartsOfMaxSize:endOfPart]
                    predictedAll[dataName][classifierName][weight]= predictedAll[dataName][classifierName][weight]+translator.translate(classifier.predict(part))
                    
                    logging.info("Hotovo: "+str(int(100*len(predictedAll[dataName][classifierName][weight])/docNum))+"% - "+str(len(predictedAll[dataName][classifierName][weight]))+"/"+str(docNum))
            else:
                predictedAll[dataName][classifierName][weight]=translator.translate(classifier.predict(data))
             
            logging.info("konec predikování cílů pro "+dataName+" pomocí "+classifierName+" s váhou "+str(weight))
            
        logging.info("konec predikování cílů")
        
        predicted=[]
        logging.info("začátek kombinování predikovaných cílů")
        for docIndex in range(docNum):
            targets={}
            for dataName, classifierName, _, weight in self.__classifiers:
                tarName=predictedAll[dataName][classifierName][weight][docIndex]
                if tarName not in targets:
                    targets[tarName]=0
                    
                targets[tarName]=targets[tarName]+weight

                
            predicted.append(translator.getOriginal(max(targets, key=lambda x: targets[x])))
            
        logging.info("konec kombinování predikovaných cílů")
        return predicted
        
    def predictProba(self, data, splitIntoPartsOfMaxSize=None):    
        """
        Predikuje cíle pro data. 
        
        :param data: dict -- Obsahující data pro predikci. Klíč je název druhu dat.
        :param splitIntoPartsOfMaxSize: Pokud je uveden rozdělí množinu dokumentů do částí s maximálním počtem dokumentů definovaných v tomto parametru.
        :returns:  list -- S jakou jistotou patří data do natrénovaných cílů.
        """
        
        if hasattr(data[next(iter(data))], "shape"):
            docNum=data[next(iter(data))].shape[0]
        else:
            docNum=len([next(iter(data))])

        
        predicted=[ [0]*len(self.targets) for docIndex in range(docNum)]
        
        
        
        for dataName, classifierName, classifier, weight in self.__classifiers:

            
            logging.info("začátek predikování cílů pro "+dataName+" pomocí "+classifierName+" s váhou "+str(weight))
            if splitIntoPartsOfMaxSize:
                
                cntPred=0
                for i in range(math.ceil(docNum/splitIntoPartsOfMaxSize)):
                    part={}
                    endOfPart=(i+1)*splitIntoPartsOfMaxSize if (i+1)*splitIntoPartsOfMaxSize<docNum else docNum
                    for x in data.keys():
                        part[x]=data[x][i*splitIntoPartsOfMaxSize:endOfPart]
                    
                    for doc in classifier.predict_proba(part):
                        predicted[cntPred]=[a+b*weight for a,b in zip(predicted[cntPred], doc)]
                        cntPred=1+cntPred
                        
                    logging.info("Hotovo: "+str(int(100*cntPred/docNum))+"% - "+str(cntPred)+"/"+str(docNum))
            else:
                for x, doc in enumerate(classifier.predict_proba(data)):
                    predicted[x]=[a+b*weight for a,b in zip(predicted[x], doc)]
                    
            logging.info("konec predikování cílů pro "+dataName+" pomocí "+classifierName+" s váhou "+str(weight))
            
        logging.info("začátek kombinování predikovaných cílů")
        
        weightSum=sum([weight for _, _, _, weight in self.__classifiers])
        for docIndex in range(docNum):
            predicted[docIndex]=[x/weightSum for x in predicted[docIndex]]
                
        
        logging.info("konec kombinování predikovaných cílů")
        
        return predicted
    
    def __makeClassifiers(self, classifiersNames, classifiersParams={}):
        """
        Na základě jména vytvoří klasifikátory.
        
        :param classifiersNames: list -- trojic (název dat, název klasifikátoru, váha)
        :param classifiersParams: dict -- klíč název klasifikátoru a hodnota jsou parametry klasifikátoru.
        :returns: Klasifikátory list -- páru (název dat, název klasifikátoru,klasifikátor, váha)
        """
        
        classifiers=[]
        
        for dataName, classifierName, w in classifiersNames:
            classifiers.append((dataName, classifierName, Pipeline([
                        ('dataSel', DataTypeSelector(dataName)),
                        ('cls', self.__makeClassifier(classifierName, classifiersParams))
                    ]),w))
                    
        return classifiers
        
        
    def __makeClassifier(self, classifiersNames, classifierParams={}):
        """
        Na základě jména vytvoří klasifikátor.
        
        :param classifiersNames: název klasifikátoru.
        :param classifierParams: dict -- klíč název klasifikátoru a hodnota jsou parametry klasifikátoru.
        :returns: Klasifikátor
        """
        
        clsName=classifiersNames.lower()
        if clsName==self.multinomialNBName:
            return MultinomialNB()
        elif clsName==self.linearSVCName:
            return CCWrapper(LinearSVC())
        elif clsName==self.SVCName:
            return SVC(kernel="linear", probability=True)
        elif clsName==self.KNeighborsClassifierName:
            return KNeighborsClassifier(**classifierParams[self.KNeighborsClassifierName])
        elif clsName==self.SGDClassifierName:
            return CCWrapper(SGDClassifier())
            
            


class CCWrapper(object):
    """
    Obaluje klasifikátor, aby uměl vracet pravděpodobnosti tříd při predikci.
    Použivá k tomu CalibratedClassifierCV.
    """
    
    def __init__(self, classifier):
        """
        Inicializace.
        
        :param classifier: Klasifikátor, který chceme obalit.
        """
        self.classifier=classifier
        self.classes_=None
    
    def fit(self, X, y, sampleWeight=None):
        """
        Natrénuje klasifikátor.
        
        :param X: array-like, sparse matrix -- trénovací vektory [n_vektorů, n_příznaků]
        :param y: array-like, [n_samples] cíle k trénovacím vektorům
        :param sampleWeight: array-like [n_samples] váhy k trénovacím vektorům. Implicitně jednotková.
        """
        
        if Counter(y).most_common()[-1][1]==1:
            self.classifier.fit(X, y, sampleWeight)
            self.cls=CalibratedClassifierCV(self.classifier, cv='prefit').fit(X, y)
        elif Counter(y).most_common()[-1][1]==2:
            self.cls = CalibratedClassifierCV(self.classifier, cv=2) 
            self.cls.fit(X, y)
        else:
            self.cls = CalibratedClassifierCV(self.classifier) 
            self.cls.fit(X, y)
            
            
        self.classes_=self.cls.classes_
    
    def predict(self, X):
        """
        Predikuje cíle pro daná data
        
        :param X: array-like, sparse matrix -- vektory pro predikci [n_vektorů, n_příznaků]
        :returns: Cíle pro data.
        """
        
        return self.cls.predict(X)
    
    def predict_proba(self, X):
        """
        Získání pravděpodobností predikce k jednotlivým cílům.
        
        :param X: array-like, sparse matrix -- vektory pro predikci [n_vektorů, n_příznaků]
        :returns: Pravděpodobnosti ke každému cíli a to pro každý vektor z X. Pořádí cílů lze zjistit pomocí classes _.
        """
        
        return self.cls.predict_proba(X)
    
    
    
