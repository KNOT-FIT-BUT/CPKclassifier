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
from sklearn.metrics import confusion_matrix,accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

import math
import logging
import numpy as np
from collections import Counter
from multiprocessing import Process, cpu_count, active_children, Lock, Manager

import queue
import sys
import traceback
import gc


from CPKclassifierPack.utils.DataSet import DataTypeSelector
from CPKclassifierPack.utils.Targets import TargetsTranslator
from .Classifiers import MatchTargetClassifier, KMeansClassifier
from CPKclassifierPack.features.Features import FeaturesContainer
from CPKclassifierPack.utils.Parallel import SharedStorage




class Classification(object):
    """
    Třída pro trénování klasifikátoru a klasifikaci.
    """
    

    #Názvy klasifikátorů. Vše malé znaky.
    classifiersNames=["svc", "linearsvc", "multinomialnb", "kneighborsclassifier", "sgdclassifier", "matchtargetclassifier", "kmeansclassifier"]
    
    multinomialNBName="multinomialnb"
    linearSVCName="linearsvc"
    SVCName="svc"
    KNeighborsClassifierName="kneighborsclassifier"
    SGDClassifierName="sgdclassifier"
    matchTargetClassifierName="matchtargetclassifier"
    KMeansClassifierName="kmeansclassifier"
    
    MAX_WAIT_TIMEOUT=10
    

    def __init__(self, classifiersNames, classifierParams={}, cv=4):
        """
        Inicializace klasifikace.
        
        :param classifiersNames:  list -- čtveřic (název dat, název klasifikátoru, váha, práh)
                Pokud je váha string auto. Tak bude při trénování váha získána z úspěšnosti na trénovací/testovací množině.
                Pokud je váha číslo, tak je použito jako váhe ke všem kategoriím/cílům.
        :param classifierParams: dict -- klíč název klasifikátoru a hodnota jsou parametry klasifikátoru v podobě dict.
        :type cv: int 
        :param cv: Počet křížově validačních kroků při získávání vah, pokud je jako váha uvedeno u klasifikátoru auto.
        """
        
        self.__clsNames=classifiersNames

        #váhy jednotlivých kategorií/cílů u jednotlivých klasifikátorů. Klíč je index odpovídajícího klasifikátoru v self.__classifiers.
        self.categoriesWeights={}    
        
        self.cv=cv
        self.__classifiers=self.__makeClassifiers(classifiersNames, classifierParams)
        self.targets=[]
        self.errorBoard=None    #používá se pro hlášení chyb z ostatních procesů
        
    def __manageWorkers(self, workers):
        """
        Pokud je workers nastaveno na -1. Převede jej na počet cpu.
        
        :param workers: Počet procesů pro převod.
        :return: Počet procesů, které chceme použít.
        """
        if workers==-1:
            try:
                workers=cpu_count()
            except:
                workers=1
        
        return workers
    
    def __train(self, data, targets, workers=1):
        """
        Natrénuje klasifikátor.
        
        :param data: dict -- Obsahující data pro trénování. Klíč je název druhu dat.  Samotná data jsou uchovávána v FeaturesContainer.
        :param targets: Cíle pro trénování.
            Budou použity v pořadí v jakém jsou v listu uvedeny.
            Metodu reprezentuje (název metody, parametry)
        :param workers: Počet pracujicích procesů. Určuje kolik klasifikátorů zároveň bude trénováno.
        """

        if workers>len(self.__classifiers):
            #více pracantů něž-li je nutné
            workers=len(self.__classifiers)
            
        if workers==1:
            #jedno procesová varianta
            for dataName, classifierName, classifier, w, t in self.__classifiers:
                TrainWorker.trainCls(targets, data[dataName], dataName, classifierName, classifier, w, t)

        else:
            #více procesová varianta
            manager=Manager()
            processes=[]
            inputDataQueue=manager.Queue()
            resultsStorage=TrainWorker.ClassifiersStorage(manager=manager)
            self.errorBoard=manager.Queue()
            sharedLock=Lock()
            
            for _ in range(0,workers-1):
                p=TrainWorker(inputDataQueue, resultsStorage, self.errorBoard, sharedLock, data)
                processes.append(p)
                p.start()
            
            
            for dataName, classifierName, classifier, w, t in self.__classifiers:
                inputDataQueue.put((targets, dataName, classifierName, classifier, w, t))
                
            #vložíme příznaky konce.
            for _ in range(0,workers):
                inputDataQueue.put("EOF")
            

            #Zkontrolujeme chyby a kdyžtak pomůžeme ostatním pokud je třeba
            #přitom ukládáme výsledky

            helperP=TrainWorker(inputDataQueue, resultsStorage, self.errorBoard, sharedLock, data)

            shouldHelp=True
            while True:
                self.__controlMulPErrors()
                
                resultsStorage.acquire()
                if len(resultsStorage)>=len(self.__classifiers):
                    #hotovo
                    break
                resultsStorage.release()
                if shouldHelp:
                    #pomáháme
                    
                    if helperP.run() =="EOF":
                        shouldHelp=False
                else:
                    #čekáme na klasifikátory
                    resultsStorage.waitForChange(self.MAX_WAIT_TIMEOUT)
                    
            self.__controlMulPErrors()
            #sbíráme výsledky
            for clsInfo in resultsStorage.popResults():
                dataName, classifierName, classifier, weight, threshold=clsInfo
                for i, (actDataName, actClassifierName, _, actWeight, actThreshold) in enumerate(self.__classifiers):
                    if dataName==actDataName and classifierName==actClassifierName and weight==actWeight and threshold==actThreshold:
                        logging.info("\tUkládám. "+classifierName+" pro "+dataName+" s váhou "+str(actWeight)+" s prahem "+str(actThreshold))
                        self.__classifiers[i]=(dataName, classifierName, classifier, actWeight, actThreshold)
                        break

            #čekáme na ukončení
            for proc in processes:
                self.__controlMulPErrors()
                proc.join()  
            
        
    def train(self, data, targets, workers=1):
        """
        Natrénuje klasifikátor.
        
        :param data: dict -- Obsahující data pro trénování. Klíč je název druhu dat.  Samotná data jsou uchovávána v FeaturesContainer.
        :param targets: Cíle pro trénování.
            Budou použity v pořadí v jakém jsou v listu uvedeny.
            Metodu reprezentuje (název metody, parametry)
        :param workers: Počet pracujicích procesů. Určuje kolik klasifikátorů zároveň bude trénováno.
        """
        logging.info("začátek trénování")
                
        workers=self.__manageWorkers(workers)
        
        savedClassifiers=self.__classifiers
        self.__classifiers=[]
        clsIndexes=[]
        
        self.targets=np.unique(targets).tolist()
        #zjistíme, kde máme získat váhy automaticky a kde ne
        autoWeights=False
        for i, x in enumerate(savedClassifiers):
            if x[3]=="auto":
                autoWeights=True
                self.__classifiers.append(x)
                clsIndexes.append(i)
                
                #nulujeme
                self.categoriesWeights[i]=[0 for _ in range(len(self.targets))]
            else:
                #rovnou přidáme váhu
                self.categoriesWeights[i]=[x[3] for _ in range(len(self.targets))]


        if autoWeights:
            #je nutné získat váhy
            logging.info("\tzačátek získávání vah")
            
            #Zlomek udávající poměr mezi všemi cíli/kategoriemi, které byly poskytnuty 
            #pro trénování a cíli/kategoriemi, které daný klasifikátor má natrénované.
            #Tento poměr může být odlišný od 1, díky prázdným dokumentům, které mohou být ignorovány.
            targetsCompletnesIndex=[1]*len(self.__classifiers)    
            
            #uložiště vah
            storage=AutoWeightWorker.WeightStorage()
            
            logging.disable(logging.INFO)
            
            if workers==1:
                #jedno procesová varianta
                for i, (dataName, classifierName, classifier, w, t) in enumerate(self.__classifiers):
                    #křížová validace
                    
                    #získání neprázdných
                    actData, actTargets=self.filterMarkedDataWithTargets(data[dataName], targets)
                    
                    numpyY=np.array(actTargets)
                    
                    targetsCompletnesIndex[i]=len(set(actTargets))/len(self.targets)
                    
                    for r, (trainIndex, testIndex) in enumerate(StratifiedKFold(n_splits=self.cv, random_state=0).split(actData,actTargets)):
                        trainData, trainTargets=actData[trainIndex,:], numpyY[trainIndex]
                        testData, testTargets=actData[testIndex,:], numpyY[testIndex]
                        
                        #trénování a zjišťování úspěšnosti
                        TrainWorker.trainCls(trainTargets, FeaturesContainer(trainData, []), dataName, classifierName, classifier, w, t)
                        
                        _, _, f1, _ = precision_recall_fscore_support(testTargets, classifier.predict({dataName:testData}))
                        
                        #uložíme úspěšnost
                        storage.addResultsFor(i, f1)
                            
                        logging.disable(logging.NOTSET)
                        logging.info("\t\t"+str(round(((i*self.cv)+r+1)/(self.cv*len(self.__classifiers))*100))+"%")
                        logging.disable(logging.INFO)
                        
            else:
                #více procesová varianta
                processes=[]
                manager=Manager()
                inputDataQueue=manager.Queue()
                self.errorBoard=manager.Queue()
                sharedLock=Lock()

                
                
                #nastavíme dostatečný počet pracovníků
                tmpWorkers=len(self.__classifiers)*self.cv if workers>len(self.__classifiers)*self.cv else workers

                for _ in range(0,tmpWorkers-1):
                    p=AutoWeightWorker(inputDataQueue, storage, self.errorBoard, sharedLock)
                    processes.append(p)
                    p.start()
                    
                #vložíme všechny potřebné kroky do fronty
                for i, (dataName, classifierName, classifier, w, t) in enumerate(self.__classifiers):

                    #získání neprázdných
                    actData, actTargets=self.filterMarkedDataWithTargets(data[dataName], targets)
                    numpyY=np.array(actTargets)
                    
                    targetsCompletnesIndex[i]=len(set(actTargets))/len(self.targets)
                    
                    #všechny klasifikátory
                    for r, (trainIndex, testIndex) in enumerate(StratifiedKFold(n_splits=self.cv, random_state=0).split(actData,actTargets)):
                        #a k nim všechny křížově validační kroky
                        inputDataQueue.put((numpyY[trainIndex], 
                                            FeaturesContainer(actData[trainIndex,:], []), 
                                            dataName, 
                                            classifierName, 
                                            classifier, 
                                            w, 
                                            t,
                                            numpyY[testIndex],
                                            actData[testIndex,:],
                                            i))
                    
                #budeme se snažit pomáhat
                helperWorker=AutoWeightWorker(inputDataQueue, storage, self.errorBoard, sharedLock)
                
                
                #vložíme příznaky konce.
                for _ in range(0,tmpWorkers-1):
                    inputDataQueue.put("EOF")
                    
                completedCnt=0
                shouldHelp=True
                #Zkontrolujeme chyby a kdyžtak pomůžeme ostatním pokud je třeba
                while completedCnt<len(self.__classifiers)*self.cv:
                    self.__controlMulPErrors()
                    
                    if shouldHelp:
                        #pomůžeme
                        if helperWorker.run(True)=="EOF":
                            #končíme s pomáháním
                            shouldHelp=False
                    
                    storage.acquire()

                    #zjistíme výsledky
                    if len(storage)!=completedCnt:
                        #máme nové výsledky
                        completedCnt=len(storage)
                           
                        logging.disable(logging.NOTSET) 
                        logging.info("\t\tHotovo: "+str(round(completedCnt/(len(self.__classifiers)*self.cv)*100))+"%")
                        logging.disable(logging.INFO)
                    else:
                        #nemáme nové výsledky, počkáme na ně
                        storage.waitForChange(self.MAX_WAIT_TIMEOUT)
                        
                    storage.release()

                
                    
                #čekáme na ukončení
                for proc in processes:
                    self.__controlMulPErrors()
                    proc.join()   

            #uložíme váhy
            for i in range(len(self.__classifiers)):
                for ti, w in enumerate(storage.getAvgResultsFor(i)):
                    self.categoriesWeights[i][ti]=w*targetsCompletnesIndex[i]
                    
            logging.disable(logging.NOTSET)
            
            
            for i, (dataName, classifierName, _, _, t) in enumerate(self.__classifiers):
                logging.info("\t\tZískané váhy u klasifikátoru "+ classifierName+" pro "+dataName+" s prahem "+str(t)+": ")
                for catI, catW in sorted(enumerate(self.categoriesWeights[i]), key=lambda x: x[1], reverse=True):
                    logging.info("\t\t\t"+str(catW)+"\t"+self.targets[catI])
                    
            logging.info("\tkonec získávání vah")
            
            
        self.__classifiers=savedClassifiers
        
        #natrénujeme klasifikátor
        self.__train(data, targets, workers)
        
        self.errorBoard=None    #Odstraníme nepotřebnou frontu, také kvůli případnému dumpu.
        
        logging.info("konec trénování")
        
    def couldGetNBest(self):
        """
        Zjistí jeslti lze získat n nejlepších.
        
        :returns: bool -- True => lze získat
        """

        for _,_,classifier,_,_ in self.__classifiers:
            if not callable(getattr(classifier, "predict_proba", None)):
                return False
        
        return True
        
        
    def predictAuto(self, data, splitIntoPartsOfMaxSize=None, threshold=0.0, workers=1):
        """
        Predikuje cíle pro data. Pokud je možné použít 
        predictProba, tak jej použije pokud ne zvolí predict.
        
        :param data: dict -- Obsahující data pro predikci. Klíč je název druhu dat. Samotná data jsou uchovávána v FeaturesContainer.
        :param splitIntoPartsOfMaxSize: Pokud je uveden rozdělí množinu dokumentů do částí s maximálním počtem dokumentů definovaných v tomto parametru.
        :param threshold: Udává minimální míru jistoty, která je třeba k uznání predikce. Pokud je menší je dokument neklasifikován. Aplikuje se pouze v případě, že lze použit predictProba.
        :param workers: Udává počet procesů podílejících se na predikci. Pokud je workers!=1, tak je ignorován parametr splitIntoPartsOfMaxSize
            a jsou vybrány části automaticky, tak aby došlo k jejich vhodnému přerozdělení mezi procesy.
        :returns: list -- Cíle pro data | list --  S jakou jistotou patří data do natrénovaných cílů.
        """
        
        if self.couldGetNBest():
            return self.predictProba(data, splitIntoPartsOfMaxSize, threshold, workers)
        
        return self.predict(data, splitIntoPartsOfMaxSize, workers)

        
    def predict(self, data, splitIntoPartsOfMaxSize=None, workers=1):    
        """
        Predikuje cíle pro data.
        
        :param data: dict -- Obsahující data pro predikci. Klíč je název druhu dat. Samotná data jsou uchovávána v FeaturesContainer.
        :param splitIntoPartsOfMaxSize: Pokud je uveden rozdělí množinu dokumentů do částí s maximálním počtem dokumentů definovaných v tomto parametru.
        :param workers: Udává počet procesů podílejících se na predikci. Pokud je workers!=1, tak je parametr splitIntoPartsOfMaxSize chápán jako maximální
            velikost bloku pro jednu predikci (tyto bloky dále rovnoměrně dělí mezi procesy), pokud uveden není jsou vytvořené rovnoměrné bloky pro každý proces.
        :returns:  list -- Cíle pro data
        """
        logging.info("začátek predikování cílů")
        workers=self.__manageWorkers(workers)
        predictedAll=dict([ (dataName, {}) for dataName, _, _,_,_ in self.__classifiers])
        
        #pro setreni pameti
        translator=TargetsTranslator()
        
        
        for dataName, classifierName, classifier, weight, _ in self.__classifiers:
            logging.info("začátek predikování cílů pro "+dataName+" pomocí "+classifierName+" s váhou "+str(weight))
            
            #získáme daný druh dan, který má být předložen aktuálnímu klasifikátoru
            #Nechcem klasifikovat pomocí tohoto klasifikátoru data, která jsou označena pro vynechání.
            emptyIndexes=data[dataName].empty
            actData={dataName:data[dataName].features}
                
            docNum=actData[next(iter(actData))].shape[0]
            
            if classifierName not in predictedAll[dataName]:
                predictedAll[dataName][classifierName]={}
                
                
            if splitIntoPartsOfMaxSize or workers>1:
                
                predictedAll[dataName][classifierName][weight]=[]
                
                #nastavíme velikost jednoho bloku
                partSize=docNum/workers

                if splitIntoPartsOfMaxSize and splitIntoPartsOfMaxSize<partSize:
                    partSize=splitIntoPartsOfMaxSize
                    
                if workers>1:
                    #inicializace víceprocesového  zpracování
                    logging.info("\tpočet podílejících se procesů: "+ str(workers))
                    processes=[]
                    manager=Manager()
                    inputDataQueue=manager.Queue()
                    resultsStorage=PredictWorker.PredictedStorage(manager=manager)
                    self.errorBoard=manager.Queue()
                    
                    for i in range(0,workers-1):
                        p=PredictWorker(classifier, inputDataQueue, resultsStorage, self.errorBoard)
                        processes.append(p)
                        p.start()

                    
                for i in range(math.ceil(docNum/partSize)):
                    #extrahujeme část
                    part={}
                    endOfPart=(i+1)*partSize if (i+1)*partSize<docNum else docNum
                    part[dataName]=actData[dataName][i*partSize:endOfPart]
                    
                    if workers>1:
                        #předáváme práci ostatním procesům
                        inputDataQueue.put((i, part))
                    else:
                        #rovnou zpracováváme v tomto procesu
                        predictedAll[dataName][classifierName][weight]= predictedAll[dataName][classifierName][weight]+translator.translate(classifier.predict(part))
                        logging.info("Hotovo: "+str(int(100*len(predictedAll[dataName][classifierName][weight])/docNum))+"% - "+str(len(predictedAll[dataName][classifierName][weight]))+"/"+str(docNum))
                        
                
                if workers>1:
                    #Máme rozjeté procesy, musíme od nich získat výsledky jejich práce, 
                    #ukončit je, hlídat chyby a popřípadě jim i pomoci.
                    

                    helperWorkerP=PredictWorker(classifier, inputDataQueue, resultsStorage, self.errorBoard)
                    #Zkontrolujeme chyby a kdyžtak pomůžeme ostatním pokud je třeba
                    
                    shouldHelp=True
                    #vložíme příznaky konce.
                    for _ in range(0,workers):
                        inputDataQueue.put("EOF")
                        
                    while len(resultsStorage)<docNum:
                        #kontrola chyb
                        self.__controlMulPErrors()
   
                        if shouldHelp:
                            if helperWorkerP.run(once=True)=="EOF":
                                shouldHelp=False
                            
                        else:
                            resultsStorage.waitForChange(self.MAX_WAIT_TIMEOUT)
                        
                        resultsStorage.acquire()
                        if int(len(resultsStorage)/docNum)!=1:
                            logging.info("Hotovo: "+str(int(100*len(resultsStorage)/docNum))+"% - "+str(len(resultsStorage))+"/"+str(docNum))
                        resultsStorage.release()
                        
                            
                    logging.info("Hotovo: "+str(int(100*len(resultsStorage)/docNum))+"% - "+str(len(resultsStorage))+"/"+str(docNum))
                    
                        
                    #čekáme na ukončení
                    for proc in processes:
                        self.__controlMulPErrors()
                        proc.join()  
                            
                    #uložíme výsledky ve správném pořadí na své místo
                    predictedAll[dataName][classifierName][weight]=translator.translate(resultsStorage.popResults())

            else:
                predictedAll[dataName][classifierName][weight]=translator.translate(classifier.predict(actData))
            
            if emptyIndexes:
                #Kvůli výslednému slučování výsledků musíme vložit do predikcí i neklasifikované dokumenty, které jsme
                #odstranili, protože byly označené.
                for empDocInd in emptyIndexes:
                    predictedAll[dataName][classifierName][weight].insert(empDocInd, None)
                
                 
            logging.info("konec predikování cílů pro "+dataName+" pomocí "+classifierName+" s váhou "+str(weight))
            
        logging.info("konec predikování cílů")
        
        predicted=[]
        
        docNum=len(data[next(iter(data))])
            
        logging.info("začátek kombinování predikovaných cílů")
        for docIndex in range(docNum):
            targets={}
            for ci, (dataName, classifierName, _, weight, _) in enumerate(self.__classifiers):
                tarName=predictedAll[dataName][classifierName][weight][docIndex]
                if tarName is None:
                    #nemá se podílet na klasifikaci
                    continue
                
                if tarName not in targets:
                    targets[tarName]=0
                
                targets[tarName]=targets[tarName]+self.categoriesWeights[ci][self.targets.index(tarName)]

            if len(targets)==0:
                #Nebyl predikován žádný cíl.
                #Všechna data byla označena, že se nemají klasifikovat.
                #Vložíme neklasifikováno.
                predicted.append("")
                
            else:
                predicted.append(translator.getOriginal(max(targets, key=lambda x: targets[x])))
            
        logging.info("konec kombinování predikovaných cílů")
        return predicted
        
    def predictProba(self, data, splitIntoPartsOfMaxSize=None, threshold=0.0, workers=1):    
        """
        Predikuje cíle pro data. 
        
        :param data: dict -- Obsahující data pro predikci. Klíč je název druhu dat. Samotná data jsou uchovávána v FeaturesContainer.
        :param splitIntoPartsOfMaxSize: Pokud je uveden rozdělí množinu dokumentů do částí s maximálním počtem dokumentů definovaných v tomto parametru.
        :param threshold: Udává minimální míru jistoty, která je třeba k uznání predikce. Pokud je menší je dokument neklasifikován. Pracuje s výslednou jistotou.
        :param workers: Udává počet procesů podílejících se na predikci. Pokud je workers!=1, tak je ignorován parametr splitIntoPartsOfMaxSize
            a jsou vybrány části automaticky, tak aby došlo k jejich vhodnému přerozdělení mezi procesy.
        :returns:  list -- S jakou jistotou patří data do natrénovaných cílů.
        """
        workers=self.__manageWorkers(workers)
        docNum=len(data[next(iter(data))])

        
        predicted=[ [0]*len(self.targets) for _ in range(docNum)]

        
        
        for ci, (dataName, classifierName, classifier, weight, clsThreshold) in enumerate(self.__classifiers):
            
            emptyIndexes=data[dataName].empty
            #Nechcem klasifikovat pomocí tohoto klasifikátoru data, která jsou označena pro vynechání..
            actData={dataName:data[dataName].features}
                
            #Ne všechny klasifikátory musí mít natrénovánou stejnou množinu cílů (kvůli označeným dokumentům, které se mají vynechávat).
            #Jelikož přeskakujeme některé dokumenty, které mají označený daný druh dat.
            #Zjistíme, které indexy/kategorie chybí a na jejich místo poté vložíme nuly.
                
            actClasses=classifier.classes_.tolist()

            globalProbaIndexesMissing=[i for i, clsName in enumerate(self.targets) if clsName not in actClasses]

                
            actDocNum=actData[next(iter(actData))].shape[0]
            
            logging.info("začátek predikování cílů pro "+dataName+" pomocí "+classifierName+" s váhou "+str(weight)+" a prahem "+str(clsThreshold))
            
            
            inputDataQueue=None
            resultsQueue=None
            
            helperWorkerP=PredictProbaWorker(classifier, self.targets, 
                                                 globalProbaIndexesMissing, self.categoriesWeights[ci], clsThreshold, 
                                                 inputDataQueue, resultsQueue, self.errorBoard)
            
            
            if splitIntoPartsOfMaxSize or workers>1:

                #nastavíme velikost jednoho bloku
                partSize=actDocNum/workers

                if splitIntoPartsOfMaxSize and splitIntoPartsOfMaxSize<partSize:
                    partSize=splitIntoPartsOfMaxSize
                    
                
                
                if workers>1:
                    #inicializace víceprocesového  zpracování
                    logging.info("\tpočet podílejících se procesů: "+ str(workers))
                    processes=[]
                    manager=Manager()
                    inputDataQueue=manager.Queue()
                    resultsStorage=PredictWorker.PredictedStorage(manager=manager)
                    self.errorBoard=manager.Queue()
                    
                    for i in range(0,workers-1):
                        p=PredictProbaWorker(classifier, self.targets, 
                                                 globalProbaIndexesMissing, self.categoriesWeights[ci], clsThreshold, 
                                                 inputDataQueue, resultsStorage, self.errorBoard)
                        processes.append(p)
                        p.start()
                        
                    helperWorkerP=PredictProbaWorker(classifier, self.targets, 
                                                 globalProbaIndexesMissing, self.categoriesWeights[ci], clsThreshold, 
                                                 inputDataQueue, resultsStorage, self.errorBoard)
                        
                cntPred=0
                cntPred=self.__incWhileIn(cntPred, emptyIndexes)
                
                cnt=0

                for i in range(math.ceil(actDocNum/partSize)):
                    part={}
                    endOfPart=(i+1)*partSize if (i+1)*partSize<actDocNum else actDocNum

                    part[dataName]=actData[dataName][i*partSize:endOfPart]
                    
                    if workers>1:
                        #předáváme práci ostatním procesům
                        inputDataQueue.put((i, part))
                    else:
                        #rovnou zpracováváme v tomto procesu
                        
                        for doc in helperWorkerP.predict(classifier, part):
                            #provedeme sloučení s předchozími výsledky
                            for cntPos, actPredicted in enumerate(doc):
                                predicted[cntPred][cntPos]+=actPredicted
                            
                            cntPred+=1
                            cntPred=self.__incWhileIn(cntPred, emptyIndexes)
                            cnt+=1
                            
                        logging.info("Hotovo: "+str(int(100*cnt/actDocNum))+"% - "+str(cnt)+"/"+str(actDocNum))
                        
                if workers>1:
                    #Máme rozjeté procesy, musíme od nich získat výsledky jejich práce, 
                    #ukončit je, hlídat chyby a popřípadě jim i pomoci.
                    
                    shouldHelp=True
                    #vložíme příznaky konce.
                    for _ in range(0,workers):
                        inputDataQueue.put("EOF")

                    #Zkontrolujeme chyby a kdyžtak pomůžeme ostatním pokud je třeba
                    while len(resultsStorage)<actDocNum:
                        #kontrola chyb
                        self.__controlMulPErrors()

                        if shouldHelp:
                            #pomoc ostatním
                            if helperWorkerP.run(once=True)=="EOF":
                                shouldHelp=False
                            
                        else:
                            resultsStorage.waitForChange(self.MAX_WAIT_TIMEOUT)
                            
                        resultsStorage.acquire()
                        if int(len(resultsStorage)/actDocNum)!=1:
                            logging.info("Hotovo: "+str(int(100*len(resultsStorage)/actDocNum))+"% - "+str(len(resultsStorage))+"/"+str(actDocNum))
                        resultsStorage.release()
                                                
                    logging.info("Hotovo: "+str(int(100*len(resultsStorage)/actDocNum))+"% - "+str(len(resultsStorage))+"/"+str(actDocNum))
                    
                    
                    #čekáme na ukončení
                    for proc in processes:
                        self.__controlMulPErrors()
                        proc.join()
                        
                    logging.info("\tzačátek slučování výsledků")
                        
                    # zpracování výsledků
                    cntPred=self.__incWhileIn(0, emptyIndexes)
                    for doc in resultsStorage.popResults():
                        #provedeme sloučení s předchozími výsledky
                        for cntPos,actPredicted in enumerate(doc):
                            predicted[cntPred][cntPos]+=actPredicted
                                
                        cntPred+=1
                        cntPred=self.__incWhileIn(cntPred, emptyIndexes)
                        
                    logging.info("\tkonec slučování výsledků")
            else:
                cntPred=0
                cntPred=self.__incWhileIn(cntPred, emptyIndexes)
                    
                for doc in helperWorkerP.predict(classifier, actData):
                    #provedeme sloučení s předchozími výsledky
                    for cntPos,actPredicted in enumerate(doc):
                        predicted[cntPred][cntPos]+=actPredicted
                    
                    cntPred+=1
                    cntPred=self.__incWhileIn(cntPred, emptyIndexes)
                    
                    
            logging.info("konec predikování cílů pro "+dataName+" pomocí "+classifierName+" s váhou "+str(weight)+" a prahem "+str(clsThreshold))
            
        logging.info("začátek kombinování predikovaných cílů")
        
        #pro zprumerovani ziskame sumy vsech vah k jednotlivym kategoriim.
        completeWeightsSum={}
        for catI in range(len(self.targets)):
            completeWeightsSum[catI]=sum(weights[catI] for weights in self.categoriesWeights.values())

        for docIndex in range(docNum):
            for cntPos, x in enumerate(predicted[docIndex]):        
                finalConfidence=x/completeWeightsSum[cntPos]
                #kontrola prahu
                if finalConfidence>=threshold:
                    predicted[docIndex][cntPos]=finalConfidence
                else:
                    
                    predicted[docIndex][cntPos]=0
            
        logging.info("konec kombinování predikovaných cílů")
        self.errorBoard=None    #Odstraníme nepotřebnou frontu, také kvůli případnému dumpu.
        return predicted
    
    def __incWhileIn(self, cnt, skipSet):
        """
        Zvyšuje počítadlo cnt o 1, dokud je aktuální hodnota počítadla v skipSet. 
        :param cnt: Počitadlo.
        :param skipSet: Hodnoty, které se mají přeskočit.
        :return: Nová hodnota počítadla
        """
        
        while cnt in skipSet:
            cnt+=1
        return cnt
    
    def __makeClassifiers(self, classifiersNames, classifiersParams={}):
        """
        Na základě jména vytvoří klasifikátory.
        
        :param classifiersNames: list -- čtveřic (název dat, název klasifikátoru, váha, práh)
        :param classifiersParams: dict -- klíč název klasifikátoru a hodnota jsou parametry klasifikátoru.
        :returns: Klasifikátory list -- páru (název dat, název klasifikátoru,klasifikátor, váha)
        """
        
        classifiers=[]
        
        for dataName, classifierName, w, t in classifiersNames:
            classifiers.append((dataName, classifierName, Pipeline([
                        ('dataSel', DataTypeSelector(dataName)),
                        ('cls', self.__makeClassifier(classifierName, classifiersParams))
                    ]),w,t ))
                    
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
            return CCWrapper(SGDClassifier(**classifierParams[self.SGDClassifierName]))
        elif clsName==self.matchTargetClassifierName:
            return MatchTargetClassifier(**classifierParams[self.matchTargetClassifierName])
        elif clsName==self.KMeansClassifierName:
            return KMeansClassifier()
            
    
    @staticmethod
    def filterMarkedDataWithTargets(data, targets):
        """
        Odfiltruje označená data (None) a jejich cíle.
        
        :type data: FeaturesContainer
        :param data: Data pro filtraci.
        :param targets: Cíle k datům
        :returns: odfiltrovaná ( data, cíle)
        """

        return (data.features, [t for d,t  in zip(data,targets) if d is not None])
    
    def __controlMulPErrors(self):
        """
        Kontrola chyb z ostatních procesů.
        """
        if self.errorBoard and not self.errorBoard.empty():
            print("Vznikla chyba. Ukončuji všechny procesy.", file=sys.stderr)
            for p in active_children():
                p.terminate()
            exit()
            
    def __getCls(self, dataName=None, classifierName=None, weight=None, threshold=None):
        """
        Získání klasifikátorú splňujících určité vlastnosti.
        
        :param dataName: Název dat.
        :param classifierName: Název klasifikátoru.
        :param weight: Váha.
        :param threshold: Práh.
        :return: list -- obsahující všechny klasifikátory, které vyhovují zadaným parametrům. Pokud parametr má None, tak se neuvažuje.
        """
        searched=[]
        for actDataName, actClassifierName, actClassifier, actWeight, actThreshold in self.__classifiers:
            if dataName is not None and dataName!=actDataName:
                continue
            if classifierName is not None and classifierName!=actClassifierName:
                continue
            if weight is not None and weight!=actWeight:
                continue
            if threshold is not None and threshold!=actThreshold:
                continue
            
            searched.append(actClassifier)
        return searched
    
class PredictWorker(Process):
    """
    Třída reprezentující jeden pracující proces provádějící predikci.
    """
    
    class PredictedStorage(SharedStorage):
        """
        Synchronizované uložiště predikcí.
        """
        
        def __init__(self, manager=None):
            """
            Inicializace uložiště.
            
            :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
            """
            if manager is None:
                manager=Manager()
            super().__init__(SharedStorage.StorageType.DICT, manager)
            
        def addResult(self, partNumber, pred):
            """
            Přidá predikce z části, která má pořadové číslo partNumber do uložiště.
            
            :param partNumber: Pořadové číslo části.
            :param pred: Predikce
            """

            self._safeAcquire()
            try:
                
                self._storage[partNumber]=pred
                self._numOfData.value+=pred.shape[0] if hasattr(pred,"shape") else len(pred)
            except:
                raise
            finally:
                self._safeRelease()
                
            self._notifyChange()
                
        def popResults(self):
            """
            Vyjme všechny predikce z uložiště a vráti je v seřazené podobě, dle pořadových čísel části.
            
            :rtype: array-like | sparse matrix | None
            :return: Predikce. None -> prázdno
            """
            
            self._safeAcquire()
                
            try:

                res=None
                
                if len(self._storage)>0:
                    for partNum in range(min(self._storage.keys()), max(self._storage.keys())+1):
                        #Projedeme od nejmenšího po největší index a tím získáme seřazenou posloupnost.
                        
                        if partNum in self._storage:
                            try:
                                res=self._storage[partNum] if res is None else np.vstack((res, self._storage[partNum]))

                                #Odstraníme z uložiště, protože děláme pop.
                                del self._storage[partNum]
                            except :
                                print("partnum", partNum, file=sys.stderr)
                                if res is not None:
                                    print("res shape", res.shape[0], res.shape[1] , file=sys.stderr)
                                if self._storage[partNum] is not None:
                                    print("part shape", self._storage[partNum].shape[0], self._storage[partNum].shape[1], file=sys.stderr)
                                print("type of part", type(self._storage[partNum]), file=sys.stderr)
                                print("part", self._storage[partNum], file=sys.stderr)
                                raise
                            
                    #nastavíme počítadlo
                    self._numOfData.value=0

                return res
            except:
                raise
            finally:
                self._safeRelease()
                
            self._notifyChange()
            
    
    def __init__(self, classifier, inputDataQueue, resultsStorage, errorBoard):
        """
        Inicializace procesu.
        
        :param classifier: Clasifikátor, který bude použit k predikci.
        :type inputDataQueue: Queue
        :param inputDataQueue: Z této řady přímá data k predikci.
                        Jedem záznam ve frontě je n-tice:
                            (partNumber, data)
        :type resultsStorage: PredictedStorage
        :param resultsStorage: Zde ukládá výsledky predikce.
        :type errorBoard: Queue
        :param errorBoard: Oznámení o chybách.
        """
        
        super().__init__()
        
        self.__classifier=classifier
        self.__inputDataQueue=inputDataQueue
        self.__resultsStorage=resultsStorage
        self.__errorBoard=errorBoard
    
    
    def predict(self, classifier, data):
        """
        Provedení predikce.
        
        :param classifier: Klasifikátor pro predikci.
        :param data: Data, která chceme klasifikovat.    
        :return: Výsledky klasifikace.
        """
        return classifier.predict(data)
    
        
    def run(self, once=False):
        """
        Čekání na vstupní data a predikce.
        
        :param once: Pokud je True. Zpracuje pávě jeden blok, pokud není ihned k dispozici, tak končí.
        """
        
        try:
            while True:
                try:
                    msg=self.__inputDataQueue.get(timeout=1 if once else None)

                except queue.Empty:
                    return
                else:
                    if msg == "EOF":
                        return "EOF"
                    partNumber, data =msg
                    
                    self.__resultsStorage.addResult(partNumber, self.predict(self.__classifier, data))
                    
                gc.collect()
                if once:
                    return
        except:
            self.__errorBoard.put("ERROR")
            print(traceback.format_exc(), file=sys.stderr)
            
            
class PredictProbaWorker(PredictWorker):
    """
    Třída reprezentující jeden pracující proces provádějící predikci s výsledky v podobě pravděpodobností.
    """
    
    
    def __init__(self, classifier, targets, globalProbaIndexesMissing, weights, clsThreshold, inputDataQueue, resultsStorage, errorBoard):
        """
        Inicializace procesu.
        
        :param classifier: Clasifikátor, který bude použit k predikci.
        
        :type targets: list 
        :param targets: Obsahuje všechny možné predikovatelné cíle/kategorie. Ne pouze množinu cílů/kategorií, které
            má naučený aktuální klasifikátor.
        
        :type globalProbaIndexesMissing: list 
        :param globalProbaIndexesMissing: Obsahuje indexy, které se mají doplnit do výsledku predikce.
            Na těchto místech budou doplněny nuly.
        
        :type weights: list 
        :param weights: Váhy klasifikátoru pro jednotlivé kategorie/cíle.
        
        :type clsThreshold: float 
        :param clsThreshold: Udává minimální míru jistoty, která je třeba k uznání predikce. Pokud je menší je dokument neklasifikován.
        
        :type inputDataQueue: Queue
        :param inputDataQueue: Z této řady přímá data k predikci.
                        Jedem záznam ve frontě je n-tice:
                            (partNumber, data)
                            
        :type resultsStorage: PredictedStorage
        :param resultsStorage: Zde ukládá výsledky predikce.
                            
        :type errorBoard: Queue
        :param errorBoard: Oznámení o chybách.
        """
        
        super().__init__(classifier, inputDataQueue, resultsStorage, errorBoard)
        
        self.__targets=targets
        self.__globalProbaIndexesMissing=globalProbaIndexesMissing
        self.__weights=weights
        self.__clsThreshold=clsThreshold
        
    
    def predict(self, classifier, data):
        """
        Provedení predikce s výsledky v podobě pravděpodobností.
        
        :param classifier: Klasifikátor pro predikci.
        :param data: Data, která chceme klasifikovat.    
        :return: Výsledky klasifikace.
        """
        
        predicted=[]
        for doc in classifier.predict_proba(data):
            
            if len(self.__globalProbaIndexesMissing)>0:
                #vložíme 0 na chybějící třídy/kategorie, které nemá klasifikátor natrénované
    
                alreadyFiledCnt=0
                docFilled=[]
                for x in range(len(self.__targets)):
                    if x in self.__globalProbaIndexesMissing:
                        docFilled.append(0)
                        alreadyFiledCnt+=1
                    else:
                        docFilled.append(doc[x-alreadyFiledCnt])
                doc=docFilled
            

            for i, actPredicted in enumerate(doc):
                doc[i]=actPredicted*self.__weights[i] if actPredicted>=self.__clsThreshold else 0

            predicted.append(doc)
            
            
        
        return np.array(predicted)
        
            
class TrainWorker(Process):
    """
    Třída reprezentující jeden pracující proces provádějící trénování klasifikátoru.
    """
    
    class ClassifiersStorage(SharedStorage):
        """
        Uložiště natrénovaných klasifikátorů. Synchronizované mezi procesy.
        """
        
        def __init__(self, manager=None):
            """
            Inicializace uložiště.
            
            :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
            """
            if manager is None:
                manager=Manager()
            super().__init__(SharedStorage.StorageType.LIST, manager)

        
        def addResult(self, dataName, classifierName, classifier, w, t):
            """
            Přidání klasifikátoru.
            
            :param dataName: Název dat pro trénování. Používá se pro logování.
            :param classifierName: Název klasifikátoru. Používá se pro logování.
            :param classifier: Klasifikátor, který má být natrénován.
            :param w: Váha klasifikátoru. Používá se pro logování.
            :param t: Práh klasifikátoru. Používá se pro logování.
            """

            self._safeAcquire()
                
            try:
                self._storage.append((dataName, classifierName, classifier, w, t))
                self._numOfData.value+=1
            except:
                raise
            finally:
                self._safeRelease()
                
            self._notifyChange()
                
        def popResults(self):
            """
            Vyjme všechny klasifikátory z uložiště a vrátí je.
            
            :rtype: list | None
            :return: Klasifikátory.. None -> prázdno.
            """
            
            self._safeAcquire()
            try:
                res=None
                
                if len(self._storage)>0:
                    res=[]
                    
                    for clsInfo in self._storage:
                        res.append(clsInfo)
                            
                    self._storage.clear()
                    
                    #nastavíme počítadlo
                    self._numOfData.value=0

                return res
            except:
                raise
            finally:
                self._safeRelease()
                
            self._notifyChange()
    
    def __init__(self, inputDataQueue, resultsStorage, errorBoard, sharedLock, data):
        """
        Inicializace procesu.
        
        :param inputDataQueue: Z této řady přímá klasifikátory a data k jejich trénování.
                        Jeden záznam ve frontě je n-tice:
                            (targets, data, dataName, classifierName, classifier, weight, threshold)
                            
        :type resultsStorage: ClassifiersStorage
        :param resultsStorage: Zde ukládá klasifikátory.
        :param errorBoard: Oznámení o chybách.
        :param sharedLock: Sdílený zámek. Používá se pro výpis logů.
        :param data: All data that can be used for training.
        """
        
        super().__init__()
        
        self.__inputDataQueue=inputDataQueue
        self.__resultsStorage=resultsStorage
        self.__errorBoard=errorBoard
        self.__sharedLock=sharedLock
        self.__data=data
        
    @staticmethod
    def trainCls(targets, data, dataName, classifierName, classifier, w, t, sharedLock=None):
        """
        Natrénuje klasifikátor
        
        :param targets: Cíle pro trénování
        :param data: Data pro trénováni (konkrétní druh dat).
        :param dataName: Název dat pro trénování. Používá se pro logování.
        :param classifierName: Název klasifikátoru. Používá se pro logování.
        :param classifier: Klasifikátor, který má být natrénován.
        :param w: Váha klasifikátoru. Používá se pro logování.
        :param t: Práh klasifikátoru. Používá se pro logování.
        :param sharedLock: Sdílený zámek. Používá se pro logování. Pokud nepoužíváme více procesorovou variantu lze jej vynechat.
        """
        acquired=False
        try:
            if sharedLock is not None:
                sharedLock.acquire()
                acquired=True
                
            logging.info("začátek trénování klasifikátoru "+ classifierName+" pro "+dataName+" s váhou "+str(w)+" a prahem "+str(t))
            
            if sharedLock is not None:
                sharedLock.release()
                acquired=False
                        
            #odfiltrujeme označené dokumenty
            actData, actTargets =Classification.filterMarkedDataWithTargets(data, targets)
                
            actData={dataName:actData}
                
            classifier.fit(actData, actTargets)
                        
            if sharedLock is not None:
                sharedLock.acquire()
                acquired=True
                
            logging.info("konec trénování klasifikátoru "+ classifierName+" pro "+dataName+" s váhou "+str(w)+" a prahem "+str(t))
            logging.info("\tNatrénovaných cílů: "+ str(len(classifier.classes_)))
            logging.info("\tNatrénováno na "+str(actData[dataName].shape[0])+" dokumentech.")
            
            if sharedLock is not None:
                sharedLock.release()
                acquired=False
        except:
            if acquired:
                sharedLock.release()
                
            raise

    def run(self, once=False, timeoutInQueue=1):
        """
        Čekání na vstupní data a trénování klasifikátoru.
        
        :param once: Pokud je True. Zpracuje pávě jeden blok, pokud není ihned k dispozici, tak končí.
        :param timeoutInQueue: Bere se v úvahu pouze pokud je parametr once=true. Nastavuje timeout pro čekání ve frontě na vstupní data.
        :return: "EOF"|None
        """
        try:
            while True:
                try:
                    
                    msg=self.__inputDataQueue.get(timeout=timeoutInQueue if once else None)

                except queue.Empty:
                    return
                else:
                    if msg == "EOF":
                        return "EOF"
                    targets, dataName, classifierName, classifier, w, t =msg
                    self.trainCls(targets, self.__data[dataName], dataName, classifierName, classifier, w, t, self.__sharedLock)
                    
                    
                    self.__resultsStorage.addResult(dataName, classifierName, classifier, w, t)
                    
                    
                gc.collect()
                if once:
                    return
        except:
            self.__errorBoard.put("ERROR")
            print(traceback.format_exc(), file=sys.stderr)
        
class AutoWeightWorker(Process):
    """
    Třída reprezentující jeden pracující proces provádějící křížově validační krok pro získání vahy klasifikátoru.
    """
    
    class WeightStorage(SharedStorage):
        """
        Uložiště vah. Ukládá váhy všech paralelně zpracovávaných klasifikátorů
        """
        
        def __init__(self, manager=None):
            """
            Inicializace uložiště.
            
            :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
            """
            if manager is None:
                manager=Manager()
            super().__init__(SharedStorage.StorageType.DICT_SIMPLE, manager)
            
        def __len__(self):
            """
            Zjištení počtu všech uložených výsledků. Jedná se o souhrný počet
            pro všechny klasifikátory.
            
            :return: Počet všech uložených výsledků 
                (jako jeden výsledek jsou počítány všechny úspěšnosti předané v addResultsFor jako celek).
            :rtype: int
            """

            self._safeAcquire()
                
            try:
                return sum( len(x) for x in self._storage.values())
            except:
                raise
            finally:
                self._safeRelease()
                
        
        def addResultsFor(self, clsId, res):
            """
            Přidání výsledků ke klasifikátoru.
            
            :param clsId: Identifikátor klasifikátoru.
            :param res: Výsledky uspěšnosti v jednotlivých kategoriích.
            """
            self._safeAcquire()
            try:
                if clsId not in self._storage:
                    self._storage[clsId]=[]
                    
                #kvůli manageru nelze pouzit append :(
                self._storage[clsId]=self._storage[clsId]+[res]

            except:
                raise
            finally:
                self._safeRelease()
                
            #oznámíme, že došlo ke změně připadným čekajícím
            self._notifyChange()
            
        def getResultsFor(self, clsId):    
            """
            Vrátí výsledky pro zadaný klasifikátor.
            
            :param clsId: Identifikátor klasifikátoru.
            :return: Uložený výsledky.
            :rtype: list
            """
            return self._storage[clsId]
            
            
        def getAvgResultsFor(self, clsId):
            """
            Vrátí zprůměrované výsledky pro zadaný klasifikátor.
            
            :param clsId: Identifikátor klasifikátoru.
            :return: Zprůměrované výsledky uspěšností v jednotlivých kategoriích.
            :rtype: list
            """
            clsData=self.getResultsFor(clsId)
            avgData={}
            #prvně uděláme sumu
            for res in clsData:
                for i, r in enumerate(res):
                    if i not in avgData:
                        avgData[i]=0
                    avgData[i]+=r
                
            
            
            return [avgData[i]/len(clsData) for i in range(len(avgData))]
                
            
        
    
    def __init__(self, inputDataQueue, storage, errorBoard, sharedLock):
        """
        Inicializace procesu.
        
        :param inputDataQueue: Z této řady přímá klasifikátory a data k jejich trénování a testování.
                        Jeden záznam ve frontě je n-tice:
                            (targets, data, dataName, classifierName, classifier, weight, threshold, testTargets, testData, clsId)
        :type storage: WeightStorage
        :param storage: Zde bude přidávat výsledky testování
        :param errorBoard: Oznámení o chybách.
        :param sharedLock: Sdílený zámek. Používá se pro výpis logů.
        """
        
        super().__init__()
        
        self.__inputDataQueue=inputDataQueue
        self._storage=storage
        self.__errorBoard=errorBoard
        self.__sharedLock=sharedLock
        
    def run(self, once=False):
        """
        Čekání na vstupní data a získávání vah.
        
        :param once: Pokud je True. Zpracuje pávě jeden blok, pokud není ihned k dispozici, tak končí.
        """
        try:
            while True:
                try:
                    msg=self.__inputDataQueue.get(timeout=1 if once else None)

                except queue.Empty:
                    return
                else:
                    if msg == "EOF":
                        return "EOF"
                    
                    #zprava
                    targets, data, dataName, classifierName, classifier, w, t, testTargets, testData, clsId =msg
                    
                    #natrenovani klasifikatoru
                    TrainWorker.trainCls(targets, data, dataName, classifierName, classifier, w, t, self.__sharedLock)
                    
                    #zjisteni a ulozeni uspesnosti
                    _, _, f1, _ = precision_recall_fscore_support(testTargets, classifier.predict({dataName:testData}))
                    self._storage.addResultsFor(clsId, f1)
                    
                gc.collect()
                if once:
                    return

        except:
            self.__errorBoard.put("ERROR")
            print(traceback.format_exc(), file=sys.stderr)
        

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
        self.cls=None

    def __getstate__(self):
        d = dict(self.__dict__)
        if 'classes_' in d:
            del d['classes_']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
    
    @property
    def classes_(self):
        """
        Natrénované kategorie/cíle.
        """
        
        return None if self.cls is None else self.cls.classes_
    
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
    
class ConfusionClassifierWrapper(object):
    """
    Upravuje výsledky klasifikace na základě matice záměn, která je získávána při trénování klasifikátor.
    Trénovací data jsou rozdělena v jednotlivých krocích na testovací a novou trénovací množinu.
    Na nové trénovací množině je klasifikátor natrénován a na testovací množině je otestován.
    Matice může mít zprůměrované hodnoty z jednotlivých kroků křížové validace.
    """
    
    numOfSteps=2
    cv=3
    
    def __init__(self, classifier):
        """
        Inicializace.
        
        :param classifier: Klasifikátor, který chceme obalit.
        """
        self.__classifier=classifier
        
        #Zde budou uloženy matice, která budou určovat s jakým podílem 
        #patří dokument, do kterých kategorií, pokud je predikována určená kategorie.
        #Příklad uveden v metodě fit.
        self.__belongsToIndexes=[]
    
    def __getstate__(self):
        d = dict(self.__dict__)
        if 'classes_' in d:
            del d['classes_']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
        
    @property
    def classes_(self):
        """
        Natrénované kategorie/cíle.
        """
        
        return self.__classifier.classes_
    
    def __learnIndexes(self, cMat):
        """
        Získání indexů, které přerozdělují výslednou jistotu získanou z obaleného klasifikátorů,
        do dalších kategoriíí/cílů na základě matice záměn.
        
        :type cMat: array-like 
        :param cMat: Matice záměn.
        """

        #Zjistíme kolikrát byl predikovaný, který cíl/kategorie
        predCnt=cMat.sum(axis=0)
        
        #Projdeme matici a vytvoříme pro každý predikovaný cíl hodnotu, která udává podíl
        #s jakým patří daný cíl do jiných cílů.
        #Příklad:
        #               predikované
        #    pravé    A    B    C
        #        A    7    1    0
        #        B    3    9    0
        #        C    0    0    10
        #
        #   predCnt   10   10   10
        #
        #Převedeme na:
        #               predikované
        #    pravé    A      B      C
        #        A    0.7    0.1    0.0
        #        B    0.3    0.9    0.0
        #        C    0.0    0.0    1
        #
        #Tedy například pro A. Tuto kategorii jsme predikovali celkem 10x. Z toho se ve skutečnosti jednalo 7x o A a 3x o B.
        #Řekneme tedy, že predikování kategorie A znamená, že dokument patří do A s 0.7 a do B s 0.3. 
        for ri, row in enumerate(cMat):
            for ci, col in enumerate(row):
                #pokud nebyla kategorie ani jednou predikována nastavíme 0
                #jinak podíl
                cMat[ri][ci]=col/predCnt[ci] if predCnt[ci]!=0 else 0

        
        #kvůli pozdějšímu procházení matici ještě transponujeme
        #Tedy dle uvedeného příkladu dostaneme:
        #               pravé
        # predikované  A      B      C
        #        A    0.7    0.3    0.0
        #        B    0.1    0.9    0.0
        #        C    0.0    0.0    1
                   
        self.__belongsToIndexes.append(cMat.T)
 
    def fit(self, X, y, sampleWeight=None):
        """
        Natrénuje klasifikátor.
        
        :param X: array-like, sparse matrix -- trénovací vektory [n_vektorů, n_příznaků]
        :param y: array-like, [n_samples] cíle k trénovacím vektorům
        :param sampleWeight: array-like [n_samples] váhy k trénovacím vektorům. Implicitně jednotková.
        """
        self.__belongsToIndexes=[]


        
        #výsledky predikce z jednotlivých křížově validačních kroků
        cvPred=[]
        
        cvY=[]
        
        #zde vypočítáme průměrnou matici
        cMat=None
        
        acc=0
        
        numpyY=np.array(y)
        #křížová validace
        for trainIndex, testIndex in StratifiedKFold(n_splits=self.cv, random_state=0).split(X,y):
            trainData, trainTargets=X[trainIndex,:], numpyY[trainIndex]
            testData, testTargets=X[testIndex,:], numpyY[testIndex]
            
            self.__classifier.fit(trainData, trainTargets, sampleWeight)
            
            cvPred.append(self.__classifier.predict_proba(testData))
            cvY.append(testTargets)
            
            cvPredictedY=[self.classes_[docProba.argmax(axis=0)] for docProba in cvPred[-1]]
            
            m=confusion_matrix(testTargets, cvPredictedY).astype(float)
            if cMat is None:
                cMat=m
            else:
                cMat+=m
                
            acc+=accuracy_score(testTargets, cvPredictedY) 
            
        
        cMat=cMat/float(self.cv)
        
        acc=acc/float(self.cv)
        
        #Dvojice obsahuje správnost a index do allMat, kde se nachází matice, kdy došlo k maximu.
        #None značí, že nejlépe vychází varianta bez zahrnutí indexů.
        
        maxAcc=(acc, None)
        
        steps=self.numOfSteps
        
        i=0
        while steps>0 and maxAcc[0]<1:
            #naučíme se nové indexy a zjistíme jak změnily zprávnost

            self.__learnIndexes(cMat)

            cMat=None
            acc=0
            for cvi, (predProb, testY) in enumerate(zip(cvPred, cvY)):
                #predictedY=self.__predict(None, predProb)
                
                mNew=self.__applyIndexMatric(predProb, self.__belongsToIndexes[-1])  
                predictedY=[self.classes_[docProba.argmax(axis=0)] for docProba in mNew]

                cvPred[cvi]=mNew
                
                
                                
                m=confusion_matrix(testY, predictedY).astype(float)
                if cMat is None:
                    cMat=m
                else:
                    cMat+=m
                acc+=accuracy_score(testY, predictedY) 
                       
            cMat=cMat/float(self.cv)
            acc=acc/float(self.cv)
            
            if acc>maxAcc[0]:
                maxAcc=(acc, i)
                steps=self.numOfSteps
            else:
                steps-=1


            i+=1
                
        if maxAcc[1] is None:
            self.__belongsToIndexes=[]
        else:
            #Odsraníme zbytečné matice
            while len(self.__belongsToIndexes)>(maxAcc[1]+1):
                self.__belongsToIndexes.pop()


        #natrénujeme klasifikátor
        self.__classifier.fit(X, y, sampleWeight)

    
    def __applyIndexMatric(self, predProba, indexMat):
        """
        Upraví pravděpodobnosti z predProba aplikováním indexMat.
        
        :param predProba: Pravděpodobnosti ke každému cíli a to pro každý vektor z X. Pořádí cílů lze zjistit pomocí classes _.
        :param indexMat: Matice udávající s jakým podílem patří dokument, do kterých kategorií, pokud je predikována určená kategorie.
        :returns: Pravděpodobnosti ke každému cíli a to pro každý vektor z X. Pořádí cílů lze zjistit pomocí classes _.
        """
        
        predProbRes=[[0]*predProba.shape[1] for _ in range(predProba.shape[0])]
        
        #optimalizujeme procházení, tak že vybereme jen některé sloupce, které se mají procházet
        #    odfiltrujeme nulové (blízké nule) položky z matice
        useColumns=[]
        for belongsToRow in indexMat:
            useColumns.append([ci for ci, x in enumerate(belongsToRow) if not np.isclose(x, 0) ])

        for proba, predProbResRow in zip(predProba,predProbRes):
            for bi, belongsToRow in enumerate(indexMat):
                cell=proba[bi]
                if not np.isclose(cell, 0):
                    for ci in useColumns[bi]: 
                        predProbResRow[ci]+=cell*belongsToRow[ci]

        return np.array(predProbRes)
        

    def __predict_proba(self, X, predYProb=None):
        """
        Získání pravděpodobností predikce k jednotlivým cílům.
        
        :param X: array-like, sparse matrix -- vektory pro predikci [n_vektorů, n_příznaků]
        :param predYProb: Před predikované pravděpodobnosti. Vynechá predikci a aplikuje pouze indexy.
        :returns: Pravděpodobnosti ke každému cíli a to pro každý vektor z X. Pořádí cílů lze zjistit pomocí classes _.
        """

        predYProb=self.__classifier.predict_proba(X) if predYProb is None else predYProb
              
        
        #Projdeme všechny predikce a upravíme příslušnosti do jednotlivých kategorií
        #dle natrénovaných matic v __belongsToIndexes.
        
        for indexMat in self.__belongsToIndexes:
            predYProb=self.__applyIndexMatric(predYProb, indexMat)
                    
        return predYProb
    
    def __predict(self, X, predYProb=None):
        """
        Získání pravděpodobností predikce k jednotlivým cílům.
        
        :param X: array-like, sparse matrix -- vektory pro predikci [n_vektorů, n_příznaků]
        :param predYProb: Před predikované pravděpodobnosti. Vynechá predikci a aplikuje pouze indexy.
        :returns: Cíle pro data.
        """
        
        predictedLabels=[]
        
        for docProba in self.__predict_proba(X, predYProb):
            predictedLabels.append(self.classes_[docProba.argmax(axis=0)])
        
        return predictedLabels
    
    def predict_proba(self, X):
        """
        Získání pravděpodobností predikce k jednotlivým cílům.
        
        :param X: array-like, sparse matrix -- vektory pro predikci [n_vektorů, n_příznaků]
        :returns: Pravděpodobnosti ke každému cíli a to pro každý vektor z X. Pořádí cílů lze zjistit pomocí classes _.
        """
        return self.__predict_proba(X)
    
    
    def predict(self, X):
        """
        Predikuje cíle pro daná data
        
        :param X: array-like, sparse matrix -- vektory pro predikci [n_vektorů, n_příznaků]
        :returns: Cíle pro data.
        """
        return self.__predict(X)
    
    