# -*- coding: UTF-8 -*-
"""
Obsahuje třídu pro extrakci příznaků a další potřebné nástroje.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer

from .Vectorizers import D2VVectorizer, MatchTargetVectorizer
from CPKclassifierPack.utils.DocReader import DocReaderDataString
from CPKclassifierPack.utils.DataSet import DataTypeSelector

import functools
import logging
    
import math
import gc

from scipy.sparse import vstack

from multiprocessing import Process, cpu_count, active_children, Lock, Manager

import queue
import sys
import traceback


from CPKclassifierPack.utils.Parallel import SharedStorage


class FeaturesNoData(Exception):
    """
    Nebyly poskytnuty data.
    """
    pass

class Features(object):
    """
    Třída pro extrakci příznaků.
    """

 
    #Názvy nástrojů pro vektorizaci.
    vectorizersNames=["tfidfvectorizer", "doc2vec", "countvectorizer", "hashingvectorizer", "matchtargetvectorizer"]
    
    countVectorizerName="countvectorizer"
    doc2VecName="doc2vec"
    tfidfVectorizerName="tfidfvectorizer"
    hashingVectorizerName="hashingvectorizer"
    matchTargetVectorizer="matchtargetvectorizer"
        
    #Názvy analyzátorů pro fulltext. První je používán jako defaultní. Vše malé znaky.
    fulltextAnalyzersNames=["ngram"]
    
    #Názvy analyzátorů pro metadata. První je používán jako defaultní. Vše malé znaky.
    metaAnalyzersNames=["ngram", "wholeitem"]
    
    ngramName="ngram"
    wholeitemName="wholeitem"
    
    #U těchto extraktorů se nepoužívá fit (fir je prázdná operace).
    #Aplikuje pouze transform na částech datasetu. (Takto můžeme vypisovat postup).
    #Je zde pouze hashingVectorizer (zatím?).
    noFit=[hashingVectorizerName]
    
    #Zde je nutné doplnit všechny vektorizátory, které používají ke své prácí cíle/kategorie.
    useTargets=set(["matchtargetvectorizer"])
    
    MAX_WAIT_TIMEOUT=10

    def __init__(self, getFulltext, getMetaFields, fullTextVectorizer, fullTextAnalyzer,
                 metaVectorizers, metaAnalyzers, hashingVectorizer, doc2Vec, lemmatizer, fulltextName="fulltext", markEmpty=True):
        """
        Konstruktor. Připraví nástroje pro extrakci příznaků.
        
        :param __getFulltext: bool -- True zahrne fulltext. False nezahrne.
        :param __getMetaFields: list -- Názvy metadatových polí, které se mají zahrnout do extrakce příznaků. 
        :param fullTextVectorizer: název nástroje pro vektorizaci plného textu
        :param fullTextAnalyzer: (název analyzátor pro plný text, parametr)
        :param metaVectorizers: dict -- klíč název metadat, hondnota je název nástroje pro vektorizaci
        :param metaAnalyzers: dict -- klíč název metadat, hondnota je dvojice (analyzátor, parametr)
        :param hashingVectorizer: dict -- s nastavením pro HashingVectorizer
            {
                "NON_NEGATIVE":,
                "N_FEATURES":
            }
        :param doc2Vec: dict -- s nastavením pro Doc2Vec
            {
                "SIZE":,
                "ALPHA":,
                "WINDOW":,
                "MIN_COUNT":,
                "WORKERS":,
                "ITER":,
                "SAMPLE":,
                "DM":,
                "NEGATIVE":
            }
        :type lemmatizer: Lemmatizer
        :param lemmatizer: Lemmatizuje. 
        :param __fulltextName: Název dat s plným textem
        :param markEmpty: Pokud true, označí/extrahuje na None prázdná data.
        """
        self.__getFulltext=getFulltext
        self.__getMetaFields=getMetaFields
        self.fullTextVectorizer=fullTextVectorizer
        self.fullTextAnalyzer=fullTextAnalyzer
        self.metaVectorizers=metaVectorizers
        self.metaAnalyzers=metaAnalyzers
        self.hashingVectorizer=hashingVectorizer
        self.doc2Vec=doc2Vec
        self.lemmatizer=lemmatizer
        self.__fulltextName=fulltextName
        self.markEmpty=markEmpty

        self.__transformers=self.__makeTransformerDict()
     
        
        
        self.errorBoard=None
        
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
    
    def __transformersNeedFit(self):
        """
        Získání transformerů, které potřebují fázi fit.
        
        :return: dict- s tranformery, které potřebují fázi fit. Klíč je název dat. Hodnota je transformer.
        """
        needFit={}
        
        allVectorizers={self.__fulltextName:self.fullTextVectorizer}
        if self.metaVectorizers:
            allVectorizers.update(self.metaVectorizers)
        
        for dataName, transformer in self.__transformers.items():
            if allVectorizers[dataName] not in self.noFit:
                needFit[dataName]=transformer
                    
        return needFit

    def flush(self):
        """
        Vhodné pro uvolnění paměti.
        Použít v případě, kdy se už objekt nebude používat.
        """
        del self.__transformers
        
    def learnVocabularies(self, data):
        """
        Naučí se z poskytnutých dat slovníky pro dané vectorizery.
        
        :type data: dict 
        :param data: Klíč název dat. Hodnota data, ze kterých se získá slovník.
        """
        
        for dataName, dataVal in data.items():
            logging.info("začátek vytváření slovníku pro: "+dataName)
            vectorizer=self.getVectorizer(dataName)
            
            if isinstance(vectorizer, D2VVectorizer):
                vectorizer.buildVocab(dataVal)
            else:
                vocDic=self._makeVocabularyDict(dataVal)
                logging.disable(logging.INFO)
                self.__transformers[dataName]=Pipeline([
                        ('dataSel', DataTypeSelector(dataName)),
                        ('vect', self.__makeVectorizer(dataName, vocDic))
                    ])
                logging.disable(logging.NOTSET)
                
                logging.info("\tvelikost slovníku: "+str(len(vocDic)))
                
            logging.info("konec vytváření slovníku pro: "+dataName)
            
    def _makeVocabularyDict(self, data):
        """
        Vytvoří z dat slovník.
        
        :param data: Data, ze kterých bude získán slovník.
        :rtype: dict
        :return: Klíč term. Hodnota je index do příznakového vektoru.
        """
        
        voca={}
        for examp in data:
            try:
                for word in examp:
                    if word not in voca:
                        voca[word]=len(voca)
            except TypeError:
                pass
            
        
        return voca
        
    def extractAndLearn(self, data, targets=None, splitIntoPartsOfMaxSize=None, workers=1):
        """
        Učí model a extrahuje příznaky z dat. Poskytuje také možnost zobrazení postupu pro extraktory v noFit.
        
        :param data: Pro extrakci příznaků a učení nástroje pro extrakci příznaků.
        :param targets: Cíle dat.
        :param splitIntoPartsOfMaxSize: Tento parametr určuje maximální velikost části dat, která se bude v jednom kroku extrahovat.
            Pokud je uvedeno, bude vypisovat pro vektorizery v noFit postup.
        :param workers: Počet pracujicích procesů. Určuje kolik modelů bude zároveň trénováno/(provádět extrakci).
        :returns: dict -- s extrahovanámi příznaky Klíč je název dat.
        """
        workers=self.__manageWorkers(workers)

        trans={}    #zde uložíme extrahovaná data
        allVectorizers={self.__fulltextName:self.fullTextVectorizer}
        if self.metaVectorizers:
            allVectorizers.update(self.metaVectorizers)
                
        if workers>1:
            #více procesová varianta

            #Nejprve zpracujeme ty, které je nutné neprve natrénovat.
            #Pro urychlení použijeme fit_transform variantu.
            self.learn(data, targets, workers, extracted=trans)
            
            #Zjistíme jaké modely není nutné natrénovat.
            transformers={dataName: transformer for dataName, transformer in self.__transformers.items() if allVectorizers[dataName] in self.noFit}
            
            #S modely, které není nutné trénovat spustíme extrakci.
            trans.update(self.__extract(data, splitIntoPartsOfMaxSize, transformers, workers))
            
            return trans
        
        for dataName, transformer in self.__transformers.items():
            
            
            emptyIndexes=[]

            if self.markEmpty:
                #Schováme si indexy dat, které pozdějí označíme za prázdná.
                emptyIndexes, actData, actTargets=self.filterMarked(data[dataName], targets)
            else:
                actData=data[dataName]
                actTargets=targets

            
            if splitIntoPartsOfMaxSize and allVectorizers[dataName] in self.noFit:
                if self.markEmpty:
                    logging.info("Počet neprázdných dokumentů pro "+dataName+": "+str(len(actData)))
                    
                logging.info("začátek učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků pro data "+dataName)
                
                    
                transformer.fit({dataName:actData}, actTargets)
                logging.info("konec učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků pro data "+dataName)
                logging.info("začátek extrakce příznaků pomocí "+allVectorizers[dataName]+" pro "+dataName)

                    
                docNum=len(actData)
                for i in range(math.ceil(docNum/splitIntoPartsOfMaxSize)):
                    endOfPart=(i+1)*splitIntoPartsOfMaxSize if (i+1)*splitIntoPartsOfMaxSize<docNum else docNum
                    newS=transformer.transform({dataName:actData[i*splitIntoPartsOfMaxSize:endOfPart]})
                    if dataName not in trans:
                        trans[dataName]=newS
                    else:
                        trans[dataName]=vstack((trans[dataName], newS))
                
                    logging.info("\tHotovo: "+str(int(100*trans[dataName].shape[0]/docNum))+"% - "+str(trans[dataName].shape[0])+"/"+str(docNum))
                
                logging.info("konec extrakce příznaků pomocí "+allVectorizers[dataName]+" pro "+dataName)
            else:
                if self.markEmpty:
                    logging.info("Počet neprázdných dokumentů pro "+dataName+": "+str(len(actData)))
                logging.info("začátek učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků a samotná extrakce příznaků pro data "+dataName)
                trans[dataName]=transformer.fit_transform({dataName:actData}, actTargets)
                logging.info("konec učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků a samotná extrakce příznaků pro data "+dataName)
                    
            trans[dataName]=FeaturesContainer(trans[dataName], emptyIndexes)
            
        return trans
    
    def __extract(self, data, splitIntoPartsOfMaxSize, transformers, workers=1):
        """
        Extrahuje příznaky z dat.

        :param data: Pro extrakci příznaků.
        :param splitIntoPartsOfMaxSize: Tento parametr určuje maximální velikost části dat, která se bude v jednom kroku extrahovat.
            Pokud je uvedeno, bude vypisovat pro vektorizery v noFit postup.
        :type transformers: dict
        :param transformers: Knihovna, kde klíč je název dat a hodnota, je příslušný transformer.
                    (tedy něco jako self.__transformers)
        :param workers: Počet pracujicích procesů. Určuje kolik modelů bude zároveň trénováno.
        :returns: dict -- s extrahovanými příznaky Klíč je název dat.
        """
        
        trans={}
        allVectorizers={self.__fulltextName:self.fullTextVectorizer}
        if self.metaVectorizers:
            allVectorizers.update(self.metaVectorizers)
            
        workers=self.__manageWorkers(workers)
        
        for dataName, transformer in transformers.items():
            logging.info("začátek extrakce příznaků pomocí "+allVectorizers[dataName]+" pro "+dataName)
            
            emptyIndexes=[]

            if self.markEmpty:
                #Schováme si indexy dat, které pozdějí označíme za prázdná.
                emptyIndexes, actData=self.filterMarked(data[dataName])
            else:
                actData=data[dataName]
                
            if self.markEmpty:
                logging.info("\tPočet neprázdných dokumentů: "+str(len(actData)))
                
            docNum=len(actData)
            if splitIntoPartsOfMaxSize or workers>1:
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
                    featuresStorage=FeaturesExtractWorker.FeaturesStorage(manager=manager)
                    self.errorBoard=manager.Queue()
                    
                    for i in range(0,workers-1):
                        p=FeaturesExtractWorker(transformer, inputDataQueue, featuresStorage, self.errorBoard, actData, dataName)
                        processes.append(p)
                        p.start()
                        
                    helperWorkerP=FeaturesExtractWorker(transformer, inputDataQueue, featuresStorage, self.errorBoard, actData, dataName)
                    
                    
                    #Budeme vkládat části pro zpracování.
                    #Zkontrolujeme chyby a kdyžtak pomůžeme ostatním pokud je třeba
                    
                    
                    #rozsekáme data na části a vložíme je do fronty
                    for i in range(math.ceil(docNum/partSize)):
                        endOfPart=(i+1)*partSize if (i+1)*partSize<docNum else docNum
                        inputDataQueue.put((i, slice(i*partSize, endOfPart)))
                    
                    #vložíme příznaky konce.
                    for _ in range(0,workers):
                        inputDataQueue.put("EOF")
                    
                    shouldHelp=True
                    
                    while len(featuresStorage)<docNum:
                        featuresStorage.acquire()
                        
                        if int(len(featuresStorage)/docNum)!=1:
                            logging.info("\tHotovo: "+str(int(100*len(featuresStorage)/docNum))+"% - "+str(len(featuresStorage))+"/"+str(docNum))
                        
                        if shouldHelp:
                            #pomoc ostatním
                            featuresStorage.release()
                            if helperWorkerP.run(True)=="EOF":
                                shouldHelp=False
                            featuresStorage.acquire()
                        else:
                            featuresStorage.waitForChange(self.MAX_WAIT_TIMEOUT)

                        featuresStorage.release()
                        
                            
                        #kontrola chyb
                        self.__controlMulPErrors()
                        
                    logging.info("\tHotovo: "+str(int(100*len(featuresStorage)/docNum))+"% - "+str(len(featuresStorage))+"/"+str(docNum))
                    
                    
                        
                    #vyzvedneme výsledky
                    trans[dataName]=featuresStorage.popResults()
                    #čekáme na ukončení
                    for proc in processes:
                        self.__controlMulPErrors()
                        proc.join()
                    
                else:
                    
                    for i in range(math.ceil(docNum/partSize)):
                        endOfPart=(i+1)*partSize if (i+1)*partSize<docNum else docNum
                        actPart={dataName:actData[i*partSize:endOfPart]}

                        newS=transformer.transform(actPart)
                        if dataName not in trans:
                            trans[dataName]=newS
                        else:
                            trans[dataName]=vstack((trans[dataName], newS))                   
                        logging.info("\tHotovo: "+str(int(100*trans[dataName].shape[0]/docNum))+"% - "+str(trans[dataName].shape[0])+"/"+str(docNum))
                
            else:
                trans[dataName]=transformer.transform({dataName:actData})

            trans[dataName]=FeaturesContainer(trans[dataName], emptyIndexes)

            logging.info("konec extrakce příznaků pomocí "+allVectorizers[dataName]+" pro "+dataName)
        
        self.errorBoard=None    #Odstraníme nepotřebnou frontu, také kvůli případnému dumpu.
        return trans
        
    def extract(self, data, splitIntoPartsOfMaxSize, workers=1):
        """
        Extrahuje příznaky z dat.
        
        :param data: Pro extrakci příznaků.
        :param splitIntoPartsOfMaxSize: Tento parametr určuje maximální velikost části dat, která se bude v jednom kroku extrahovat.
            Pokud je uvedeno, bude vypisovat pro vektorizery v noFit postup.
        :param workers: Počet pracujicích procesů. Určuje kolik modelů bude zároveň trénováno.
        :returns: dict -- s extrahovanými příznaky Klíč je název dat.
        """
        
        return self.__extract(data, splitIntoPartsOfMaxSize, self.__transformers, workers)
        
    
    def learn(self, data, targets=None, workers=1, extracted=None):
        """
        Učí model.
        
        :param targets: Cíle dat.
        :param data: Pro učení nástroje pro extrakci příznaků.
        :param workers: Počet pracujicích procesů. Určuje kolik modelů bude zároveň trénováno.
        :type extracted: dict | None 
        :param extracted: Pokud není none (fit_transform) provede i extrakci dat, které byly použity pro trénovaní a uloží je do tohoto parametru. 
            None pouze fit.
        """
        
        allVectorizers={self.__fulltextName:self.fullTextVectorizer}
        if self.metaVectorizers:
            allVectorizers.update(self.metaVectorizers)
            
        workers=self.__manageWorkers(workers)

        needFit=self.__transformersNeedFit()
        if workers>len(needFit):
            #více pracantů něž-li je nutné
            workers=len(needFit)
            
        if workers==1:
            
            for dataName, transformer in self.__transformers.items():
                if self.markEmpty:
                    emptyIndexes, actData, actTargets=self.filterMarked(data[dataName], targets)
                else:
                    actData=data[dataName]
                    actTargets=targets
                    emptyIndexes=[]
                    
                if self.markEmpty:
                    logging.info("Počet neprázdných dokumentů pro "+dataName+": "+str(len(actData)))
                if dataName in needFit:
                    if extracted is not None:
                        logging.info("začátek učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků a samotná extrakce příznaků pro data "+dataName)
                        extracted[dataName]=FeaturesContainer(transformer.fit_transform({dataName:actData}, actTargets), emptyIndexes)
                        logging.info("konec učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků a samotná extrakce příznaků pro data "+dataName)
                    else:
                        logging.info("začátek učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků pro data "+dataName)
                        transformer.fit({dataName:actData}, actTargets)
                        logging.info("konec učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků pro data "+dataName)
                
        else:
            #více procesová varianta
            processes=[]
            manager=Manager()
            inputDataQueue=manager.Queue()
            resultsQueue=manager.Queue()
            self.errorBoard=manager.Queue()
            sharedLock=Lock()
            
            for _ in range(0,workers-1):

                p=FeaturesTrainWorker(inputDataQueue, resultsQueue, self.errorBoard, sharedLock, extracted is not None)
                processes.append(p)
                p.start()
                
            emptyForData={} #zde budeme ukládat seznamy prázdných indexů pro jednotlivá data.
            for dataName, transformer in self.__transformers.items():
                if self.markEmpty:
                    emptyForData[dataName], actData, actTargets=self.filterMarked(data[dataName], targets)
                else:
                    actData=data[dataName]
                    actTargets=targets
                    emptyForData[dataName]=[]
                    
                actTargets=None
                if allVectorizers[dataName] in self.useTargets:
                    actTargets=targets
                
                if self.markEmpty:
                    logging.info("Počet neprázdných dokumentů pro "+dataName+": "+str(len(actData)))
                    
                if dataName in needFit:
                    
                    inputDataQueue.put((transformer, {dataName:actData}, actTargets, allVectorizers[dataName], dataName))
            
            cntSaved=0  #počítadlo uložených natrénovaných modelů do paměti
            
            helperP=FeaturesTrainWorker(inputDataQueue, resultsQueue, self.errorBoard, sharedLock, extracted is not None)

            #Zkontrolujeme chyby a kdyžtak pomůžeme ostatním pokud je třeba
            #zároveň ukládáme výsledky
            while cntSaved<len(needFit):
                self.__controlMulPErrors()
                
                #pokusíme se pomoci
                helperP.run(True)
                
                # zpracování výsledků
                try:
                    if extracted is not None:
                        actDataName, actExtracted, actTrainedModel=resultsQueue.get()
                    else:
                        actDataName, actTrainedModel=resultsQueue.get()
                except queue.Empty:
                    pass
                else:
                    if extracted is not None:
                        extracted[actDataName]=FeaturesContainer(actExtracted, emptyForData[actDataName])
                    self.__transformers[actDataName]=actTrainedModel
                        
                    cntSaved+=1
                
                
                  
            #vložíme příznaky konce.
            for _ in range(0,workers-1):
                inputDataQueue.put("EOF")

            
            #čekáme na ukončení
            for proc in processes:
                self.__controlMulPErrors()
                proc.join()
                    
        self.errorBoard=None    #Odstraníme nepotřebnou frontu, také kvůli případnému dumpu.
            
    def getVectorizer(self, dataName):
        """
        Získání vektorizer k danému druhu dat.
        
        :param dataName: Název druhu dat.
        :return: Vektorizer k danému druhu dat. None pokud neexistuje
        """
        
        if dataName in self.__transformers:
            return self.__transformers[dataName].named_steps["vect"]
        
        return None
        
    def __controlMulPErrors(self):
        """
        Kontrola chyb z ostatních procesů.
        """
        if self.errorBoard and not self.errorBoard.empty():
            print("Vznikla chyba. Ukončuji všechny procesy.", file=sys.stderr)
            for p in active_children():
                p.terminate()
            exit()
            
    def __makeTransformerDict(self):
        """
        Na základě konfigurace vyrobí transformer dict.
        
        :returns: list -- transformer dict
        :raises: FeaturesNoData
        """
        
        trList={}
        
        allDataName=[]
        #prvně fulltext
        if self.__getFulltext:
            allDataName.append(self.__fulltextName)
        
        if self.__getMetaFields:
            allDataName=allDataName+self.__getMetaFields
            
        if not allDataName:
            #žádná data
            raise FeaturesNoData()
            
        for dataName in allDataName:
            vect=self.__makeVectorizer(dataName)
            trList[dataName]=Pipeline([
                        ('dataSel', DataTypeSelector(dataName)),
                        ('vect', vect)
                    ])        
        
        return trList
    
    
    def __makeVectorizer(self, dataName, useVocabulary=None):
        """
        Vytvoří nástroj pro vektorizaci, pro daný druh dat.
        
        :param dataName: Jméno dat.
        :param useVocabulary: Pokud není None použije (kde je to možné) přednastavený slovník.
        :returns: Vectorizer
        """

        analyzerUse=None
        vectName=None
        analyzerNameLog=""
        if dataName==self.__fulltextName:
            vectName=self.fullTextVectorizer
            
            analyzerNameLog=self.fullTextAnalyzer[0]
            if self.fullTextAnalyzer[1]:
                analyzerNameLog=analyzerNameLog+"/"+str(self.fullTextAnalyzer[1])
                
            analyzerUse=functools.partial(analyzerFulltextNgrams, self.fullTextAnalyzer[1])
            
            if self.fullTextAnalyzer[1]==1 and vectName==self.doc2VecName:
                #bez použití analyzeru se proces zrychli
                analyzerUse=None
        else:
            vectName=self.metaVectorizers[dataName]
            analyzerUse={
                "ngram":functools.partial(analyzerNgrams, self.metaAnalyzers[dataName][1]),
                "wholeitem":analyzerWholeItemX
                }[self.metaAnalyzers[dataName][0]]
                
            analyzerNameLog=self.metaAnalyzers[dataName][0]
            if self.metaAnalyzers[dataName][1]:
                analyzerNameLog=analyzerNameLog+"/"+str(self.metaAnalyzers[dataName][1])
            
        logging.info("Vytvářím "+vectName+" pro "+dataName+" ("+analyzerNameLog+").")
        
        return {
            "countvectorizer":CountVectorizer(analyzer=analyzerUse, vocabulary=useVocabulary), 
            "doc2vec": D2VVectorizer(size=self.doc2Vec["SIZE"], 
                                     alpha=self.doc2Vec["ALPHA"], 
                                     window=self.doc2Vec["WINDOW"], 
                                     minCount=self.doc2Vec["MIN_COUNT"], 
                                     workers=self.doc2Vec["WORKERS"], 
                                     iterCnt=self.doc2Vec["ITER"],
                                     sample=self.doc2Vec["SAMPLE"], 
                                     dm=self.doc2Vec["DM"], 
                                     negative=self.doc2Vec["NEGATIVE"],
                                     analyzer=analyzerUse) ,
            "tfidfvectorizer":TfidfVectorizer(analyzer=analyzerUse, vocabulary=useVocabulary), 
            "hashingvectorizer": HashingVectorizer(analyzer=analyzerUse, 
                                                   alternate_sign=self.hashingVectorizer["NON_NEGATIVE"], 
                                                   n_features=self.hashingVectorizer["N_FEATURES"]),
            "matchtargetvectorizer":MatchTargetVectorizer(analyzer=analyzerUse, lemmatizer=self.lemmatizer)
            }[vectName]
        
    @staticmethod
    def filterMarked(data, targets=None):
        """
        Odfiltruje prázdná data a vrátí pozice, na kterých se prázdná data nacházela.
        
        :param data: Data pro filtraci.
        :param targets: Pokud je použito odfiltruje také příslušné cíle k datům
        :returns: (list obsahující indexi, na kterých se nacházela prázdná data., odfiltrovaná data, popřípadě i cíle)
        """
        
        emptyIndexes=[]
        newData=[]
        newTargets=[]
        
        for i, d in enumerate(data):
            if d is not None and len(d)!=0:
                newData.append(d)
                if targets:
                    newTargets.append(targets[i])
            else:
                emptyIndexes.append(i)
        
        if targets:
            return (emptyIndexes, newData, newTargets)
        return (emptyIndexes, newData)
    
    
class FeaturesContainer(list):
    """
    Třída pro uchování extrhovaných příznaků dokumentů.
    Umožňuje ponechat extrahované příznaky v řídké matici a mít u některých dokumentů příznak empty.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Konstruktor.
        
        :param features: : array-like | sparse matrix -- Extrahované příznaky dokumentů.
        :param empty: set|list -- Indexy, které označuji prázdné dokumenty.
                    Doplňuje features. Tedy, pokud je zde uložen index například 3, tak bude vložena do features na indexu 3 značka.
                    Features se tedy celé posune. 
        """
        super().__init__()
        self.features=args[0]
        self.empty = set(args[1])
        self.len=None

        
    def __len__(self):
        """
        Počet dokumentů.
        
        :returns:  int -- Délka listu.
        """
        if self.len is None:
            self.len=self.features.shape[0]+len(self.empty)
        return self.len
    
    def vectorSize(self):
        """
        Vrácí velikost vektoru.
        """
        
        return self.features[0].shape[1]
        
    def __getitem__(self, ind):
        """
        Získej položku na daném indexu.
        
        :param ind: index položky
        :returns: položka na indexu
        """
        if type(ind) is list:
            #máme vybrat na základě listu s indexy
            #nejprve vytvoříme nový list s prázdnými dokumenty
            #poté zpracujeme neprázdné
            #ovšem indexy v ind mohou být přeházené, teoreticky se i vyskytovat vícekrát
            #je nutné vytvořít novou množinu empty a features, tak aby zůstalo dané pořadí v ind.
            
            
            newEmpty=[]
            newFeaturesIndexes=[]
            
            
            
            if len(self.empty)>0:
                #Máme prázdné, tudíž budeme muset posouvat indexy.
                
                #kvůli optimalizaci si neprveme projdeme, o kolik máme
                #indexy posouvat
            
                beforeEmpty={x:0 for x in set(ind)-self.empty}
                i=0
                beforeEmptyCnt=0
                while(i<len(self)):
                    if i in self.empty:
                        beforeEmptyCnt+=1
                    elif i in beforeEmpty:
                        beforeEmpty[i]=beforeEmptyCnt
                        
                    i+=1

                for i, origI in enumerate(ind):
                    if origI in self.empty:
                        newEmpty.append(i)
                    else:
                        newFeaturesIndexes.append(origI-beforeEmpty[origI])
            else:
                newFeaturesIndexes=ind

            return FeaturesContainer(self.features[newFeaturesIndexes], newEmpty)
            
        if ind in self.empty:
            return None
        
        beforeEmpty=0
        for i in sorted(self.empty):
            if i>=ind:
                break
            beforeEmpty+=1
            
        return self.features[ind-beforeEmpty, :]
    
    def __iter__(self):
        """
        Iteruje přes list.
        """
        i=0
        beforeEmpty=0
        while i<len(self):
            if i in self.empty:
                beforeEmpty+=1
                yield None
            else:
                yield self.features[i-beforeEmpty, :]
            
            i+=1
        
    def __str__(self):
        """
        Konverze tohoto listu na string.
        """
        
        strRepr=""
        for row in self:
            strRepr=strRepr+"\n"+str(row)
        return strRepr

    def __repr__(self):
        """
        Reprezentace tohoto listu.
        """
        return '%s(features=%s, empty=%s)' % (self.features, self.empty)    

    

def analyzerWholeItemX(x):
    """
    Analyzátor pro metadata, který bere celou položku v metadatovém poli jako celek.
    
    :param x:
    :returns: Celou položku.
    """
    return x
    

def analyzerNgrams(n, x):
    """
    Analyzátor tvořící ngramy pro metadata.
    
    :param n: velikost (například 2 vytvoří bigramy)
    :param x: list -- s prvky metadatového pole
    :returns: Ngramy.
    """
    allI=[]
    for item in x:
        splitedI=item.split()
        if len(splitedI)<n:
            allI.append(" ".join(splitedI))
        
        for i in range(len(splitedI)-n+1):
            allI.append(" ".join(splitedI[i:i+n]))
    
    return allI
    
 
def analyzerFulltextNgrams(n, x):
    """
    Analyzátor tvořící ngramy pro plný text.
    
    :param n: velikost (například 2 vytvoří bigramy)
    :param x: list -- se slovy plného textu
    :returns: Ngramy.
    """
    allI=[]
    if len(x)<n:
        allI.append("_".join(x))
        
    words=x
    if isinstance(x, DocReaderDataString):
        #z důvodů výkonu načteme dokument do paměti
        words=[w for w in x]
      
    for i in range(len(words)-n+1):
        allI.append("_".join(words[i:i+n]))
    
    return allI
    

class FeaturesExtractWorker(Process):
    """
    Třída reprezentující jeden pracující proces provádějící exrtrakci příznaků.
    """
    
    class FeaturesStorage(SharedStorage):
        """
        Uložiště extrahovaných příznaků. Synchronizované mezi procesy.
        """
        
        def __init__(self, manager=None):
            """
            Inicializace uložiště.
            
            :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
            """
            if manager is None:
                manager=Manager()
            super().__init__(SharedStorage.StorageType.DICT, manager)

        
        def addResult(self, partNumber, vecs):
            """
            Přidá vektory z části, která má pořadové číslo partNumber do uložiště.
            
            :param partNumber: Pořadové číslo části.
            :param vecs: Extrahované vektory.
            """

            self._safeAcquire()
            try:
                self._storage[partNumber]=vecs
                self._numOfData.value+=vecs.shape[0] if hasattr(vecs,"shape") else len(vecs)
            except:
                raise
            finally:
                self._safeRelease()
            self._notifyChange()
                
        def popResults(self):
            """
            Vyjme všechny vektory z uložiště a vráti je v seřazené podobě, dle pořadových čísel části.
            
            :rtype: array-like | sparse matrix | None
            :return: Extrahované příznaky. None -> prázdno
            """
            
            self._safeAcquire()
                
            try:

                res=None
                
                if len(self._storage)>0:
                    for partNum in range(min(self._storage.keys()), max(self._storage.keys())+1):
                        #Projedeme od nejmenšího po největší index a tím získáme seřazenou posloupnost.
                        
                        if partNum in self._storage:
                            if res is None:
                                res=self._storage[partNum]
                            else:
                                res=vstack((res, self._storage[partNum]))
                            #Odstraníme z uložiště, protože děláme pop.
                            del self._storage[partNum]
                            
                    #nastavíme počítadlo
                    self._numOfData.value=0

                return res
            except:
                raise
            finally:
                self._safeRelease()
                
            self._notifyChange()
            
        
            
    
    def __init__(self, model, inputDataQueue, featuresStorage, errorBoard, data, dataName):
        """
        Inicializace procesu.
        
        :param model: Model k extrahování příznaků.
        :type inputDataQueue: Queue
        :param inputDataQueue: Z této řady přímá data k extrakci.
                        Jeden záznam ve frontě je n-tice:
                            (partNumber, data)
        :type featuresStorage: FeaturesStorage
        :param featuresStorage: Zde ukládá výsledky.
        :type errorBoard: Queue
        :param errorBoard: Oznámení o chybách.
        :param data: Data pro extrakci příznaků (konkrétní druh).
        :param dataName: Název dat.
        """
        
        super().__init__()
        
        self.__model=model
        self.__inputDataQueue=inputDataQueue
        self.__featuresStorage=featuresStorage
        self.__errorBoard=errorBoard
        self.__data=data
        self.__dataName=dataName
    
    def run(self, once=False, timeoutInQueue=1):
        """
        Čekání na vstupní data a extrakce.
        
        :param once: Pokud je True. Zpracuje pávě jeden blok, pokud není ihned k dispozici, tak končí.
        :param timeoutInQueue: Bere se v úvahu pouze pokud je parametr once=true. Nastavuje timeout pro čekání ve frontě na vstupní data.
        :return: "EOF"| None
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
                    partNumber, dataSel =msg
                    
                    data={self.__dataName:self.__data[dataSel]}
                        
                    self.__featuresStorage.addResult(partNumber, self.__model.transform(data))
                    
                gc.collect()
                if once:
                    return
        except:
            self.__errorBoard.put("ERROR")
            print(traceback.format_exc(), file=sys.stderr)
            
class FeaturesTrainWorker(Process):
    """
    Třída reprezentující jeden pracující proces provádějící učení modelu pro extrakci příznaků.
    """
    
    def __init__(self, inputDataQueue, resultsQueue, errorBoard, sharedLock, extract=False):
        """
        Inicializace procesu.

        :type inputDataQueue: Queue
        :param inputDataQueue: Z této řady přímá data k trénování modelu.
                        Jedem záznam ve frontě je n-tice:
                            (partNumber, data)
        :type resultsQueue: Queue
        :param resultsQueue: Zde vrací výsledkný model.
                        Jedem záznam ve frontě je n-tice:
                            (dataName, model)
        :type errorBoard: Queue
        :param errorBoard: Oznámení o chybách.
        :param sharedLock: Sdílený zámek. Používá se pro výpis logů.
        :type extract: boolean
        :param extract: True => použije fit_transform a zárověň tedy i extrahuje příznaky.
        """
        
        super().__init__()
        
        self.__inputDataQueue=inputDataQueue
        self.__resultsQueue=resultsQueue
        self.__errorBoard=errorBoard
        self.__sharedLock=sharedLock
        self.__extract=extract
        
    def run(self, once=False):
        """
        Čekání na vstupní data a učení modelu.
        
        :param once: Pokud je True. Zpracuje pávě jeden blok, pokud není ihned k dispozici, tak končí.
        :return: "EOF"| None
        """
        acquired=False
        try:
            while True:
                try:
                    msg=self.__inputDataQueue.get(timeout=1 if once else None)

                except queue.Empty:
                    return
                else:
                    if msg == "EOF":
                        return "EOF"
                    model, data, actTargets, vectName, dataName =msg
                    
                    self.__sharedLock.acquire()
                    if self.__extract:
                        logging.info("začátek učení modelu "+vectName+" pro extrakci příznaků a samotná extrakce příznaků pro data "+dataName)
                    else:
                        logging.info("začátek učení modelu "+vectName+" pro extrakci příznaků pro data "+dataName)
                    
                    acquired=True
                    self.__sharedLock.release()
                    
                    if self.__extract:
                        self.__resultsQueue.put((dataName, model.fit_transform(data, actTargets), model))
                    else:
                        self.__resultsQueue.put((dataName, model.fit(data, actTargets)))
                    
                    self.__sharedLock.acquire()
                    if self.__extract:
                        logging.info("konec učení modelu "+vectName+" pro extrakci příznaků a samotná extrakce příznaků pro data "+dataName)
                    else:
                        logging.info("konec učení modelu "+vectName+" pro extrakci příznaků pro data "+dataName)
                    acquired=True
                    self.__sharedLock.release()
                    
                gc.collect()
                if once:
                    return
        except:
            if acquired:
                self.__sharedLock.release()
            self.__errorBoard.put("ERROR")
            print(traceback.format_exc(), file=sys.stderr)