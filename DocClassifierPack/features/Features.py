# -*- coding: UTF-8 -*-
"""
Obsahuje třídu pro extrakci příznaků a další potřebné nástroje.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer, TfidfVectorizer

from .Vectorizers import D2VVectorizer
from ..utils.DocReader import DocReaderDataString
from ..utils.DataSet import DataTypeSelector

import functools
import logging
    
import math

from scipy.sparse import vstack

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
    vectorizersNames=["tfidfvectorizer", "doc2vec", "countvectorizer", "hashingvectorizer"]
    
    countVectorizerName="countvectorizer"
    doc2VecName="doc2vec"
    tfidfVectorizerName="tfidfvectorizer"
    hashingVectorizerName="hashingvectorizer"
        
    #Názvy analyzátorů pro fulltext. První je používán jako defaultní. Vše malé znaky.
    fulltextAnalyzersNames=["ngram"]
    
    #Názvy analyzátorů pro metadata. První je používán jako defaultní. Vše malé znaky.
    metaAnalyzersNames=["ngram", "wholeitem"]
    
    ngramName="ngram"
    wholeitemName="wholeitem"
    
    #U těchto extraktorů se použije fit a poté se aplikuje transform na částech. Takto můžeme vyppisovat postup.
    #Je zde pouze hashingVectorizer (zatím?), protože má prázdnou fázi fit.
    doPartial=[hashingVectorizerName]

    def __init__(self, getFulltext, getMetaFields, fullTextVectorizer, fullTextAnalyzer,
                 metaVectorizers, metaAnalyzers, hashingVectorizer, doc2Vec, fulltextName="fulltext"):
        """
        Konstruktor. Připraví nástroje pro extrakci příznaků.
        
        :param getFulltext: bool -- True zahrne fulltext. False nezahrne.
        :param getMetaFields: list -- Názvy metadatových polí, které se mají zahrnout do extrakce příznaků. 
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
        :param fulltextName: Název dat s plným textem
        """
        self.getFulltext=getFulltext
        self.getMetaFields=getMetaFields
        self.fullTextVectorizer=fullTextVectorizer
        self.fullTextAnalyzer=fullTextAnalyzer
        self.metaVectorizers=metaVectorizers
        self.metaAnalyzers=metaAnalyzers
        self.hashingVectorizer=hashingVectorizer
        self.doc2Vec=doc2Vec
        self.fulltextName=fulltextName
        
        self.__transformers=self.__makeTransformertList()
        

    def flush(self):
        """
        Vhodné pro uvolnění paměti.
        Použít v případě, kdy se už objekt nebude používat.
        """
        del self.__transformers
        
    def extractAndLearn(self, data, splitIntoPartsOfMaxSize=None):
        """
        Učí model a extrahuje příznaky z dat. Poskytuje také možnost zobrazení postupu pro extraktory v doPartial.
        
        :param data: Pro extrakci příznaků a učení nástroje pro extrakci příznaků.
        :param splitIntoPartsOfMaxSize: Tento parametr určuje maximální velikost části dat, která se bude v jednom kroku extrahovat.
            Pokud je uvedeno, bude vypisovat pro vektorizery v doPartial postup.
                
        :returns: dict -- s extrahovanámi příznaky Klíč je název dat.
        """
        
        trans={}
        
        allVectorizers={self.fulltextName:self.fullTextVectorizer}
        if self.metaVectorizers:
            allVectorizers.update(self.metaVectorizers)
        for dataName, transformer in self.__transformers:
            if splitIntoPartsOfMaxSize and allVectorizers[dataName] in self.doPartial:
                logging.info("začátek učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků pro data "+dataName)
                transformer.fit(data)
                logging.info("konec učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků pro data "+dataName)
                logging.info("začátek extrakce příznaků pomocí "+allVectorizers[dataName]+" pro "+dataName)
                
                docNum=len(data[dataName])
                for i in range(math.ceil(docNum/splitIntoPartsOfMaxSize)):
                    endOfPart=(i+1)*splitIntoPartsOfMaxSize if (i+1)*splitIntoPartsOfMaxSize<docNum else docNum
                    newS=transformer.transform({dataName:data[dataName][i*splitIntoPartsOfMaxSize:endOfPart]})
                    if dataName not in trans:
                        trans[dataName]=newS
                    else:
                        trans[dataName]=vstack((trans[dataName], newS))
                
                    logging.info("Hotovo: "+str(int(100*trans[dataName].shape[0]/docNum))+"% - "+str(trans[dataName].shape[0])+"/"+str(docNum))
                
                logging.info("konec extrakce příznaků pomocí "+allVectorizers[dataName]+" pro "+dataName)
            else:
                logging.info("začátek učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků a samotná extrakce příznaků pro data "+dataName)
                trans[dataName]=transformer.fit_transform(data)
                logging.info("konec učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků a samotná extrakce příznaků pro data "+dataName)
        return trans
        
    def extract(self, data, splitIntoPartsOfMaxSize):
        """
        Extrahuje příznaky z dat.
        
        :param data: Pro extrakci příznaků.
        :param splitIntoPartsOfMaxSize: Tento parametr určuje maximální velikost části dat, která se bude v jednom kroku extrahovat.
            Pokud je uvedeno, bude vypisovat pro vektorizery v doPartial postup.
            
        :returns: dict -- s extrahovanámi příznaky Klíč je název dat.
        """
        
        trans={}
        allVectorizers={self.fulltextName:self.fullTextVectorizer}
        if self.metaVectorizers:
            allVectorizers.update(self.metaVectorizers)
        for dataName, transformer in self.__transformers:
            logging.info("začátek extrakce příznaků pomocí "+allVectorizers[dataName]+" pro "+dataName)
            docNum=len(data[dataName])
            if splitIntoPartsOfMaxSize and allVectorizers[dataName] in self.doPartial:
                for i in range(math.ceil(docNum/splitIntoPartsOfMaxSize)):
                    endOfPart=(i+1)*splitIntoPartsOfMaxSize if (i+1)*splitIntoPartsOfMaxSize<docNum else docNum
                    newS=transformer.transform({dataName:data[dataName][i*splitIntoPartsOfMaxSize:endOfPart]})
                    if dataName not in trans:
                        trans[dataName]=newS
                    else:
                        trans[dataName]=vstack((trans[dataName], newS))
                        
                    logging.info("Hotovo: "+str(int(100*trans[dataName].shape[0]/docNum))+"% - "+str(trans[dataName].shape[0])+"/"+str(docNum))
            else:
                
                trans[dataName]=transformer.transform(data)
            logging.info("konec extrakce příznaků pomocí "+allVectorizers[dataName]+" pro "+dataName)
        return trans
    
    def learn(self, data):
        """
        Učí model.
        
        :param data: Pro učení nástroje pro extrakci příznaků.
        """
        
        allVectorizers={self.fulltextName:self.fullTextVectorizer}
        if self.metaVectorizers:
            allVectorizers.update(self.metaVectorizers)
            
        for dataName, transformer in self.__transformers:
            logging.info("začátek učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků pro data "+dataName)
            transformer.fit(data)
            logging.info("konec učení modelu "+allVectorizers[dataName]+" pro extrakci příznaků pro data "+dataName)
        
    def __makeTransformertList(self):
        """
        Na základě konfigurace vyrobí transformer list.
        
        :returns: list -- transformer list
        :raises: FeaturesNoData
        """
        
        trList=[]
        
        allDataName=[]
        #prvně fulltext
        if self.getFulltext:
            allDataName.append(self.fulltextName)
        
        if self.getMetaFields:
            allDataName=allDataName+self.getMetaFields
            
        if not allDataName:
            #žádná data
            raise FeaturesNoData()
            
        for dataName in allDataName:
            trList.append((dataName, Pipeline([
                        ('dataSel', DataTypeSelector(dataName)),
                        ('vect', self.__makeVectorizer(dataName))
                    ])))
        
        
        return trList
    
    def __makeVectorizer(self, dataName):
        """
        Vytvoří nástroj pro vektorizaci, pro daný druh dat.
        
        :param dataName: Jméno dat.
        :returns: Vectorizer
        """

        analyzerUse=None
        vectName=None
        analyzerNameLog=""
        if dataName==self.fulltextName:
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
            "countvectorizer":CountVectorizer(analyzer=analyzerUse), 
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
            "tfidfvectorizer":TfidfVectorizer(analyzer=analyzerUse), 
            "hashingvectorizer": HashingVectorizer(analyzer=analyzerUse, 
                                                   non_negative=self.hashingVectorizer["NON_NEGATIVE"], 
                                                   n_features=self.hashingVectorizer["N_FEATURES"])
            }[vectName]
        
     

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
    
