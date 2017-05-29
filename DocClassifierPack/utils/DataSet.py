# -*- coding: UTF-8 -*-
"""
Obsahuje třídu pro získání dat k trénování/testování klasifikátoru.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

from .DocReader import DocReader, DocReaderMetadata, DocReaderInvalidMetadataFields
import csv

import logging


class DataSetInvalidTarget(Exception):
    """
    Nevalidní cíl pro trénování.
    """
    pass

class DataSetInvalidMetadataFields(Exception):
    """
    Některá pole nejsou ve vstupním souboru s metadaty.
    """
    pass

class DataSetNoDataPath(Exception):
    """
    Nezadána cesta k datovému souboru s dokumenty.
    """
    pass

class DataSet(object):
    """
    Třída pro získání a zápis dat. Vytváření trénovací množiny.
    """


    def __init__(self, metadata, data=None, getMetaFields=None, getFulltext=None, targetField=None, 
                 lazyEvalData=None, nonEmpty=None, empty=None, fieldRegex=None, minPerField=None, maxPerField=None, 
                 itemDelimiter=None, selectWords=None, selectItems=None, fulltextName="fulltext"):
        """
        Konstruktor. Připraví nástroje pro čtení dat.
        
        :param metadata: Cesta k souboru s metadaty.
        :param data: Cesta k souboru s daty.
        :param getMetaFields: Názvy metadatových polí, které se mají zahrnout. 
        :param getFulltext: Určuje jestli má být zahrnutý plný text(data) True/False
        :param targetField: Pole, které bude použito pro cíle. 
            Obsahuje-li více položek bude použita první. Popřípadě stačí vybrat pomocí SELECT_ITEMS.
        :param lazyEvalData: Použít líné vyhodnocení. Namísto stringu bude vrácen DocReaderDataString s daty.
        :param nonEmpty: list -- polí z metadat, které nesmí být prázdné
        :param empty: list -- polí z metadat, které musí být prázdné.
        :param fieldRegex: dict -- ve formátu: klíč jako jméno pole a hodnota je dvojice (regulární výraz, flagy).
            Položka v poli bude vynechána pokud regulární výraz neodpovídá.
        :param minPerField: dict -- ve formátu: klíč jako jméno pole a hodnota s minimálním počtem dokumentů
            Filtr definuje minimální počet dokumentů na položku v poli.
            Pro příklad: Pokud pole obsahuje kategorii a kategorie má méně než minimální počet dokumentů, celá kategorie bude vynechána.
        :param maxPerField: dict -- ve formátu: klíč jako jméno pole a hodnota s maximálním počtem dokumentů
            Filtr definuje maximální počet dokumentů na položku v poli.
            Pokud je hodnota float v intervalu <0,1>, maximálně x procent (int((numberOfDcuments*x)+0.5)) dokumentů v poli bude přečteno.
        :param itemDelimiter: Oddělovač (řetězec), který separuje položky v poli.
        :param selectWords: slice/integer -- pro vybrání slov v každém dokumentu
        :param selectItems: dict -- klíč je název pole a hodnota je slice/integer pro vybrání položek.
        :param fulltextName: Název dat s plným textem
        :raises: DataSetInvalidMetadataFields
        """
        
        self.getMetaFields=getMetaFields
        self.getFulltext=getFulltext
        self.targetField=targetField
        self.fulltextName=fulltextName

        #získání objektu pro čtení
        logging.info("začátek čtení dat")
        
        try:
            self.reader=self.__getDataReader(metadata, data, lazyEvalData, nonEmpty, empty, fieldRegex, minPerField, maxPerField, 
                 itemDelimiter, selectWords, selectItems)
        except DocReaderInvalidMetadataFields:
            raise DataSetInvalidMetadataFields()
        
        logging.info("konec čtení dat")
        #získání všech druhů metadat      
        
        if not set(self.getMetaFields).issubset(self.reader.fieldNames):
            raise DataSetInvalidMetadataFields()
        
    def writeFullText(self, fileName):
        """
        Zapíše data plného textu do souboru. Pokud nebyl zadán plný text, jedná se o prázdnou operaci.
        
        :param fileName: Název souboru (cesta).
        """
        
        if self.getFulltext:
            
            with open(fileName, "w") as fWrite:
                
                logging.info("začátek zápisu plného textu")
                for doc in self.reader:
                    fWrite.write(" ".join(doc[0])+"\n")
                logging.info("konec zápisu plného textu")
                
    def writeMetadata(self, fileName):
        """
        Zapíše metadat do souboru. Ve formátu csv. Pokud nebyla zadána metadata, jedná se o prázdnou operaci.
        
        :param fileName: Název souboru (cesta).
        """
        
        if self.getMetaFields:
            with open(fileName, "w") as fWrite:
                logging.info("začátek zápisu metadat: "+", ".join(self.getMetaFields))
                writerMeta = csv.DictWriter(fWrite, fieldnames=self.getMetaFields)
                writerMeta.writeheader()
                
                for doc in self.reader:
                    if self.getFulltext:
                        doc=doc[1]
                    writerMeta.writerow(self.__transMetaToWriteFormat(doc))
                    
                logging.info("konec zápisu metadat: "+", ".join(self.getMetaFields))
        
    def writeMetaAndFulltextData(self, metadataFileName, fulltextDataFileName):
        """
        Zapíše metadata a plný tex do souborů. Pokud nebyla zadána metadata nebo plný text, jedná se o prázdnou operaci.
        
        :param metadataFileName: Název/cesta k metadatovémo souboru.
        :param fulltextDataFileName: Název/cesta k souboru pro plný text.
        """
        
        if self.getMetaFields and self.getFulltext:
            with open(fulltextDataFileName, "w") as fWrite, open(metadataFileName, "w") as fMetaWrite:
                logging.info("začátek zápisu plného textu a metadat: "+", ".join(self.getMetaFields))
                writerMeta = csv.DictWriter(fMetaWrite, fieldnames=self.getMetaFields)
                writerMeta.writeheader()
                for doc in self.reader:
                    fWrite.write(" ".join(doc[0])+"\n")
                    writerMeta.writerow(self.__transMetaToWriteFormat(doc[1]))
                logging.info("konec zápisu plného textu a metadat: "+", ".join(self.getMetaFields))
        
    def __transMetaToWriteFormat(self, metaData):
        """
        Transformuje metadata dokumentu, do formátu pro zápis do souboru.
        
        :param metaData: Metadata dokumentu.
        """
        if self.reader.itemDelimiter:
            return { k: self.reader.itemDelimiter.join(metaData[k]) if metaData[k] is not None else metaData[k]  for k in self.getMetaFields }
        else:
            return { k: "".join(metaData[k]) if metaData[k] is not None else metaData[k] for k in self.getMetaFields }
        
    
    def getData(self):
        """
        Získá data.
        
        :returns: data
        """
        
        
        allDataNames=[]
        if self.getFulltext:
            allDataNames.append(self.fulltextName)
        if self.getMetaFields:
            allDataNames=allDataNames+self.getMetaFields
        allDataNames=" ".join(allDataNames)
        logging.info("začátek získávání dat "+allDataNames)
        dataForTraining=self.__getData(targets=False)
        logging.info("konec získávání dat "+allDataNames)
        return dataForTraining
        
    def getTrainData(self):
        """
        Získá data ve vhodné formě pro trénování.
        
        :returns: pair -- (data pro trénování, cíle)
        :raises: DataSetInvalidTarget
        """
        
        if not self.targetField or not set([self.targetField]).issubset(self.reader.fieldNames):
            raise DataSetInvalidTarget()
        
        allDataNames=[]
        if self.getFulltext:
            allDataNames.append(self.fulltextName)
        if self.getMetaFields:
            allDataNames=allDataNames+self.getMetaFields
        allDataNames=" ".join(allDataNames)
        logging.info("začátek získávání dat pro trénování: "+allDataNames)
        result=self.__getData(targets=True)
        logging.info("konec získávání dat pro trénování: "+allDataNames)
        return result
    
    def __getData(self, targets=False):
        """
        Získá data.
        
        :param targets: I cíle
        :returns: data pokud target True získá i cíle (data, cíle)
        """
        data=dict( (kind, []) for kind in self.getMetaFields)
        
        targetsData=[]
        
        if self.getFulltext:
            data[self.fulltextName]=[]
            
        for doc in self.reader:
            if self.getFulltext:
                data[self.fulltextName].append(doc[0])
                doc=doc[1]
                
            for kind in self.getMetaFields:
                data[kind].append(doc[kind])
                
            if targets:
                targetsData.append(doc[self.targetField][0])
            
        if targets:
            return (data, targetsData)
        else:
            return data
        
    def __getDataReader(self, metadata, data, lazyEvalData=None, nonEmpty=None, empty=None, fieldRegex=None, minPerField=None, maxPerField=None, 
                 itemDelimiter=None, selectWords=None, selectItems=None):
        """
        Na základě konfiguračního souboru, a argumentů metadata/data, vytvoří DocReader nebo DocReaderMetadata.
        
        :param metadata: Cesta k souboru s metadaty.
        :param data: Cesta k souboru s daty.
        :param lazyEvalData: Použít líné vyhodnocení. Namísto stringu bude vrácen DocReaderDataString s daty.
        :param nonEmpty: list -- polí z metadat, které nesmí být prázdné
        :param empty: list -- polí z metadat, které musí být prázdné. 
        :param fieldRegex: dict -- ve formátu: klíč jako jméno pole a hodnota je dvojice (regulární výraz, flagy).
            Položka v poli bude vynechána pokud regulární výraz neodpovídá.
        :param minPerField: dict -- ve formátu: klíč jako jméno pole a hodnota s minimálním počtem dokumentů
            Filtr definuje minimální počet dokumentů na položku v poli.
            Pro příklad: Pokud pole obsahuje kategorii a kategorie má méně než minimální počet dokumentů, celá kategorie bude vynechána.
        :param maxPerField: dict -- ve formátu: klíč jako jméno pole a hodnota s maximálním počtem dokumentů
            Filtr definuje maximální počet dokumentů na položku v poli.
            Pokud je hodnota float v intervalu <0,1>, maximálně x procent (int((numberOfDcuments*x)+0.5)) dokumentů v poli bude přečteno.
        :param itemDelimiter: Oddělovač (řetězec), který separuje položky v poli.
        :param selectWords: slice/integer -- pro vybrání slov v každém dokumentu
        :param selectItems: dict -- klíč je název pole a hodnota je slice/integer pro vybrání položek.
        
        :returns: DocReader|DocReaderMetadata
        """
        
        if self.getFulltext:
            if not data:
                raise DataSetNoDataPath()
            
            return DocReader(data, metadata, 
                             lazyEvalData=lazyEvalData, 
                             nonEmpty=nonEmpty, 
                             empty=empty,
                             fieldRegex=fieldRegex, 
                             minPerField=minPerField, 
                             maxPerField=maxPerField, 
                             itemDelimiter=itemDelimiter, 
                             selectWords=selectWords, 
                             selectItems=selectItems)
        else:
            return DocReaderMetadata(metadata, 
                                     nonEmpty=nonEmpty, 
                                     empty=empty,
                                     fieldRegex=fieldRegex, 
                                     minPerField=minPerField, 
                                     maxPerField=maxPerField,
                                     itemDelimiter=itemDelimiter, 
                                     selectItems=selectItems)
            
class DataTypeSelector(object):
    """
    Třída pro výběr druhu dat.
    """
    def __init__(self, key):
        """
        Inicializace.
        
        :param key: Klíč pro výběr.
        """
        self.key = key

    def fit(self, x, y=None):
        """
        Pouze vrací sám sebe. Parametry jsou nepodstatné.
        
        :param x:
        :param y:
        :returns: Sám sebe.
        """
        return self

    def transform(self, dataDict):
        """
        Vrátí druh data. Na základě klíče.
        
        :param dataDict: dict -- obsahující data více druhů
        :returns: Vybraný druh dat.
        """
        return dataDict[self.key]
            