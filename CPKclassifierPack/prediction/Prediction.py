# -*- coding: UTF-8 -*-
"""
Obsahuje třídu pro zpracování výsledků predikce klasifikátorů.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

import csv


class Prediction(object):
    """
    Třída pro zpracování výsledků predikce klasifikátorů.
    """

    def __init__(self, predictedFieldName, nBest, nBestPrefix, writeTo, itemDelimiter=None, confPostfix=None):
        """
        Konstrukce.
        
        :param predictedFieldName: Název pole, kde bude predikovaný cíl.
        :param nBest: Kolik nejlepších predikovaných cílu se má vypsat. Nejméně vypíše jeden.
        :param nBestPrefix: Prefix pro názvy polí, které budou obsahovat N nejlepších cílů.
        :param writeTo: Kam se uloží výsledek. STDOUT nebo klasický soubor (otevřený).
        :param itemDelimiter: Pokud je uveden budou při tisku prvky pole sloučeny za pomocí tohoto rozdělovače.
        :param confPostfix: Pokud je to možné a není None, tak výpíše jistoty k jednotlivým predikcím. Slouží jako postfix k názvu sloupce.
        """
        
        self.predictedFieldName=predictedFieldName
        self.nBest=nBest if nBest > 1 else 1
        self.nBestPrefix=nBestPrefix
        self.writeTo=writeTo
        self.itemDelimiter=itemDelimiter
        self.confPostfix=confPostfix
        
    def write(self, predicted, metaForWrite, targetsNames=[], writeHeader=False):
        """
        Vypíše výsledky predikce.
        
        :param predicted: Jistoty k jednotlivým cílům ke každému dokumentu. 
            Pokud targetsNames=[] => List s názvy predikovaných cílů k dokumentům.
        :param metaForWrite: dict -- Dodatečná metadata, která se mají vypsat společně s predikcí.
            Klíč název pole metadat. Hodnota list s hodnotami pro každý dokument.
        :param targetsNames: list -- Názvy cílů. Pořadí odpovídá pořadí jistot v predicted.
        :param firstOnly: True predicted => je pouze list s názvy predikovaných cílů k dokumentům. Nelze najít n nejlepších.
        :param writeHeader: Zapíše csv hlavičku.
        
        """
        
        allFields=[self.predictedFieldName]
        if self.confPostfix and targetsNames!=[]:
            allFields.append(self.predictedFieldName+self.confPostfix)
        showBest=self.nBest
        
        if targetsNames!=[]:
            if showBest>len(targetsNames):
                showBest=len(targetsNames)
    
            if showBest<1:
                showBest=1

            for b in range(2, showBest+1):
                allFields.append(self.nBestPrefix+str(b))
                if self.confPostfix:
                    allFields.append(allFields[-1]+self.confPostfix)
        
        allFields=allFields+list(metaForWrite.keys())
        
        writerRes = csv.DictWriter(self.writeTo, fieldnames=allFields)
        if writeHeader:
            writerRes.writeheader()
        
        for pAndMeta in zip(predicted, *(metaForWrite[i] for i in metaForWrite.keys())):
            dWrite={}
            if targetsNames==[]:
                dWrite[self.predictedFieldName]=pAndMeta[0]
            else:
                nBest=self.selectNBest(pAndMeta[0], targetsNames)
                
                confValue=pAndMeta[0][targetsNames.index(nBest[0])]
                
                #predikovaný cíl.
                if confValue==0:
                    #Nepovedlo se určit cíl.
                    dWrite[self.predictedFieldName]=""
                    confValue=""
                else:
                    dWrite[self.predictedFieldName]=nBest[0]
                    
                if self.confPostfix: 
                    dWrite[self.predictedFieldName+self.confPostfix]=confValue
                
                #Další predikované cíle, je-li to žádáno.
                for b in range(1, showBest):
                    confValue=pAndMeta[0][targetsNames.index(nBest[b])]
                    
                    if confValue==0:
                        #Nepovedlo se určit cíl.
                        dWrite[self.nBestPrefix+str(b+1)]=""
                        confValue=""
                    else:
                        dWrite[self.nBestPrefix+str(b+1)]=nBest[b]
                        
                    if self.confPostfix:
                        dWrite[self.nBestPrefix+str(b+1)+self.confPostfix]=confValue
            
                
                
            for i, mF in enumerate(metaForWrite.keys()):
                dWrite[mF]=pAndMeta[i+1]

            
            writerRes.writerow(self.__translateRowToWriteFormat(dWrite))  
            
        
    def __translateRowToWriteFormat(self, row):
        """
        Převede řádek do formátu pro výpis.
        
        :param row: Řádek pro převod
        :returns: Převedený řádek.
        """
        if not self.itemDelimiter:
            return row
        
        trRow={}
        for name, vals in row.items():
            
            if isinstance(vals, list):
                trRow[name]=self.itemDelimiter.join([ str(x) for x in vals])
            else:
                trRow[name]=vals
            
        return trRow
        
    def selectNBest(self, predicted, targetsNames):
        """
        Vybere n nejlepších cílů.
        
        :param predicted: Jistoty.
        :param targetsNames: Názvy cílů.
        :returns: list -- n nejlepších cílů
        """
        sortedTar=sorted(zip(targetsNames, predicted), key=lambda x: x[1],reverse=True)
        
        return [x[0] for x in sortedTar[:self.nBest]]
        
        
        
        
        