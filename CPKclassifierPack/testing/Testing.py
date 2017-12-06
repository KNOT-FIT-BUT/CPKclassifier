# -*- coding: UTF-8 -*-
"""
Obsahuje nástroje pro zpracování výsledků testování klasifikátoru.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""
import pandas as pd
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from ..utils.Targets import TargetsHier

class Testing(object):
    """
    Nástroje pro zpracování výsledků testování klasifikátoru.
    """

    translations={
        "exact match":"Přímá shoda",
        "direct family":"Přímá rodina",
        "family":"Rodina",
        "same top level":"Stejná nejvyšší úroveň",
        "p":"přesnost (precision)",
        "r":"úplnost (recall)",
        "f":"míra F1",
        }
    
    def __init__(self, hierDelimiter=None, nBest=1):
        """
        Inicializace.
        
        :param hierDelimiter:  Řetězec, který odděluje úrovně hierarchie. Pokud je
        dodán tiskne i statistiky na základě hierarchie.
        :param nBest: Statistiky pro n nejlepších
        """
        self.hierDelimiter=hierDelimiter
        self.partsValues=[]
        self.nBest=nBest
        self.hier=TargetsHier(None, self.hierDelimiter)
        self.unclassified=0 #udává počet dokumentů, které se nepodařilo klasifikovat
        
    def processResults(self, predicted, predictedProba, targets, targetsNames, trainTargets):
        """
        Vypočítá a uloží výsledky pro jeden krok křížové validace.
        Pokud targetsNames [] pak se předpkládá, že nemáme k dispozici pravděpodobnosti.
        
        :param predicted: list -- s názvy predikovaných cílů k dokumentům.
        :param predictedProba:  Jednotlivé jistoty příslušnosti dokumentů do dané kategorie.
        :param targets: Cíle dokumentů.
        :param targetsNames: list -- Názvy cílů. Pořadí odpovídá pořadí jistot v predictedProba.
        :param trainTargets: Cíle dokumentů v trénovací množině.
        """
        
        
        predicted, predictedProba, targets=self.__unclassifiedFilter(predicted, predictedProba, targets)
        if len(targets)==0:
            return
        
        cntTrain={}
        

        labels=unique_labels(targets)
        scoreValues={}
        
        for label in labels:
            cntTrain[label]=trainTargets.count(label)


        p, r, f1, s = precision_recall_fscore_support(targets, predicted)
        scoreValues["targets"]={}
        for i, label in enumerate(labels):
            scoreValues["targets"][label]=(p[i], r[i], f1[i], s[i], cntTrain[label])
        
        scoreValues["prf"]=[]
        for avg in ["micro", "macro", "weighted"]:
            p, r, f, s=precision_recall_fscore_support(targets, predicted, average=avg)
            
            if avg=="weighted":
                scoreValues["prf"].append((avg+" (support)", (p, r, f)))
            else:
                scoreValues["prf"].append((avg, (p, r, f)))
            
        
        
        if self.hierDelimiter is not None:
            scoreValues["accNBest"]=self.countAccuracy(predictedProba, targets, targetsNames, self.hierDelimiter is not None)
            
        self.partsValues.append(scoreValues)

    
    def printTargetsCVScore(self, crossValidationScores):
        """
        Vytiskne metriky cílů získaných z křížové validace.
        
        :param crossValidationScores: výsledek z metody crossValidationScores
        """
        print("cíl\tpřesnost\túplnost\tmíra F1\tsupport\tprůměrný počet dokumentů pro trénování")
        
        for target, v in crossValidationScores["targets"].items(): 
            print(target+"\t"+str(v["p"])+"\t"+str(v["r"])+"\t"+str(v["f"])+"\t"+str(v["s"])+"\t"+str(v["cntTrain"]))
        
    def printNBestCVScore(self, crossValidationScores):
        """
        Vytiskne metriky n nejlepších cílů získaných z křížové validace.
        
        :param crossValidationScores: výsledek z metody crossValidationScores
        """
        
            
        for n, kinds in crossValidationScores["nbest"].items():
            print(str(n)+":")
            for k, v in kinds:
                print("\t"+self.translations[k]+"\t"+str(v))
            
        
    def printAVGCVScore(self, crossValidationScores):
        """
        Vytiskne průměry metrik získaných z křížové validace.
        
        :param crossValidationScores: výsledek z metody crossValidationScores
        """
        
        for avgName, v in crossValidationScores["avg"]:
            print(avgName)
            for metName, met in sorted(v.items()):
                print("\t"+self.translations[metName]+"\t"+str(met))
    
    def crossValidationScores(self):
        """
        Statistika na základě uložených kroků křížové validace.
        :returns: dict--se statistkami na základě křížové validace.
        """

        crossValScore={
            "targets":{},
            "nbest":{},
            "avg":[]
            }
        if len(self.partsValues)==0:
            return crossValScore
        
        allTargets=set()
        for part in self.partsValues:
            for target in part["targets"]:
                allTargets.add(target)
                
        for target in allTargets: 
            p=np.mean([self.partsValues[x]["targets"][target][0] for x in range(len(self.partsValues)) if target in self.partsValues[x]["targets"]])
            r=np.mean([self.partsValues[x]["targets"][target][1] for x in range(len(self.partsValues)) if target in self.partsValues[x]["targets"]])
            f=np.mean([self.partsValues[x]["targets"][target][2] for x in range(len(self.partsValues)) if target in self.partsValues[x]["targets"]])
            s=np.mean([self.partsValues[x]["targets"][target][3] for x in range(len(self.partsValues)) if target in self.partsValues[x]["targets"]])
            cntTrain=np.mean([self.partsValues[x]["targets"][target][4] for x in range(len(self.partsValues)) if target in self.partsValues[x]["targets"]])
            
            crossValScore["targets"][target]={
                    "p":p,
                    "r":r,
                    "f":f,
                    "s":s,
                    "cntTrain":cntTrain
                }


        for i, hierAcc in enumerate(self.partsValues[0]["accNBest"]):
            crossValScore["nbest"][i+1]=[]
            for iHier, name in enumerate(hierAcc):
                name=name[0]
                
                valArMe=np.mean([self.partsValues[x]["accNBest"][i][iHier][1] for x in range(len(self.partsValues))])
                crossValScore["nbest"][i+1].append((name, valArMe))

        for i,avgName in enumerate(self.partsValues[0]["prf"]):
            avgName=avgName[0]
            crossValScore["avg"].append((avgName, {
                    "p":np.mean([self.partsValues[x]["prf"][i][1][0] for x in range(len(self.partsValues))]),
                    "r":np.mean([self.partsValues[x]["prf"][i][1][1] for x in range(len(self.partsValues))]),
                    "f":np.mean([self.partsValues[x]["prf"][i][1][2] for x in range(len(self.partsValues))])
                }))
            
            
        return crossValScore    
        
        
    def countAccuracy(self, predicted, targets, targetsNames, hier=False):
        """
        Spočítá accuracy.
        
        :param predicted: Jednotlivé jistoty příslušnosti dokumentů do dané kategorie.
            Pokud targetsNames je [], pak se jedná pouze o list s cíli.
        :param targets: Cíle dokumentů.
        :param targetsNames: list -- Názvy cílů. Pořadí odpovídá pořadí jistot v predicted. 
        :param hier: True vypočítá i metriky zavíslé na hierarchii.
        """
        getBest=1
        if targetsNames!=[]:
            getBest=len(targetsNames) if self.nBest > len(targetsNames) else self.nBest
            
        okCnt=[[0,0,0,0] for x in range(0, getBest)]

        
        for p,t in zip(predicted, targets):
            if targetsNames!=[]:
                sortedCls=[sX[0] for sX in sorted(zip(targetsNames, p), key=lambda x: x[1],reverse=True)]
            else:
                sortedCls=[p]
            
            sortSlice=[]
            for x in range(0,getBest):
                sortSlice.append(sortedCls[x])
                
                if t in sortSlice:
                    okCnt[x][0]=okCnt[x][0]+1
                    okCnt[x][1]=okCnt[x][1]+1  
                    okCnt[x][2]=okCnt[x][2]+1  
                    okCnt[x][3]=okCnt[x][3]+1
                elif hier and any([self.hier.inCloseFamily(xC, t) for xC in sortSlice]):
                    okCnt[x][1]=okCnt[x][1]+1  
                    okCnt[x][2]=okCnt[x][2]+1  
                    okCnt[x][3]=okCnt[x][3]+1
                elif hier and any([self.hier.inFamily(xC, t) for xC in sortSlice]):
                    okCnt[x][2]=okCnt[x][2]+1    
                    okCnt[x][3]=okCnt[x][3]+1
                elif hier and any([self.hier.sameTopLevel(xC, t) for xC in sortSlice]):
                    okCnt[x][3]=okCnt[x][3]+1
        
        retVals=[]
        for x in okCnt:
            if hier:
                retVals.append([
                ("exact match", x[0]/len(targets)),
                ("direct family", x[1]/len(targets)),
                ("family", x[2]/len(targets)),
                ("same top level", x[3]/len(targets)),
                ])
            else:
                retVals.append([("exact match", x[0]/len(targets))])

        return retVals
    
    def __unclassifiedFilter(self, predicted, predictedProba, targets):
        """
        Odfiltruje neklasifikovaná data (a jejich cíle) a počet neklasifikovaných dat příčte k self.unclassified.
        
        :param predicted: Predikované cíle
        :param predictedProba: Predikované pravděpodobnosti k jednotlivým cílům.
        :param targets: Pravé cíle.
        :returns: odfiltrované(predikované cíle, pravdepodobnosti, prave cíle)
        """

        newData=[]
        newPredictedProba=[]
        newTargets=[]
        
        for p,pb, t  in zip(predicted, predictedProba, targets):
            if p=="":
                self.unclassified+=1
                
            else:
                newData.append(p)
                newPredictedProba.append(pb)
                newTargets.append(t)
        
        return (newData, newPredictedProba, newTargets)
    
    @staticmethod   
    def selectBest(predicted, targetsNames=[]):
        """
        Vybere predikovaný cíl dokumentu s největší jistotou. Pokud targetsNames [] pak se předpkládá, že již predicted tento cíl přímo obsahuje.
        
        :param predicted: Jistoty k jednotlivým cílům ke každému dokumentu. 
            Nebo přímo List s názvy predikovaných cílů k dokumentům.
        :param targetsNames: list -- Názvy cílů. Pořadí odpovídá pořadí jistot v predicted.
        """
        if targetsNames==[]:
            return predicted
        
        tmpPred=predicted.tolist() if isinstance(predicted, np.ndarray) else predicted
        
        res=[]
        for p in tmpPred:
            maxP=max(p)
            if maxP==0:
                #nepovedlo se klasifikovat
                res.append("")
            else:
                res.append(targetsNames[p.index(maxP)])
        return res
        
    @staticmethod
    def writeConfMatTo(predicted, targetTest, writeTo):
        """
        Zapíše matici záměn do souboru.
        
        :param predicted: list -- s názvy predikovaných cílů k dokumentům.
        :param targetTest: list -- Pravé cíle.
        :param writeTo: Otevřený soubor pro zápis matice.
        """
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', len(max(predicted, key=len)) if len(max(predicted, key=len))>len(max(targetTest, key=len)) else len(max(targetTest, key=len)))
        
        writeTo.write(str(pd.crosstab(pd.Series(targetTest), pd.Series(predicted), rownames=['Pravé'], colnames=['Predikované'], margins=True)))
        writeTo.write("\n")
                    
            
    @staticmethod
    def matchPredictTarget(predicted, targets):
        """
        Vytvoří list, se značkami 1 -predikovaný a pravý cíl odpovídají nebo 0 neodpovídají.
        
        :param predicted: list -- s názvy predikovaných cílů k dokumentům.
        :param targets: list -- Pravé cíle.
        :result: list -- se značkami.
        """
     
        return [ [1] if p==t else [0] for p,t in zip(predicted, targets)]
    