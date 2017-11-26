# -*- coding: UTF-8 -*-
"""
Obsahuje nástroje pro testování klasifikátoru.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, LeaveOneOut, KFold

class SplitTestSet(object):
    """
    Třída pro rozdělení všech dat na trénovací a testovací množiny.
    """

    #Názvy metod. Vše malé znaky.
    splitMethods = ["stratifiedkfold", "stratifiedshufflesplit", "leaveoneout", "kfold"]
    
    
    stratifiedShuffleSplitName="stratifiedshufflesplit"
    stratifiedKFoldName="stratifiedkfold"
    leaveoneoutName="leaveoneout"
    kFoldName="kfold"
    
    
    def __init__(self, allData, targets, additionalData=None):
        """
        Inicializace. Implicitní metoda pro získání trénovací a testovací množiny
        je leave one out.
        
        :param allData: dict -- s druhy dat, kde každý druh obsahuje list s daty k dokumentům.
        :param targets: list -- s cíli dat.
        :param additionalData: dodatečná data
        """
        
        self.allData=allData
        self.targets=targets
        self.additionalData=additionalData
        self.method=LeaveOneOut()
        
    def numOfIteratations(self):
        """
        Počet iterací.
        :returns: Počet iterací.
        """
        if isinstance(self.method, LeaveOneOut):
            return len(self.allData[next(iter(self.allData))])
        
        return self.splits
        
    def __iter__(self):
        """
        Rozděluje množinu.
        
        :returns: (trénovací data, trénovací cíle, testovací data, testovací cíle, dodatečná data k testovací množině)
        """
        
        for trainIndex, testIndex in self.method.split(self.allData[next(iter(self.allData))], self.targets):
            trainData={}
            testData={}
            
            
            trainTargets=[self.targets[i] for i in trainIndex]
            testTargets=[self.targets[i] for i in testIndex]

                
            for dataKind in  self.allData:
                
                trainData[dataKind]=[self.allData[dataKind][i] for i in trainIndex]
                testData[dataKind]=[self.allData[dataKind][i] for i in testIndex]
            
            testAddData={}
            
            if self.additionalData:
                for metaKind in self.additionalData:
                    testAddData[metaKind]=[]
                    for i in testIndex:
                        testAddData[metaKind].append(self.additionalData[metaKind][i]) 
            
            yield (trainData, trainTargets, testData, testTargets, testAddData)

        
    def useMethodSSS(self, splits, testSize):
        """
        Použije pro získání trénovací a testovací množiny stratified shuffle split.
        Zachová procento vzorku v každém cíli.
        
        :param splits: Počet rozdělovacích iterací.
        :param testSize: Velikost testovací množíny. Float => (0,1) procento z počtu. Int => počet.
        """
        
        self.splits=splits
        self.method=StratifiedShuffleSplit(n_splits=splits, test_size=testSize, random_state=0)

    def useMethodSKF(self, splits):
        """
        Použije pro získání trénovací a testovací množiny stratified k fold.
        Zachová procento vzorku v každém cíli.
        
        :param splits: Počet rozdělovacích iterací.
        :param testSize: Velikost testovací množíny. Float => (0,1) procento z počtu. Int => počet.
        """
        
        self.splits=splits
        self.method=StratifiedKFold(n_splits=splits, random_state=0)
        

    def useMethodKF(self, splits):
        """
        Použije pro získání trénovací a testovací množiny  k fold.

        :param splits: Počet rozdělovacích iterací.
        """
        
        self.splits=splits
        self.method=KFold(n_splits=splits, random_state=0)
        
        
    def useMethodLeaveOneOut(self):
        """
        Použije pro získání trénovací a testovací množiny leave one out. (implicitní nastavení)
        """
        
        self.method=LeaveOneOut()
        

    
