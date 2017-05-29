# -*- coding: UTF-8 -*-
"""
Obsahuje třídy pro práci s cíly.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""
class TargetsTranslatorUnknownTarget(Exception):
    """
    Neznámí cíl.
    """
    pass

class TargetsTranslator(object):
    """
    Třída pro překlad cílů. Vhodné pro ušetření paměti.
    """
    
    def __init__(self):
        """
        Počáteční inicializace.
        """
        self.__map={}
        
    def translate(self, targets):
        """
        Přeloží cíle.
        
        :param targets: list -- obsahující cíle. Může být i string.
        :returns: list-- obsahující přeložené cíle. Nebo int.
        """
        translatedTargets=[]
        selFirst=False
        if isinstance(targets, str):
            selFirst=True
            targets=[targets]

        for tar in targets:
            if tar not in self.__map:
                self.__map[tar]=len(self.__map)
                
            translatedTargets.append(self.__map[tar])
                
        if selFirst:
            return translatedTargets[0]
        return translatedTargets
    
    def getOriginal(self, targets):
        """
        Přeloží cíle do jejich originální podoby.
        
        :param targets: list -- obsahující přeložené cíle. Může být i int.
        :returns: list-- obsahující cíle v originální podobě. Nebo str.
        :raises: TargetsTranslatorUnknownTarget
        """
        selFirst=False
        if isinstance(targets, int):
            selFirst=True
            targets=[targets]
            
        origTargets=[]
        for tar in targets:
            oTar=None
            for origTar, transTar in self.__map.items():
                if transTar==tar:
                    oTar=origTar
                    break
                    
            if oTar is None:
                raise TargetsTranslatorUnknownTarget()
            
            origTargets.append(oTar)
            
        if selFirst:
            return origTargets[0]
        return origTargets
        
        
        
class TargetsHier(object):
    """
    Třída pro získávání hierarchie z cílů.
    """
    
    def __init__(self, targets=None, hierDelimiter="->"):
        """
        Inicializace hierarchie.
        
        :param targets: list -- cílů dokumentů
            :example:
            Tři dokumenty:
            0    cat_1
            1    cat_2
            2    cat_1
            [cat_1, cat_2, cat_1]
        :param hierDelimiter: Řetězec, který odděluje úrovně hierarchie.
        """
        self.hierDelimiter=hierDelimiter
        if targets:
            self.loadHier(targets)
        
    def parseLevels(self, target):
        """
        Získání jednotlivých úrovní z řetězce cíle.
        Upozornění: Pracuje pouze s řetězcem. Nevaliduje se s pravou hierarchií.
        
        :param target:
            Cíl v hierarchickém formátu.
        :returns: list
        """
        return target.split(self.hierDelimiter)
        
    def loadHier(self, targets):
        """
        Vytvoří hierarchii z cílů dokumentů.
        
        :param targets: list -- s cíli dokumentů
            :example:
            Tři dokumenty:
            0    cat_1
            1    cat_2
            2    cat_1
            [cat_1, cat_2, cat_1]

        """
        
        self.hier={}
        #vytvoří hier dict
        for i, cat in enumerate(targets):
            levels=self.parseLevels(cat)
            actHier=self.hier
            for hierCat in levels:
                if hierCat not in actHier:
                    actHier[hierCat]=(set(), {})
                    
                actHier[hierCat][0].add(i)
                actHier=actHier[hierCat][1]

    
    def inFamily(self, first, second):
        """
        Určuje jestli dva cíle jsou ve stejné rodině.
        Ve stejné rodině jsou cíle, když jeden nebo druhý je v množině rodičů/dětí toho druhého (nebo dva stejné cíle.). 
        Upozornění: Porovnává pouze řetězce. Nevaliduje se s pravou hierarchií.
        
        :param first:
            Cíl v hierarchickém formátu.
        :param second:
            Cíl v hierarchickém formátu.
        :returns:  True => stejná rodina. False jinak.
        """

        for f, s in zip(self.parseLevels(first), self.parseLevels(second)):
            if f!=s:
                return False

        return True
        
    def inCloseFamily(self, first, second):
        """
        Určuje jestli dva cíle jsou ve stejné blízké rodině.
        Ve stejné blízké rodině jsou cíle, když jeden nebo druhý je přímý rodič/dětě toho druhého (nebo dva stejné cíle.). 
        Upozornění: Porovnává pouze řetězce. Nevaliduje se s pravou hierarchií.
        
        :param first:
            Cíl v hierarchickém formátu.
        :param second:
            Cíl v hierarchickém formátu.
        :returns:  True => stejná blízká rodina. False jinak.
        """
        firstLevels=self.parseLevels(first)
        secondLevels=self.parseLevels(second)
        
        if abs(len(firstLevels)-len(secondLevels))>1:
            #couldn't be direct parent/child
            return False
        
        for f, s in zip(firstLevels, secondLevels):
            if f!=s:
                return False

        return True
    
    def sameTopLevel(self, first, second):
        """
        Určuje jestli dva cíle jsou ve stejné nejvyšší úrovni.
        Upozornění: Porovnává pouze řetězce. Nevaliduje se s pravou hierarchií.
        
        :param first:
            Cíl v hierarchickém formátu
        :param second:
            Cíl v hierarchickém formátu
        :returns:  True => stejná nejvyšší úroveň. False jinak.
        """
        
        return self.getTopLevel(first)==self.getTopLevel(second)
    
    def getTopLevel(self, target):
        """
        Získá nejvyšší cíl z cíle.
        Upozornění: Pracuje pouze s řetězcem. Nevaliduje se s pravou hierarchií.
        
        :param target:
            Cíl v hierarchickém formátu
        :returns:  Nejvyšší úroveň z hierarchie daného cíle.
        """
        return self.parseLevels(target)[0]
        
    def printHier(self, hier, level=0, stats=True):
        """
        Vytiskne hierarchie v čitelném formátu.
        
        :param hier: Hierarchie pro tisk.
        :param level: Úroveň hierarchie.
        :param stats: True -> tiskne i počet dokumentů s daným cílem
        """
        
        for name, val in sorted(hier.items(), key=lambda x: len(x[1][0]), reverse=True):
            statStr=""
            if stats:
                statStr=" ("+str(len(val[0]))+")"
            print("\t"*level+name+statStr)
            self.printHier(val[1], level+1, stats)
        
    def getHier(self):
        """
        :returns:  dict -- s hierachií

        """
        return self.hier

        