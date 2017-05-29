# -*- coding: UTF-8 -*-
"""
Modul pro vyvažovací metody.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

import random
import math

class BalancingInvalidBalancingMethod(Exception):
    """
    Nevalidní metoda pro balancování data setu.
    """
    pass


class Balancing(object):
    """
    Třída pro vyvažování nevyvážených dat.
    """
    
    #názvy metod pro vyvažování dat
    balancersNames=["randomundersampling", "randomoversampling"]
    
    
    randomUnderSampling="randomundersampling"
    
    randomOverSampling="randomoversampling"
    
    def __init__(self, samplers):
        """
        Nastavení metod pro vyvažování.
        
        :param samplers: list -- Obsahující metody vyvažování.
            Budou použity v pořadí v jakém jsou v listu uvedeny.
            Metodu reprezentuje (název metody, parametry)
        :raise BalancingInvalidBalancingMethod: Při nevalidním názvu balancovací metody.
        """
        self.samplers=[]
        for balancerPar in samplers:
            balancerPar[1]["randState"]=0

            if self.randomUnderSampling==balancerPar[0]:
                self.samplers.append(RandomUnderSampling(**balancerPar[1]))
            elif self.randomOverSampling==balancerPar[0]:
                self.samplers.append(RandomOverSampling(**balancerPar[1]))
            else:
                raise BalancingInvalidBalancingMethod()


    
    def balance(self, data, targets):
        """
        Vyvažování dat.
        
        :param data: dict -- Obsahující data. Klíč je název druhu dat.
        :param targets: Cíle pro trénování.
        :returns: (data, targets) vybalancované
        
        """
        
        balancedDataAll={}
        balancedTargets=targets
        
        for dataName in data:
            
            balancedData=data[dataName]
            balancedTargets=targets
            
            for sampler in self.samplers: 
                balancedData, balancedTargets= sampler.get(balancedData, balancedTargets)
                
            balancedDataAll[dataName]=balancedData
            
        
            
        return (balancedDataAll, balancedTargets)
    

class Sampler(object):
    """
    Rodičovská třída pro samplovací metody.
    """
    
    @staticmethod
    def targetsStats(targets):
        """
        Získá počty vzorků k jednotlivým cílům.
        
        :targets targets: Cíle
        :returns:  dict -- klíč -> cíl, hodnota -> list obsahující indexy.
        """
        
        stats={}
        
        for t in targets:
            if t not in stats:
                stats[t]=0
                
            stats[t]=stats[t]+1
            
        return stats
    
    @staticmethod
    def getDataForTargets(targets):
        """
        Získá vzorky dat k jednotlivým cílům.
        
        :targets targets: Cíle
        :returns:  dict -- klíč -> cíl, hodnota -> list obsahující indexy.
        """
        
        tarSampl={}
        
        for i, t in enumerate(targets):
            if t not in tarSampl:
                tarSampl[t]=[]
                
            tarSampl[t].append(i)
    
        return tarSampl
    
    
    def get(self, dataSamples, targets):
        """
        Vybalancuje a vrátí dataset.
        
        :param dataSamples: Vzorky dat.
        :param targets: Cíle dat.
        """
        
        raise NotImplementedError()

class RandomUnderSampling(Sampler):
    """
    Třída pro metodu Random Under-Sampling.  Náhodně vybere vzorky, které odstraní.
    """
    
    def __init__(self,  ratio, randState=None):
        """
        Nastavení metody Random Under-Sampling.
        
        :param ratio: Číslo v intervalu <0, 1>.
                Určuje kolik maximálně může být vzorků k jednomu cíli.
                Jedná se o číslo, které je vyjádřeno ratio*POCET_DOKUMENTU_V_NEJPOCETNEJSIM_CILI.
                Všechno nad tento limit bude odstraněno.
        :param randState: Semínko pro generátor náhodných čísel. None ->nebude použito.
        """
        
        self.ratio=ratio
        self.randState=randState
    
    def get(self, dataSamples, targets):
        """
        Vybalancuje a vrátí dataset.
        
        :param dataSamples: Vzorky dat.
        :param targets: list -- Cíle dat.
        :returns: Pair (dataSamples, targets)
        """
        
        stats=super().targetsStats(targets)
        
        maxNumOfSamp=math.ceil(max(stats.values())*self.ratio)
        
        sampInTar=super().getDataForTargets(targets)
        
        if self.randState is not None:
            random.seed(self.randState)
        
        resIndexes=[]
        
        for tar in sampInTar:
            random.shuffle(sampInTar[tar])
            resIndexes=resIndexes+sampInTar[tar][:maxNumOfSamp]
            
        
        random.shuffle(resIndexes)
        
        return (dataSamples[resIndexes], [targets[i] for i in resIndexes])
        
        
        
        

class RandomOverSampling(Sampler):
    """
    Třída pro metodu Random Over-Sampling. Náhodně vybere vzorky, které naklonuje.
    """
    

    def __init__(self,  ratio, randState=None):
        """
        Nastavení metody Random Over-Sampling.
        
        :param ratio: Určuje horní limit klonování dat.
            Jedná se o číslo, které je vyjádřeno ratio*POCET_DOKUMENTU_V_NEJPOCETNEJSIM_CILI.
        :param randState: Semínko pro generátor náhodných čísel.
        """
        
        self.ratio=ratio
        self.randState=randState
        
    
    def get(self, dataSamples, targets):
        """
        Vybalancuje a vrátí dataset.
        
        :param dataSamples: Vzorky dat.
        :param targets: list -- Cíle dat.
        :returns: Pair (dataSamples, targets)
        """
        
        stats=super().targetsStats(targets)
        
        numOfSamp=math.ceil(max(stats.values())*self.ratio)
        
        sampInTar=super().getDataForTargets(targets)
        
        if self.randState is not None:
            random.seed(self.randState)
        
        resIndexes=[]
        
        for tar in sampInTar:
            cloned=sampInTar[tar]
            
            while (len(cloned)<numOfSamp):
                cloned.append(random.choice(cloned))
                
            resIndexes=resIndexes+cloned
                
            
        
        random.shuffle(resIndexes)
        
        return (dataSamples[resIndexes], [targets[i] for i in resIndexes])
    
    
    
    