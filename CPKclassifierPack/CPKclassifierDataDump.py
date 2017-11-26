# -*- coding: UTF-8 -*-
"""
Obsahuje třídu pro uložení a načtení dat pro CPKclassifier.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import logging
import os.path

from sklearn.externals import joblib

from .features.Features import Features
from .classification.Classification import Classification

class CPKclassifierDataDumpInvalidFile(Exception):
    """
    Vyjímka při čtení datového souboru.
    """
    pass


class CPKclassifierDataDump(object):
    """
    Třída pro načtení uložení dat z/do souboru.
    """

    featuresToolExtension=".ft"
    extractedFeaturesExtension=".ef"
    classificatorExtension=".cls"
    
    def __init__(self, targets=None, featuresTool=None, features=None, classificator=None, configFea=None, configCls=None):
        """
        Inicializuje objekt pro ukládání a načítání dat.

        :param targets: Cíle pro trénování.
        :param featuresTool: Nástroj pro extrakci příznaků.
        :param features: Extrahované příznaky.
        :param classificator: Klasifikátor.
        :param configFea: Konfigurace programu použitá pro extrahování příznaků.
        :param configCls: Konfigurace programu použitá pro trénování klasifikátoru.
        """
        self.targets=targets
        self.featuresTool=featuresTool
        self.features=features
        self.classificator=classificator
        self.configFea=configFea
        self.configCls=configCls
        
    def addTagets(self, targets):
        """
        Přidává cíle pro uložení.
        
        :param targets: Cíle, které budou uloženy.
        """
        
        self.targets=targets
    def addFeatures(self, features):
        """
        Přikládá příznaky pro uložení.
        
        :param features: Extrahované příznaky.
        """
        self.features=features
        
    def addFeaturesTool(self, featuresTool):
        """
        Přikládá nástroj pro extrakci příznaků.
        
        :param featuresTool: Nástroj pro extrakci příznaků.
        """
        self.featuresTool=featuresTool
    
    def addClassificator(self, classificator):
        """
        Přidává klasifikátor pro uložení.
        
        :param classificator: Klasifikátor.
        """
        self.classificator=classificator
        
    def addConfigFea(self, config):
        """ 
        Přidává programovou konfiguraci. Konfigurace programu použitá pro extrahování příznaků.
        
        :param config: Konfigurace pro uložení.
        """
        
        self.configFea=config
        
    def addConfigCls(self, config):
        """
        Přidává programovou konfiguraci. Konfigurace programu použitá pro trénování klasifikátoru.
        
        :param config: Konfigurace pro uložení.
        """
        
        self.configCls=config
        
        
    def save(self, fileName):
        """
        Uloží objekt do souboru.

        :param fileName: Cesta kde bude objekt uložen.
        """
        logging.info("začátek ukládání dat")
        self.saveBasicData(fileName)
        self.saveExtractedFeatures(fileName)
        self.saveFeaturesTool(fileName)
        self.saveClassificator(fileName)
        logging.info("konec ukládání dat")
        
        
    def saveBasicData(self, filename):
        """
        Uložení konfigurací, cílů a ID dokumentů do souboru.

        :param fileName: Cesta kde budou data uložena.
        """
        if self.configFea is not None or self.configCls is not None or self.targets is not None:
            logging.info("začátek ukládání konfigurací a cílů ")
            joblib.dump({
                "configFea":self.configFea,
                "configCls":self.configCls,
                "targets":self.targets
                }, filename)
            logging.info("konec ukládání konfigurací a cílů")
    
    def saveExtractedFeatures(self, filename):
        """
        Uložení extrahovaných příznaků do souboru.

        :param fileName: Cesta kde budou data uložena. Dodatečná přípona s druhem dat bude automaticky přidána.
        """
        if self.features is not None:
            logging.info("začátek ukládání extrahovaných příznaků")
            joblib.dump({"extFeat":self.features}, filename+self.extractedFeaturesExtension)
            logging.info("konec ukládání extrahovaných příznaků")
        
    def saveFeaturesTool(self, filename):
        """
        Uložení nástroje pro extrahovaní příznaků do souboru.

        :param fileName: Cesta kde budou data uložena. Dodatečná přípona s druhem dat bude automaticky přidána.
        """
        
        if self.featuresTool is not None:
            logging.info("začátek ukládání nástroje pro extrahovaní příznaků")
            joblib.dump(self.featuresTool, filename+self.featuresToolExtension)
            logging.info("konec ukládání nástroje pro extrahovaní příznaků")
        
    def saveClassificator(self, filename):
        """
        Uložení klasifikátoru do souboru.

        :param fileName: Cesta kde budou data uložena. Dodatečná přípona s druhem dat bude automaticky přidána.
        """
        if self.classificator is not None:
            logging.info("začátek ukládání klasifikátoru")
            joblib.dump(self.classificator, filename+self.classificatorExtension)
            logging.info("konec ukládání klasifikátoru")
    
    def load(self, filename):
        """
        Načtení CPKclassifierDataDump ze souboru.
        
        :param filename: Cesta k souboru ze kterého bude objekt načten.
        :raises: CPKclassifierDataDumpInvalidFile při nevalidním souboru.
        """
        logging.info("začátek načítání dat")
        
        self.loadBasicData(filename)
        self.loadExtractedFeatures(filename)
        self.loadFeaturesTool(filename)
        self.loadClassificator(filename)

        logging.info("konec načítání dat")
        
    def loadBasicData(self, filename):
        """
        Načtení konfigurací a cílů.
        
        :param filename: Cesta k souboru ze kterého bude načítáno.
        :raises: CPKclassifierDataDumpInvalidFile při nevalidním souboru.
        """
        if os.path.isfile(filename):
            logging.info("začátek načítání konfiguracía a cílů")
            try:
                lData=joblib.load(filename)
            except:
                raise CPKclassifierDataDumpInvalidFile()
            if not isinstance(lData, dict) or "configFea" not in lData or "configCls" not in lData or "targets" not in lData:
                raise CPKclassifierDataDumpInvalidFile()
    
            self.configFea=lData["configFea"]
            self.configCls=lData["configCls"]
            self.targets=lData["targets"]
            logging.info("konec načítání konfigurací a cílů")
        
    def loadExtractedFeatures(self, filename):
        """
        Načtení extrahovaných příznaků ze souboru.
        
        :param filename: Cesta k souboru ze kterého bude načítáno. 
            POZOR pouze základni jméno bez dodatečné přípony odlišující tento druh dat.
        :raises: CPKclassifierDataDumpInvalidFile při nevalidním souboru.
        """
        if os.path.isfile(filename+self.extractedFeaturesExtension):
            logging.info("začátek načítání extrahovaných příznaků")
            try:
                lData=joblib.load(filename+self.extractedFeaturesExtension)
            except:
                raise CPKclassifierDataDumpInvalidFile()
            if not isinstance(lData, dict) or "extFeat" not in lData:
                raise CPKclassifierDataDumpInvalidFile()
            
            self.features=lData["extFeat"]
            
            logging.info("konec načítání extrahovaných příznaků")
        
    def loadFeaturesTool(self, filename):    
        """
        Načtení nástroje pro extrahovaní příznaků ze souboru.
        
        :param filename: Cesta k souboru ze kterého bude načítáno. 
            POZOR pouze základni jméno bez dodatečné přípony odlišující tento druh dat.
        :raises: CPKclassifierDataDumpInvalidFile při nevalidním souboru.
        """
        if os.path.isfile(filename+self.featuresToolExtension):
            logging.info("začátek načítání nástroje pro extrahovaní příznaků")
            try:
                lData=joblib.load(filename+self.featuresToolExtension)
            except:
                raise CPKclassifierDataDumpInvalidFile()
            if not isinstance(lData, Features):
                raise CPKclassifierDataDumpInvalidFile()
            
            self.featuresTool=lData
            
            logging.info("konec načítání nástroje pro extrahovaní příznaků")
        
        
    def loadClassificator(self, filename):
        """
        Načtení klasifikátoru ze souboru.
        
        :param filename: Cesta k souboru ze kterého bude načítáno. 
            POZOR pouze základni jméno bez dodatečné přípony odlišující tento druh dat.
        :raises: CPKclassifierDataDumpInvalidFile při nevalidním souboru.
        """
        if os.path.isfile(filename+self.classificatorExtension):
            logging.info("začátek načítání klasifikátoru")
            try:
                lData=joblib.load(filename+self.classificatorExtension)
            except:
                raise CPKclassifierDataDumpInvalidFile()
            if not isinstance(lData, Classification):
                raise CPKclassifierDataDumpInvalidFile()
            self.classificator=lData
            logging.info("konec načítání klasifikátoru")
        
    @classmethod
    def loadFrom(cls, filename):
        """
        Načte CPKclassifierDataDump ze souboru. (Vytvoří nový objekt)
        
        :param filename: Cesta k souboru ze kterého bude objekt načten.
        :returns:  CPKclassifierDataDump -- načtený ze souboru
        :raises: CPKclassifierDataDumpInvalidFile při nevalidním souboru.
        """
        logging.info("začátek načítání dat")
        return CPKclassifierDataDump().load(filename)
        logging.info("konec načítání dat")
     

        