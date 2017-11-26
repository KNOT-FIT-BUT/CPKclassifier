#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
CPKclassifier -- Systém pro klasifikaci dokumentů.
Obsahuje také nástroje pro předzpracování, vyvažování a testování.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

import multiprocessing


import os
import sys
import logging
import traceback
import configparser
import re
import shlex
import copy


from shutil import copyfile
from argparse import ArgumentParser

from CPKclassifierPack.utils.DataSet import DataSet, DataSetInvalidMetadataFields, DataSetNoDataPath, DataSetInvalidTarget, DataSetInvalidDataFileForMetadata
from CPKclassifierPack.utils.DocReader import DocReaderMetadata, DocReaderInvalidMetadataFields, DocReaderNeedDataFile

from CPKclassifierPack.CPKclassifierDataDump import CPKclassifierDataDumpInvalidFile, CPKclassifierDataDump

from CPKclassifierPack.preprocessing.Preprocessing import Preprocessing, LemmatizerException, Lemmatizer
from CPKclassifierPack.features.Features import Features, FeaturesNoData
from CPKclassifierPack.classification.Classification import Classification
from CPKclassifierPack.balancing.Balancing import Balancing
from CPKclassifierPack.prediction.Prediction import Prediction
from CPKclassifierPack.testing.SplitTestSet import SplitTestSet
from CPKclassifierPack.testing.Testing import Testing


    
class ExceptionMessageCode(Exception):
    """
    Vyjímka se zprávou a kódem.
    """
    def __init__(self, message, code):
        """
        Vytvoření vyjímky se zprávou a kódem.
        
        :param message: Chybová zpráva.
        :param code: Chybový kód..
        """
        self.message = message
        self.code = code
    
class ErrorMessenger:
    """
    Obstarává chybové kódy a jejich příslušné zprávy.
    Píše zprávy do stderr a ukončí skript s definovaným kódem.
    """
    
    CODE_ALL_OK=0;
    CODE_INVALID_ARGUMENTS=1
    CODE_COULDNT_WORK_WITH_FILE=2
    CODE_COULDNT_READ_INPUT_FILE=3
    CODE_COULDNT_OPEN_OUTPUT_FILE_FOR_WRITE=4
    CODE_INVALID_CONFIG=5
    CODE_INVALID_INPUT_FILE=6
    CODE_NO_INPUT_DATA=7
    CODE_INV_COMB_EXT_FEAT_METHOD_AND_CLS=8
    CODE_UNKNOWN_ERROR=100
    
    """Obsahuje chybové zprávy. Indexy odpovídají chybovým kódům."""
    __ERROR_MESSAGES={
            CODE_ALL_OK:"Vše v pořádku.",
            CODE_INVALID_ARGUMENTS:"Navalidní argumenty.",
            CODE_COULDNT_WORK_WITH_FILE: "Nemohu pracovat se souborem.",
            CODE_COULDNT_READ_INPUT_FILE:"Nemohu číst vstupní soubor.",
            CODE_COULDNT_OPEN_OUTPUT_FILE_FOR_WRITE:"Nemohu otevřít výstupní soubor pro zápis.",
            CODE_INVALID_CONFIG:"Nevalidní hodnota v konfiguračním souboru.",
            CODE_INVALID_INPUT_FILE:"Nevalidní vstupní soubor.",
            CODE_NO_INPUT_DATA:"Žádná vstupní data.",
            CODE_INV_COMB_EXT_FEAT_METHOD_AND_CLS:"Nevalidní kombinace metody pro extrakci příznaků a klasifikátoru.",
            CODE_UNKNOWN_ERROR:"Nastala chyba."  ,
    }

    @staticmethod
    def echoError(message, code):
        """
        Vypisuje chybovou zprávu do stderr a ukončuje skript s daným kódem.
        
        :param message: Text chybové zprávy.
        :param code: Ukončovací kód.
        """
        print(message, file=sys.stderr)
        sys.exit(code);
        
    @classmethod
    def getMessage(cls, code):
        """
        Vrátí text chybové zprávy definovanou kódem.
        
        :param cls: Class
        :param code: Kód chybové zprávy.
        """
        return cls.__ERROR_MESSAGES[code];
    
class ArgumentParserError(Exception): pass
class ExceptionsArgumentParser(ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)
    
class ArgumentsManager:
    """
    Manažér argumentů pro CPKclassifier.
    """
    
    @classmethod
    def parseArgs(cls, cpkClassifier):
        """
        Parsování argumentů a výpis chybové zprávy.
        
        :param cls: arguments
        :param cpkClassifier: Object třídy CPKclassifier.
        
        :returns: Zpracované argumenty.
        """
        
        parser = ExceptionsArgumentParser(description="Nástroj pro klasifikaci dokumentů, hledání podobných dokumentů a předzpracování dokumentů. Dodatečné parametry lze nastavit v konfiguračním souboru.")
        
        subparsers = parser.add_subparsers()
        
        parserPreprocessing = subparsers.add_parser('preprocessing',help='Nástroje pro předzpracování textu.')
        
        parserPreprocessing.add_argument("--lemmatize", action='store_true',
                help="Provede lemmatizaci textu. Odděluje také znaky od slov.")
        parserPreprocessing.add_argument("--noSW", action='store_true',
                help="Odstraní stopslova.")
        parserPreprocessing.add_argument("--unidecode", action='store_true',
                help="Přeloží unicode znaky do jejich nejbližší možné reprezentace v ascii.")
                
        parserPreprocessing.add_argument("--sepSigns", action='store_true',
                help="Oddělení znaků od slov. (např: : ,.:;?!). Není nutné používat v kombinaci s lemmatize a pos.")

        parserPreprocessing.add_argument("--minWordLength", type=int,
                help="Vezme pouze slova, která mají počet znaků větší nebo roven MINWORDLENGTH (po případné lemmatizaci/extrakci/sepSigns).")
        
        parserPreprocessing.add_argument("--maxWordLength", type=int,
                help="Vezme pouze slova, která mají počet znaků menší nebo roven MAXWORDLENGTH (po případné lemmatizaci/extrakci/sepSigns).")
        
        parserPreprocessing.add_argument("--pos", nargs='+',
                help="Extrahuje definované slovní druhy. Slovní druhy: 1,2,3,4,5,6,7,8,9,10,Z(symboly),X(neznámé). Odděluje také znaky od slov.")
        
        parserPreprocessingCase = parserPreprocessing.add_mutually_exclusive_group()
        parserPreprocessingCase.add_argument("--uc", action='store_true',
                help="Převedení znaků na velké znaky.")
        parserPreprocessingCase.add_argument("--lc", action='store_true',
                help="Převedení znaků na malé znaky.")
        
        parserPreprocessing.add_argument("--input", type=str,
                help="Vstupní soubor s daty (povinné)", required=True)
        parserPreprocessing.add_argument("--config", type=str,
                help="Tento konfigurační soubor přenastaví parametry z defaultního konfiguračního souboru. (Pouze uvedené)")
        parserPreprocessing.add_argument("--log", type=str,
                help="Kam uložit logovací soubor.")
        parserPreprocessing.set_defaults(func=cpkClassifier.preprocessing)

        parserGetData = subparsers.add_parser('getData', 
                                               help='Výběr dat. Řídí se nastavením v konfiguračním souboru. Odstraňuje bílé znaky u plného textu a nahradí je jednou mezerou.')
        parserGetData.add_argument("--data", type=str,
                help="Vstupní datový soubor.")
        parserGetData.add_argument("--metadata", type=str,
                help="Vstupní metadatový soubor. (povinné)", required=True)
        parserGetData.add_argument("--saveDataTo", type=str,
                help="Cesta kam bude uložen výsledný datový soubor.")
        parserGetData.add_argument("--saveMetadataTo", type=str,
                help="Cesta kam bude uložen výsledný metadatový soubor.")
        parserGetData.add_argument("--config", type=str,
                help="Tento konfigurační soubor přenastaví parametry z defaultního konfiguračního souboru. (Pouze uvedené)")
        parserGetData.add_argument("--log", type=str,
                help="Kam uložit logovací soubor.")
        parserGetData.set_defaults(func=cpkClassifier.getDataArgs)

        parserFeatures = subparsers.add_parser('features', 
                                               help='Nástroje pro extrakci příznaků a utvoření souboru pro klasifikaci. Řídí se nastavením v konfiguračním souboru.')
        parserFeatures.add_argument("--data", type=str,
                help="Vstupní datový soubor.")
        parserFeatures.add_argument("--metadata", type=str,
                help="Vstupní metadatový soubor (povinné).", required=True)
        parserFeatures.add_argument("--saveTo", type=str,
                help="Cesta kam bude uložen výsledek (povinné).", required=True)
        parserFeatures.add_argument("--config", type=str,
                help="Tento konfigurační soubor přenastaví parametry z defaultního konfiguračního souboru. (Pouze uvedené)")
        parserFeatures.add_argument("--log", type=str,
                help="Kam uložit logovací soubor.")
        parserFeatures.set_defaults(func=cpkClassifier.featuresExtracting)
        
        
        parserClassification = subparsers.add_parser('classification', help='Trénování klasifikátoru. Řídí se nastavením v konfiguračním souboru.')

        parserClassification.add_argument("--features", type=str,
                help="Soubor s extrahovanými příznaky. Výstup z features. Popřípadě i z classification, pokud obsahuje příznaky (features).")
        
        parserClassificationInputDataPlain = parserClassification.add_argument_group('Přímo vstupní data.')
        parserClassificationInputDataPlain.add_argument("--data", type=str,
                help="Vstupní datový soubor.")
        parserClassificationInputDataPlain.add_argument("--metadata", type=str,
                help="Vstupní metadatový soubor (povinné).")
        
        
        parserClassification.add_argument("--config", type=str,
                help="Tento konfigurační soubor přenastaví parametry z defaultního konfiguračního souboru. (Pouze uvedené)")
        parserClassification.add_argument("--log", type=str,
                help="Kam uložit logovací soubor.")
        parserClassification.add_argument("--saveTo", type=str,
                help="Cesta kam bude uložen výsledek (povinné).", required=True)
        parserClassification.set_defaults(func=cpkClassifier.classification)
        
        
        parserPredict = subparsers.add_parser('prediction', help='Predikce cílů. Požaduje druhy dat/metadat, které byly použity pro extrakci příznaků použitých při trénování klasifikátoru. Řídí se nastavením v konfiguračním souboru.')
        parserPredict.add_argument("--data", type=str,
                help="Vstupní datový soubor.")
        parserPredict.add_argument("--metadata", type=str,
                help="Vstupní metadatový soubor.",)
        parserPredict.add_argument("--classifiers", type=str,
                help="Cesta k souboru s uloženými klasifikátory/klasifikátorem.", required=True)
        parserPredict.add_argument("--config", type=str,
                help="Tento konfigurační soubor přenastaví parametry z defaultního konfiguračního souboru. (Pouze uvedené)")
        parserPredict.add_argument("--log", type=str,
                help="Kam uložit logovací soubor.")
        parserPredict.set_defaults(func=cpkClassifier.predict)
        
        
        parserTesting = subparsers.add_parser('testing', help='Testování klasifikátoru. Řídí se nastavením v konfiguračním souboru. Statistiku píše do stdout.')
        parserTesting.add_argument("--data", type=str,
                help="Vstupní datový soubor.")
        parserTesting.add_argument("--metadata", type=str,
                help="Vstupní metadatový soubor. (povinné)", required=True)
        
        parserTesting.add_argument("--writeResults", type=str,
                help="Cesta k souboru, kde má být uložen výsledek predikcí v jednotlivích iteracích. Nastavení na základě sekce PREDICTION v konfiguračním souboru.")
        
        parserTesting.add_argument("--sepResults", action='store_true',
                help="Rozdělí výsledky klasifikace v jednotlivých křížově validačních krocích do více souborů. K názvu souboru z parametru writeResults bude přidáno číslo kroku.")
        
        
        parserTesting.add_argument("--writeConfMetrix", type=str,
                help="Cesta k souboru kde budou uloženy matice záměn.")
        
        parserTesting.add_argument("--consistency", action='store_true',
                help="Natrénuje na všech dostupných datech a na těchto datech i natrénovaný klasifikátor otestuje.")
        
        parserTesting.add_argument("--config", type=str,
                help="Tento konfigurační soubor přenastaví parametry z defaultního konfiguračního souboru. (Pouze uvedené)")
        parserTesting.add_argument("--log", type=str,
                help="Kam uložit logovací soubor.")
        parserTesting.set_defaults(func=cpkClassifier.testing)
        
        
        parserStats = subparsers.add_parser('stats', help='Získávání statistik ze souborů v metadatovém formátu. Nastavení bere z konfiguračního souboru. Filtrování lze nastavit pomocí sekce GET_DATA. Názvy polí predikovaných a pravých cílů jsou získány ze sekcí PREDICTION a GET_DATA')
        parserStats.add_argument("--input", type=str,
                help="Vstupní soubor.", required=True)
        parserStats.add_argument("--config", type=str,
                help="Tento konfigurační soubor přenastaví parametry z defaultního konfiguračního souboru. (Pouze uvedené)")
        parserStats.set_defaults(func=cpkClassifier.stats)
        parserStats.add_argument("--log", type=str,
                help="Kam uložit logovací soubor.")
        
        
        subparsersForHelp={
            'preprocessing':parserPreprocessing,
            'getData':parserGetData,
            'features':parserFeatures,
            'classification':parserClassification,
            'prediction':parserPredict,
            'testing':parserTesting,
            'stats':parserStats
            }
        
    

        if len(sys.argv)<2:
            parser.print_help()
            return None
        try:
            parsed=parser.parse_args()
            __class__.validateArguments(parsed)
            
        except ArgumentParserError as e:
            for name, subParser in subparsersForHelp.items():  
                if name==sys.argv[1]:
                    print(str(e))
                    subParser.print_help()
                    break
            
            ErrorMessenger.echoError(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS), ErrorMessenger.CODE_INVALID_ARGUMENTS)

        return parsed

    @staticmethod
    def validateArguments(parsed):
        """
        Validuje naparsované argumenty.
        
        :param parsed: Naparsované argumenty.
        """
        
        if hasattr(parsed, 'uc') and parsed.uc and  hasattr(parsed, 'lc') and parsed.lc:
            raise ArgumentParserError("Navalidní kombinace argumentů.")
            
        if hasattr(parsed, 'minWordLength') and parsed.minWordLength and  hasattr(parsed, 'maxWordLength') and parsed.maxWordLength \
            and parsed.minWordLength>parsed.maxWordLength:
            raise ArgumentParserError("maxWordLength nemůže být menší než minWordLength.")
        
        
        if 'classification'==sys.argv[1]:
            
            if (hasattr(parsed, 'features') and parsed.features is not None ) and \
                ((hasattr(parsed, 'data') and parsed.data is not None) or (hasattr(parsed, 'metadata')  and parsed.metadata is not None)):
                raise ArgumentParserError("Parametr features nelze kombinovat s data a metadata.")
            
            if (hasattr(parsed, 'features') and parsed.features is None) and (hasattr(parsed, 'metadata') and parsed.metadata is None):
                raise ArgumentParserError("Musíte uvést parametr features nebo metadata.")
            
        if 'preprocessing'==sys.argv[1]:
            if hasattr(parsed, 'pos') and parsed.pos and any([ x not in Preprocessing.posSigns for x in parsed.pos]):
                raise ArgumentParserError("Neznámý výběr slovního druhu.")
            
        if "testing"==sys.argv[1]:
            if hasattr(parsed, 'sepResults') and parsed.sepResults is not False and (not hasattr(parsed, 'writeResults') or  parsed.writeResults is None):
                raise ArgumentParserError("Při použití parametru sepResults je nutné uvést i parametr writeResults.")
                

class ConfigManager(object):
    """
    Tato třída slouží pro načítání konfigurace z konfiguračního souboru.
    """
    
    #vyhrazený název pro fulltext
    fulltextName="fulltext"
    
    sectionDefault="DEFAULT"
    sectionPreprocessing="PREPROCESSING"
    sectionGetData="GET_DATA"
    sectionFeatures="FEATURES"
    sectionClassification="CLASSIFICATION"
    sectionPredict="PREDICTION"
    sectionTesting="TESTING"
    sectionDOC2VEC="DOC2VEC"
    sectionHashingVectorizer="HASHING_VECTORIZER"
    sectionKNeighborsClassifier="K_NEIGHBORS_CLASSIFIER"
    sectionSGDClassifier="SGD_CLASSIFIER"
    
    
    
    
    def __init__(self):
        """
        Inicializace config manažéru.
        """
        
        self.configParser = configparser.ConfigParser()
    
        
    def read(self, filesPaths):
        """
        Přečte hodnoty z konfiguračních souborů. Také je validuje a převede do jejich datových typů.
        
        :param filesPaths: list s cestami ke konfiguračním souborům.
        :returns: Konfigurace.
        """
        try:
            self.configParser.read(filesPaths)
        except configparser.ParsingError as e:
            raise ExceptionMessageCode("Nevalidní konfigurační soubor: "+str(e),
                                       ErrorMessenger.CODE_INVALID_CONFIG)
        except:
            raise ExceptionMessageCode("Nevalidní konfigurační soubor.",
                                       ErrorMessenger.CODE_INVALID_CONFIG)
        
        return self.__transformVals()
        
        
    def __transformVals(self):
        """
        Převede hodnoty a validuje je.
        
        :returns: dict -- ve formátu jméno sekce jako klíč a k němu dict s hodnotami.
        """
        result={}

        result[self.sectionDefault]=self.__transformDefaultVals()
        result[self.sectionPreprocessing]=self.__transformPreprocessingVals();
        result[self.sectionGetData]=self.__transformGetDataVals()
        result[self.sectionFeatures]=self.__transformFeaturesVals(result[self.sectionGetData])
        result[self.sectionDOC2VEC]=self.__transformDOC2VECVals()
        result[self.sectionHashingVectorizer]=self.__transformHashingVectorizerVals()
        result[self.sectionClassification]=self.__transformClassificationVals(result[self.sectionGetData])
        result[self.sectionPredict]=self.__transformPredictVals()
        result[self.sectionTesting]=self.__transformTestingVals()
        result[self.sectionKNeighborsClassifier]=self.__transformKNeighborsClassifierVals()
        result[self.sectionSGDClassifier]=self.__transformSGDClassifierVals()
        
        return result
    
    def __transformDefaultVals(self):
        """
        Převede hodnoty pro DEFAULT a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        default=self.configParser[self.sectionDefault]
        result={
            "HIER_DELIMITER":"->"
            }
        
        if default["HIER_DELIMITER"]:
            result["HIER_DELIMITER"]=shlex.split(default["HIER_DELIMITER"])[0]
        
        return result
        
    
    def __transformPreprocessingVals(self):
        """
        Převede hodnoty pro předzpracování a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        preprocessing=self.configParser[self.sectionPreprocessing]
        result={
            "DICT":None,
            "WORKERS": 1,
            "STOP_WORDS":None,
            "MAX_NUMBER_OF_WORDS_PER_LINE_PART":100000
            }
        
        if preprocessing["DICT"]: 
            
            if preprocessing["DICT"][0]!="/":
                result["DICT"]=os.path.dirname(os.path.realpath(__file__))+"/"+preprocessing["DICT"]
            else:
                result["DICT"]=preprocessing["DICT"]
                
            if not os.path.isfile(result["DICT"]):
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionPreprocessing+\
                                           " v parametru DICT není cesta k existujícímu souboru.",
                                               ErrorMessenger.CODE_INVALID_CONFIG)
                
           
                
        if preprocessing["WORKERS"]:
            
            try:
                result["WORKERS"]=int(preprocessing["WORKERS"])
                if result["WORKERS"]<1 and result["WORKERS"]!=-1:
                    raise ValueError()
            
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci "+self.sectionPreprocessing+" u parametru: WORKERS",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            
            
        if preprocessing["STOP_WORDS"]:
            result["STOP_WORDS"]=shlex.split(preprocessing["STOP_WORDS"])
        
        if preprocessing["MAX_NUMBER_OF_WORDS_PER_LINE_PART"]:
            
            try:
                result["MAX_NUMBER_OF_WORDS_PER_LINE_PART"]=int(preprocessing["MAX_NUMBER_OF_WORDS_PER_LINE_PART"])
                if result["MAX_NUMBER_OF_WORDS_PER_LINE_PART"]<1:
                    raise ValueError()
            
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci "+self.sectionPreprocessing+" u parametru: MAX_NUMBER_OF_WORDS_PER_LINE_PART",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            
        return result
    
    def __transformGetDataVals(self):
        """
        Převede hodnoty pro sekci GET_DATA a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        result={
            "TARGET_FIELD":"TARGET",
            "ITEM_DELIMITER": None,
            "COPY":{}
            }
        getData=self.configParser[self.sectionGetData]
        
        result["LAZY_EVAL_DATA"]=getData["LAZY_EVAL_DATA"].lower()=="true"
        result["NON_EMPTY"]=shlex.split(getData["NON_EMPTY"])
        result["EMPTY"]=shlex.split(getData["EMPTY"])
        
        if getData["TARGET_FIELD"]:
            result["TARGET_FIELD"]=shlex.split(getData["TARGET_FIELD"])[0]
        
        
        result["FIELD_REGEX"]=self.__createDict(getData["FIELD_REGEX"])
        
        if result["FIELD_REGEX"] is not None:
            #validace
            for k, value in result["FIELD_REGEX"].items():
                
                try:
                    #zkusime jestli je v poradku
                    re.compile(value)
                except re.error:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. Nevalidní hodnota regulárního výrazu ve FIELD_REGEX "+k+":"+value,
                                               ErrorMessenger.CODE_INVALID_CONFIG)
                    
                    
                result["FIELD_REGEX"][k]=(value,0)
        else:
            result["FIELD_REGEX"]={}
        
        result["MIN_PER_FIELD"]=self.__createDict(getData["MIN_PER_FIELD"], numValues="int")
        result["MAX_PER_FIELD"]=self.__createDict(getData["MAX_PER_FIELD"], numValues="float")

        if getData["ITEM_DELIMITER"]:
            result["ITEM_DELIMITER"]=shlex.split(getData["ITEM_DELIMITER"])[0]
        
        result["SELECT_WORDS"]=self.__parseSelectorVal(getData["SELECT_WORDS"])
        result["SELECT_ITEMS"]=self.__createDict(getData["SELECT_ITEMS"], convertSelectors=True)
        
        
        result["GET_FULLTEXT"]=getData["GET_FULLTEXT"].lower()=="true"
        
        result["GET_META_FIELDS"]=shlex.split(getData["GET_META_FIELDS"])

        if result["GET_META_FIELDS"]:
 
            for i, x in enumerate(result["GET_META_FIELDS"]):
                parts=x.split(":")
                if len(parts)>2:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. GET_META_FIELDS obsahuje nevalidní hodnotu.",
                                       ErrorMessenger.CODE_INVALID_CONFIG)
                if len(parts)==2:
                    if parts[1] in result["GET_META_FIELDS"]:
                        raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V GET_META_FIELDS nelze přejmenovat druh dat na název, který již existuje.",
                                       ErrorMessenger.CODE_INVALID_CONFIG)
                        
                    result["GET_META_FIELDS"][i]=parts[1]
                        
                    if tuple(parts[0].split("+")) not in result["COPY"]:
                        result["COPY"][tuple(parts[0].split("+"))]=[]

                    result["COPY"][tuple(parts[0].split("+"))].append(parts[1])
                    
                    
                    
        if self.fulltextName in result["GET_META_FIELDS"]:
            raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. GET_META_FIELDS nesmí obsahovat název "+self.fulltextName+", který je vyhrazen pouze pro data: "+getData["GET_META_FIELDS"],
                                       ErrorMessenger.CODE_INVALID_CONFIG)
            
        if len(result["GET_META_FIELDS"])!=len(set(result["GET_META_FIELDS"])):
            raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. Některá pole v GET_META_FIELDS se opakují: "+getData["GET_META_FIELDS"],
                                       ErrorMessenger.CODE_INVALID_CONFIG)

        
        return result
    
    def __transformFeaturesVals(self, transformedGetData):
        """
        Převede hodnoty pro sekci FEATURES a validuje je.
        
        :param transformedGetData: Pro validaci je nutné, poskytnout převedenou sekci get data.
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        
        result={
            "FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON": None,
            "META_VECTORIZERS_BUILD_VOCABULARY_ON": None,
            "SKIP_EMPTY":True,
            "WORKERS":1
            }
        
        features=self.configParser[self.sectionFeatures]
        
        
        if features["SKIP_EMPTY"]:
            result["SKIP_EMPTY"]=features["SKIP_EMPTY"].lower()=="true"

        if features["WORKERS"]:
            try:
                result["WORKERS"]=int(features["WORKERS"])
        
                if result["WORKERS"]<1 and result["WORKERS"]!=-1:
                    raise ValueError()
                
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci "+self.sectionFeatures+" u parametru: WORKERS",
                                       ErrorMessenger.CODE_INVALID_CONFIG)
        
        
        result["FULL_TEXT_VECTORIZER"]=None
        result["FULL_TEXT_ANALYZER"]=None
        result["META_VECTORIZERS"]=None
        result["META_ANALYZERS"]=None
        
        if transformedGetData["GET_FULLTEXT"]:
            result["FULL_TEXT_VECTORIZER"]=Features.hashingVectorizerName
            if features["FULL_TEXT_VECTORIZER"]:
                if features["FULL_TEXT_VECTORIZER"].lower() not in Features.vectorizersNames:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V poli FULL_TEXT_VECTORIZER: "+features["FULL_TEXT_VECTORIZER"],
                                           ErrorMessenger.CODE_INVALID_CONFIG)
                
    
                result["FULL_TEXT_VECTORIZER"]=features["FULL_TEXT_VECTORIZER"].lower() if features["FULL_TEXT_VECTORIZER"] else None
        
        
            
            if features["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"]:
                result["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"]=features["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"]
                    
            result["FULL_TEXT_ANALYZER"]=(Features.fulltextAnalyzersNames[0], 1)
            if features["FULL_TEXT_ANALYZER"]:
                
                if len(features["FULL_TEXT_ANALYZER"].split("/", 1))!=2:
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. Chybí parametr pro analyzátor ve FULL_TEXT_ANALYZER : "+features["FULL_TEXT_ANALYZER"],
                            ErrorMessenger.CODE_INVALID_CONFIG)
                    
                name, par=features["FULL_TEXT_ANALYZER"].lower().split("/", 1)
                    
                try:
                    par = int(par)
                    if par<1:
                        raise ValueError()
                except ValueError:
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. Nevhodný parametr pro analyzátor ve FULL_TEXT_ANALYZER (musí být celé číslo větší než 0): "+features["FULL_TEXT_ANALYZER"],
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
                if name not in Features.fulltextAnalyzersNames:
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. FULL_TEXT_ANALYZER obsahuje neznámé jméno analyzátoru. Použijte prosím následující: "+" ".join(Features.fulltextAnalyzersNames),
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
                result["FULL_TEXT_ANALYZER"]=(name, par)
        else:
                
            if features["FULL_TEXT_VECTORIZER"]:
                raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. Nelze použít FULL_TEXT_VECTORIZER, když není vybrán plný text.",
                            ErrorMessenger.CODE_INVALID_CONFIG)
                    
            if features["FULL_TEXT_ANALYZER"]:
                raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. Nelze použít FULL_TEXT_ANALYZER, když není vybrán plný text.",
                            ErrorMessenger.CODE_INVALID_CONFIG)    
                
            if features["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"]:
                raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. Nelze použít FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON, když není vybrán plný text.",
                            ErrorMessenger.CODE_INVALID_CONFIG) 
        
        if transformedGetData["GET_META_FIELDS"]:
            result["META_VECTORIZERS"]=dict([(mName,Features.tfidfVectorizerName) for mName in transformedGetData["GET_META_FIELDS"]])

            tmpVect=self.__createDict(features["META_VECTORIZERS"])
                  
            
            if tmpVect:
                for name, vect in tmpVect.items():
                    if name not in transformedGetData["GET_META_FIELDS"]:
                        raise ExceptionMessageCode(
                            "Nevalidní hodnota v konfiguračním souboru. Název pole v META_VECTORIZERS neodpovídá žádnému z GET_META_FIELDS: "+ name,
                                ErrorMessenger.CODE_INVALID_CONFIG)
                    
                    vect=vect.lower()
                    if vect not in Features.vectorizersNames:
                        raise ExceptionMessageCode(
                            "Nevalidní hodnota v konfiguračním souboru. V META_VECTORIZERS je uveden neznámý nástroj pro vektorizaci: "+ vect,
                                ErrorMessenger.CODE_INVALID_CONFIG)
                        
                    result["META_VECTORIZERS"][name]=vect


            if features["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
                result["META_VECTORIZERS_BUILD_VOCABULARY_ON"]=self.__createTuple(features["META_VECTORIZERS_BUILD_VOCABULARY_ON"])
                
                allDataKinds=transformedGetData["GET_META_FIELDS"].copy()
                if transformedGetData["GET_FULLTEXT"]:
                    allDataKinds+=[ConfigManager.fulltextName]
                    
                for name, _ in result["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
                    if name not in allDataKinds:
                        raise ExceptionMessageCode(
                            "Nevalidní hodnota v konfiguračním souboru. Název pole v META_VECTORIZERS_BUILD_VOCABULARY_ON neodpovídá žádnému z GET_META_FIELDS (nebo fulltext): "+ name,
                                ErrorMessenger.CODE_INVALID_CONFIG)

            result["META_ANALYZERS"]=dict([(mName,(Features.metaAnalyzersNames[0], 1)) for mName in transformedGetData["GET_META_FIELDS"]])
            
            tmpAnaly=self.__createDict(features["META_ANALYZERS"])
            if tmpAnaly:
                for name, analy in tmpAnaly.items():
                    if name not in transformedGetData["GET_META_FIELDS"]:
                        raise ExceptionMessageCode(
                            "Nevalidní hodnota v konfiguračním souboru. Název pole v META_ANALYZERS neodpovídá žádnému z GET_META_FIELDS: "+ name,
                                ErrorMessenger.CODE_INVALID_CONFIG)
                        
                    analy=analy.lower()
                    analySplit=analy.split("/",1)
                    
                    if analySplit[0] not in Features.metaAnalyzersNames:
                        raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. META_ANALYZERS obsahuje neznámé jméno analyzátoru. Použijte prosím následující: "+" ".join(Features.metaAnalyzersNames),
                            ErrorMessenger.CODE_INVALID_CONFIG)
                    
                    if analySplit[0]==Features.ngramName and len(analySplit)!=2:
                        raise ExceptionMessageCode(
                            "Nevalidní hodnota v konfiguračním souboru. Nevalidní parametr pro "+Features.ngramName+" v META_ANALYZERS: "+ analy,
                                ErrorMessenger.CODE_INVALID_CONFIG)
                    
                    par=None
                    if analySplit[0]!=Features.wholeitemName:
                        try:
                            par = int(analySplit[1])
                            if par<1:
                                raise ValueError()
                        except ValueError:
                            raise ExceptionMessageCode(
                                "Nevalidní hodnota v konfiguračním souboru. Nevhodný parametr pro analyzátor v META_ANALYZERS (musí být celé číslo větší než 0): "+analy,
                                ErrorMessenger.CODE_INVALID_CONFIG)
                        
                        
                    if analySplit[0] not in Features.metaAnalyzersNames:
                        raise ExceptionMessageCode(
                            "Nevalidní hodnota v konfiguračním souboru. V META_ANALYZERS je uveden neznámý analyzátor: "+ analy,
                                ErrorMessenger.CODE_INVALID_CONFIG)
                    
                    result["META_ANALYZERS"][name]=(analySplit[0], par)
        else:

            if features["META_VECTORIZERS"]:
                raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. Nelze použít META_VECTORIZERS, když je prázdné GET_META_FIELDS.",
                            ErrorMessenger.CODE_INVALID_CONFIG)
                    
            if features["META_ANALYZERS"]:
                raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. Nelze použít META_ANALYZERS, když je prázdné GET_META_FIELDS.",
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
            if features["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
                raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. Nelze použít META_VECTORIZERS_BUILD_VOCABULARY_ON, když je prázdné GET_META_FIELDS.",
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
        return result
    
    def __transformDOC2VECVals(self):
        """
        Převede hodnoty pro sekci DOC2VEC a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        result={
            "SIZE":500,
            "ALPHA":0.025,
            "WINDOW":5,
            "MIN_COUNT":5,
            "WORKERS":1,
            "ITER":10,
            "SAMPLE":0,
            "DM":1,
            "NEGATIVE":0
            }

        
        doc2vec=self.configParser[self.sectionDOC2VEC]
        
        
        try:
            if doc2vec["SIZE"]:
                param="SIZE"
                result["SIZE"]=int(doc2vec["SIZE"])
                
                if result["SIZE"]<1:
                    raise ValueError()
                
            if doc2vec["WINDOW"]:
                param="WINDOW"
                result["WINDOW"]=int(doc2vec["WINDOW"])
                
                if result["WINDOW"]<1:
                    raise ValueError()
                
            if doc2vec["MIN_COUNT"]:
                param="MIN_COUNT"
                result["MIN_COUNT"]=int(doc2vec["MIN_COUNT"])
            
            if doc2vec["WORKERS"]:
                param="WORKERS"
                result["WORKERS"]=int(doc2vec["WORKERS"])
                
            if doc2vec["ITER"]:
                param="ITER"
                
                result["ITER"]=int(doc2vec["ITER"])
                if result["ITER"]<1:
                    raise ValueError()        
            if doc2vec["SAMPLE"]:
                param="SAMPLE"
                result["SAMPLE"]=float(doc2vec["SAMPLE"])
                
            if doc2vec["DM"]:
                param="DM"
                result["DM"]=int(doc2vec["DM"])
                
            if doc2vec["NEGATIVE"]:
                param="NEGATIVE"
                result["NEGATIVE"]=int(doc2vec["NEGATIVE"])
            
            
            
        except ValueError:
            raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci DOC2VEC u parametru: "+param+"",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            
            
        try:
            if doc2vec["ALPHA"]:
                result["ALPHA"]=float(doc2vec["ALPHA"])
            
        except ValueError:
            raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci DOC2VEC u parametru: ALPHA (pouze číslo)",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            
            
        return result
    
    def __transformHashingVectorizerVals(self):
        """
        Převede hodnoty pro sekci HASHING_VECTORIZER a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        result={}
        
        heshVec=self.configParser[self.sectionHashingVectorizer]

        try:
            result["N_FEATURES"]=int(heshVec["N_FEATURES"])
        except ValueError:
            raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci HASHING_VECTORIZER u parametru: N_FEATURES (pouze celé číslo)",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            
            
        result["NON_NEGATIVE"]=heshVec["NON_NEGATIVE"].lower()=="true"
            
        
            
        return result
    

    def __transformClassificationVals(self, transformedGetData):
        """
        Převede hodnoty pro sekci CLASSIFICATION a validuje je.
        
        :param transformedGetData: Pro validaci je nutné, poskytnout převedenou sekci get data.
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        
        #defaultní hodnoty
        result={
            "CLASSIFIER":[],
            "BALANCING":None,
            "WORKERS":1,
            "WEIGHT_AUTO_CV":4
            }

        clsFromConfig=self.configParser[self.sectionClassification]["CLASSIFIER"]
        
        if clsFromConfig:  
                
            for actData in self.__createTupleQuater(clsFromConfig):
                clsWeight=1
                clsThreshold=0
                dataName=actData[0]
                clsName=actData[1].lower()
                if len(actData)>2:
                    clsWeight=actData[2]
                if len(actData)>3:
                    clsThreshold=actData[3]
            
                if clsName not in Classification.classifiersNames:
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionClassification+" u parametru CLASSIFIER je uveden neznámý klasifikátor: "+ clsName,
                            ErrorMessenger.CODE_INVALID_CONFIG)
                    
                if dataName not in transformedGetData["GET_META_FIELDS"]+[self.fulltextName]:
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionClassification+" u parametru CLASSIFIER je uveden neznámý název dat: "+ dataName,
                            ErrorMessenger.CODE_INVALID_CONFIG)
                    
                
                try:
                    clsWeight = float(clsWeight) if clsWeight!="auto" else clsWeight
                except ValueError:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci CLASSIFICATION u CLASSIFIER (musí být jako váha uvedeno číslo nebo auto): "+str(clsWeight),
                                ErrorMessenger.CODE_INVALID_CONFIG)
                    
                try:
                    clsThreshold = float(clsThreshold)
                    if clsThreshold<0 or clsThreshold>1:
                        raise ValueError()
                except ValueError:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci CLASSIFICATION u CLASSIFIER (musí být jako práh uvedeno číslo <0,1>): "+str(clsThreshold),
                                ErrorMessenger.CODE_INVALID_CONFIG)
                    
                
                    
                for actDataName, actClsName, actClsWeight, _ in result["CLASSIFIER"]:
                    if actDataName==dataName and actClsName==clsName and actClsWeight==clsWeight:
                        raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci CLASSIFICATION u CLASSIFIER se vyskytují vícekrát stejné klasifikátory nad stejným druhem dat a se stejnou váhou. : "+str(clsThreshold),
                                ErrorMessenger.CODE_INVALID_CONFIG)
                
                result["CLASSIFIER"].append((dataName, clsName, clsWeight, clsThreshold))
                
        allDataNames=[xName for xName, _, _, _ in result["CLASSIFIER"]]
        
        if transformedGetData["GET_FULLTEXT"] and self.fulltextName not in allDataNames:
            result["CLASSIFIER"].append((self.fulltextName,Classification.linearSVCName, 1, 0))
            
        if transformedGetData["GET_META_FIELDS"]:
            for mName in transformedGetData["GET_META_FIELDS"]:
                if mName not in allDataNames:
                    result["CLASSIFIER"].append((mName,Classification.linearSVCName, 1, 0))
                    
                    
        balFromConfig=self.configParser[self.sectionClassification]["BALANCING"]   
        if balFromConfig:
            result["BALANCING"]=[]
            for balName, balRat in self.__createTuple(balFromConfig, numValues="float"):
                balName=balName.lower()
                
                if balName not in Balancing.balancersNames:
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionClassification+" u parametru BALANCING je uvedena neznámá metoda: "+ balName,
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
                if balRat <= 0 or (balName==Balancing.randomUnderSampling and balRat>1):
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionClassification+" u parametru BALANCING. "+ str(balRat),
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
                result["BALANCING"].append((balName, {"ratio":balRat}))
                
    
        workersFromConfig=self.configParser[self.sectionClassification]["WORKERS"]  

        if workersFromConfig:
            try:
                result["WORKERS"]=int(workersFromConfig)
        
                if result["WORKERS"]<1 and result["WORKERS"]!=-1:
                    raise ValueError()
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci "+self.sectionClassification+" u parametru: WORKERS",
                                       ErrorMessenger.CODE_INVALID_CONFIG)
                
        if self.configParser[self.sectionClassification]["WEIGHT_AUTO_CV"]:
            try:
                result["WEIGHT_AUTO_CV"]=int(self.configParser[self.sectionClassification]["WEIGHT_AUTO_CV"])
                
                if result["WEIGHT_AUTO_CV"]<2:
                    raise ValueError()
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci "+self.sectionClassification+" u parametru: WEIGHT_AUTO_CV",
                                       ErrorMessenger.CODE_INVALID_CONFIG)
        
            
        return result
    
    def __transformPredictVals(self):
        """
        Převede hodnoty pro sekci PREDICTION a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        
        result={
            "PREDICTED_FIELD_NAME":"PREDICTED",
            "N_BEST":1,
            "N_BEST_PREFIX": "BEST_",
            "CONF_POSTFIX": None,
            "USE_PROB":True,
            "THRESHOLD":0.0,
            "WORKERS":1
            }
        
        
        predict=self.configParser[self.sectionPredict]
        
        if predict["PREDICTED_FIELD_NAME"]:
            result["PREDICTED_FIELD_NAME"]=shlex.split(predict["PREDICTED_FIELD_NAME"])[0]
         
        if predict["WORKERS"]:   
            try:
                result["WORKERS"]=int(predict["WORKERS"])
        
                if result["WORKERS"]<1 and result["WORKERS"]!=-1:
                    raise ValueError()
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci "+self.sectionPredict+" u parametru: WORKERS",
                                       ErrorMessenger.CODE_INVALID_CONFIG)
                
        if predict["THRESHOLD"]:
            try:
                threshold = float(predict["THRESHOLD"])
                if threshold<0 or threshold>1:
                    raise ValueError()
                result["THRESHOLD"]=threshold
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionPredict+" u THRESHOLD (musí být uvedeno číslo <0,1>): "+str(threshold),
                                ErrorMessenger.CODE_INVALID_CONFIG)
            
            
        if predict["N_BEST"]:
            try:
                result["N_BEST"]=int(predict["N_BEST"])
                if result["N_BEST"]<1:
                    raise ValueError()
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru v sekci "+self.sectionPredict+" u parametru: N_BEST (pouze kladné celé číslo)",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
        
        if predict["N_BEST_PREFIX"]:
            result["N_BEST_PREFIX"]=shlex.split(predict["N_BEST_PREFIX"])[0]
            
        if predict["CONF_POSTFIX"]:
            result["CONF_POSTFIX"]=shlex.split(predict["CONF_POSTFIX"])[0]    
            
        result["WRITE_META_FIELDS"]=shlex.split(predict["WRITE_META_FIELDS"])
        
        useProbFromConfig=self.configParser[self.sectionPredict]["USE_PROB"]
        if useProbFromConfig:
            result["USE_PROB"]=useProbFromConfig.lower()=="true"
        
        return result
    
    def __transformTestingVals(self):
        """
        Převede hodnoty pro sekci TESTING a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        
        result={
            "SPLIT_METHOD":SplitTestSet.stratifiedKFoldName,
            "PREDICT_MATCH_FIELD_NAME":None,
            "SPLITS":4,
            "TEST_SIZE":0.25
            }
        testing=self.configParser[self.sectionTesting]
        
        
        if testing["SPLIT_METHOD"]:
            splitMethod=testing["SPLIT_METHOD"].lower()
            
            if splitMethod not in SplitTestSet.splitMethods:
                raise ExceptionMessageCode(
                    "Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionTesting+" v parametru SPLIT_METHOD je uvedena neznámá metoda: "+ testing["SPLIT_METHOD"],
                        ErrorMessenger.CODE_INVALID_CONFIG)
                
            result["SPLIT_METHOD"]=splitMethod

        if testing["PREDICT_MATCH_FIELD_NAME"]:
            result["PREDICT_MATCH_FIELD_NAME"]=shlex.split(testing["PREDICT_MATCH_FIELD_NAME"])[0]
            
            
        
        if testing["SPLITS"]:
            v=testing["SPLITS"]
            try:
                v=int(v)
                if v<1:
                    raise ValueError()
                
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionTesting+" u SPLITS (musí být kladné celé číslo): "+str(v),
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
                
            if result["SPLIT_METHOD"]==SplitTestSet.kFoldName:
                if v<2:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionTesting+" u SPLITS pro "+testing["SPLIT_METHOD"]+" musí být kladné celé číslo větší než 1: "+str(v),
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
            result["SPLITS"]=v
                
        if testing["TEST_SIZE"]:
            v=testing["TEST_SIZE"]
            try:
                v=int(v)
                if v<1:
                    raise ValueError()
            except ValueError:
                try:
                    v = float(v)
                    if v<=0:
                        raise ValueError()
                except ValueError:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionTesting+" u TEST_SIZE (musí být kladné číslo): "+str(v),
                                ErrorMessenger.CODE_INVALID_CONFIG)
        
            result["TEST_SIZE"]=v
        
        return result

    
    
    def __transformKNeighborsClassifierVals(self):
        """
        Převede hodnoty pro sekci K_NEIGHBORS_CLASSIFIER a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        
        result={
            "NUM_OF_NEIGHBORS":3,
            "WORKERS": 1,
            "WEIGHTS": "distance"
            }
        
        knc=self.configParser[self.sectionKNeighborsClassifier]
        
        if knc["NUM_OF_NEIGHBORS"]:
            v=knc["NUM_OF_NEIGHBORS"]
            try:
                v=int(v)
                if v<1:
                    raise ValueError()
                
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionKNeighborsClassifier+" u NUM_OF_NEIGHBORS (musí být kladné celé číslo): "+str(v),
                            ErrorMessenger.CODE_INVALID_CONFIG)
        
            result["NUM_OF_NEIGHBORS"]=v
            
        if knc["WORKERS"]:
            v=knc["WORKERS"]
            try:
                v=int(v)
                if v<-1:
                    raise ValueError()
                
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionKNeighborsClassifier+" u WORKERS (musí být celé číslo > -2): "+str(v),
                            ErrorMessenger.CODE_INVALID_CONFIG)
                
            result["WORKERS"]=v
            
        if knc["WEIGHTS"]:
            v=knc["WEIGHTS"].lower()
            if v not in ["uniform", "distance"]:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionKNeighborsClassifier+" u WEIGHTS (musí být uniform nebo distance): "+str(v),
                            ErrorMessenger.CODE_INVALID_CONFIG)
            result["WEIGHTS"]=v
            
        return result
    
    def __transformSGDClassifierVals(self):
        """
        Převede hodnoty pro sekci SGDClassifier.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        
        result={
            "LOSS":"hinge"
            }
        
        sgdVals=self.configParser[self.sectionSGDClassifier]
        if sgdVals["LOSS"]:
            if sgdVals["LOSS"] not in ["hinge", "squared_hinge", "perceptron", "log", "modified_huber", "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionSGDClassifier+" u loss (neznámý parametr): "+str(sgdVals["LOSS"]),
                            ErrorMessenger.CODE_INVALID_CONFIG)
            result["LOSS"]=sgdVals["LOSS"]
        
        
        return result
        

    
    def __createDict(self, confStr, convertSelectors=False, numValues=None):
        """
        Z konfigurační hodnoty vytvoří dictionary.
        
        :param confStr: Konfigurační hodnota.
        :param convertSelectors: True => Převede Selectory.
        :param numValues: "int" => celá čísla "float" => čísla (všechny neprázdné stringy kromě int). Validuje a konvertuje.
        :returns: dict -- převedená konfigurační hodnota
        """
        
        result=self.__createTuple(confStr, convertSelectors, numValues)
        if not result:
            return None
        
        if len(result)!=len(set([x[0] for x in result])):
            raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru (více údajů se mapuje na stejný klíč): "+confStr,
                                                   ErrorMessenger.CODE_INVALID_CONFIG)
        return dict(result)
    
    def __createTuple(self, confStr, convertSelectors=False, numValues=None):
        """
        Z konfigurační hodnoty vytvoří list dvojic.
        
        :param confStr: Konfigurační hodnota.
        :param convertSelectors: True => Převede Selectory.
        :param numValues: "int" => celá čísla "float" => čísla (všechny neprázdné stringy kromě int). Validuje a konvertuje.
        :returns: list -- dvojic (klíč, hodnota)
        """
        
        pairs=shlex.split(confStr)
        result=[]
        for p in pairs:
            try:
                k,v= p.split(":")
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru: "+confStr,
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            
            if convertSelectors:
                v=self.__parseSelectorVal(v)
            
            elif numValues:
                
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru (musí být číslo): "+confStr,
                                                   ErrorMessenger.CODE_INVALID_CONFIG)
                
                if numValues=="int":
                    try:
                        v = int(v)
                    except ValueError:
                        raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru (musí být celé číslo): "+confStr,
                                                   ErrorMessenger.CODE_INVALID_CONFIG)

                
            result.append((k,v))
            
        return result
    
    def __createTriplet(self, confStr):
        """
        Z konfigurační hodnoty vytvoří list trojic.
        
        :param confStr: Konfigurační hodnota.
        :returns: list -- trojic
        """
        
        triplets=shlex.split(confStr)
        result=[]
        for t in triplets:
            try:
                v1,v2,v3= t.split(":")
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru: "+confStr,
                                           ErrorMessenger.CODE_INVALID_CONFIG)
                
            result.append((v1,v2,v3))
            
        return result
    
    def __createTupleQuater(self, confStr):
        """
        Z konfigurační hodnoty vytvoří list dvojic až čtveřic.
        
        :param confStr: Konfigurační hodnota.
        :returns: list -- dvojic až čtveřic
        """
        
        parse=shlex.split(confStr)
        result=[]
        for t in parse:
            splT=t.split(":")
            if len(splT)<2 or len(splT)>4:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru: "+confStr,
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            result.append(tuple(splT))
                
            
            
        return result
        
    def __parseSelectorVal(self, txtWithSelVal):
        """
        Vytvoří hodnotu pro selector.
        
        :param txtWithSelVal: Řetězec obshaující hodnotu pro selektor.
        """
        if not txtWithSelVal:
            return None
        
        sliceParts=txtWithSelVal.split("-")

        for i, sp in enumerate(sliceParts):
            try:
                if not sliceParts[i]:
                    sliceParts[i]=None
                else:
                    sliceParts[i]=int(sp)
            except ValueError:
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru (selektor musí mít celočíselné hodnoty): "+txtWithSelVal,
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            

        v=None
        if len(sliceParts)==1:
            v=sliceParts[0]
        elif len(sliceParts)==2:
            v=slice(sliceParts[0], sliceParts[1])
        elif len(sliceParts)==3:
            v=slice(sliceParts[0], sliceParts[1], sliceParts[2])
        else:
            raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru: "+txtWithSelVal,
                                       ErrorMessenger.CODE_INVALID_CONFIG)
        
        return v
        
        

class CPKclassifier(object):
    """
    Nástroj pro klasifikaci dokumentů, hledání podobných dokumentů a předzpracování dokumentů.
    """

    
    
    def __init__(self, config, logAfterLines=100, logAfterSec=30):
        """
        Inicializace.
        
        :param config: Cesta k souboru se základní konfigurací.
        :param logAfterLines: Udává po kolika přečtených řádcích má dojit k logování.
        :param logAfterSec: Udává po kolika sekundách (nejdříve) dojde k logování. Používa se například při preprocessingu.
        """
        self.logAfterLines=logAfterLines
        self.logAfterSec=logAfterSec
        self.initialConfig=config
        
        self.partSize=1000    #používá se u predikace a extrakce příznaků
    
    def __initCheck(self, args):
        """
        Počáteční kontrola.
        
        :param args: Argumenty z argument manažéru.
        :raises ExceptionMessageCode: Pokud neexistuje konfigurační soubor.
        """
        self.config=ConfigManager()
        
        configFiles=[self.initialConfig]
        
        if args.config:
            
            if os.path.isfile(args.config):
                configFiles.append(args.config)
            else:
                raise  ExceptionMessageCode("Konfigurační soubor neexistuje: "+args.config, 
                        ErrorMessenger.CODE_INVALID_ARGUMENTS)

        
        self.configAll=self.config.read(configFiles)
    
    def preprocessing(self,args):
        """
        Obstarání předzpracování
        
        :param args: Argumenty z argument manažéru.
        """
        self.__initCheck(args)
        
        if self.configAll[ConfigManager.sectionPreprocessing]["STOP_WORDS"] is None:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Chybí STOP_WORDS.",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
         
        if self.configAll[ConfigManager.sectionPreprocessing]["DICT"] is None:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Chybí DICT.",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            
        
        try:
            Preprocessing(args, self.configAll[ConfigManager.sectionPreprocessing]["STOP_WORDS"], 
                      self.configAll[ConfigManager.sectionPreprocessing]["DICT"], 
                      self.configAll[ConfigManager.sectionPreprocessing]["WORKERS"], self.logAfterSec,
                      self.configAll[ConfigManager.sectionPreprocessing]["MAX_NUMBER_OF_WORDS_PER_LINE_PART"]).start()
        except LemmatizerException:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Problém s dict.",
                                           ErrorMessenger.CODE_INVALID_CONFIG)   
        

    def getDataArgs(self, args):
        """
        Obstarání argumentu programu getData.
        
        :param args: Argumenty z argument manažéru.
        """
        self.__initCheck(args)

        dataSet=self.__getDataSet(args)

            
        print("Počet dokumentů:\t"+str(len(dataSet.reader)))
        
        if self.configAll[ConfigManager.sectionGetData]["TARGET_FIELD"] and \
            self.configAll[ConfigManager.sectionGetData]["TARGET_FIELD"] in dataSet.reader.fieldNames:
            
            stats=dataSet.reader.fieldsStats([self.configAll[ConfigManager.sectionGetData]["TARGET_FIELD"]])[self.configAll[ConfigManager.sectionGetData]["TARGET_FIELD"]]
            for name, val in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                print(name+"\t"+str(val))


        if args.saveDataTo and  self.configAll[ConfigManager.sectionGetData]["GET_FULLTEXT"] and \
            args.saveMetadataTo and self.configAll[ConfigManager.sectionGetData]["GET_META_FIELDS"]:
            dataSet.writeMetaAndFulltextData(args.saveMetadataTo, args.saveDataTo)
        elif args.saveDataTo and self.configAll[ConfigManager.sectionGetData]["GET_FULLTEXT"]:
            dataSet.writeFullText(args.saveDataTo)
        elif args.saveMetadataTo:
            dataSet.writeMetadata(args.saveMetadataTo)
        
    def featuresExtracting(self, args):
        """
        Obstarává extrakci příznaků.
        
        :param args: Argumenty z argument manažéru.
        """
  
        self.__initCheck(args)
        
        actUseConfig=self._getConfigForVocaBuilding()
        
        dSet=self.__getDataSet(args, useConfig=actUseConfig)        
            
        trainData, trainTargets=self.__getData(args, useDataset=dSet)
        
        
        extracted, features=self.__performFeaturesExtracting(trainData, trainTargets)
        
        for dataName, dFeat in extracted.items():
            logging.info("Velikost vektoru pro "+dataName+": "+str(dFeat.vectorSize()))
        
        CPKclassifierDataDump(targets=trainTargets, featuresTool=features, features=extracted, configFea=self.configAll).save(args.saveTo)


    def __getDataSet(self, args, useConfig=None):
        """
        Vytvoření objektu pro prací s daty.
        
        :param args: Argumenty z argument manažéru. Jsou použity pro získání souborů data a metadat.
        :param useConfig: Místo self.configAll použije daný config.
        """
        
        if not useConfig:
            useConfig=self.configAll
        
        try:        
            
            dataSet=DataSet(metadata=args.metadata, 
                            data=args.data, 
                            getMetaFields=useConfig[ConfigManager.sectionGetData]["GET_META_FIELDS"], 
                            getFulltext=useConfig[ConfigManager.sectionGetData]["GET_FULLTEXT"], 
                            targetField=useConfig[ConfigManager.sectionGetData]["TARGET_FIELD"], 
                            lazyEvalData=useConfig[ConfigManager.sectionGetData]["LAZY_EVAL_DATA"], 
                            nonEmpty=useConfig[ConfigManager.sectionGetData]["NON_EMPTY"], 
                            empty=useConfig[ConfigManager.sectionGetData]["EMPTY"], 
                            fieldRegex={dataName: (re.compile(x[0]),x[1]) for dataName, x in useConfig[ConfigManager.sectionGetData]["FIELD_REGEX"].items()}, 
                            minPerField=useConfig[ConfigManager.sectionGetData]["MIN_PER_FIELD"], 
                            maxPerField=useConfig[ConfigManager.sectionGetData]["MAX_PER_FIELD"], 
                            itemDelimiter=useConfig[ConfigManager.sectionGetData]["ITEM_DELIMITER"], 
                            selectWords=useConfig[ConfigManager.sectionGetData]["SELECT_WORDS"], 
                            selectItems=useConfig[ConfigManager.sectionGetData]["SELECT_ITEMS"], 
                            copyAndRen=useConfig[ConfigManager.sectionGetData]["COPY"],
                            fulltextName=ConfigManager.fulltextName)
            
        except DataSetInvalidMetadataFields:
            raise ExceptionMessageCode(
                ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" V konfiguračním souboru je název pole metadat, který není ve vstupním souboru s metadaty.",
                ErrorMessenger.CODE_INVALID_CONFIG)
        except DataSetNoDataPath:
            raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS)+" Je nutné zadat cestu k datovému souboru s dokumenty.", 
                        ErrorMessenger.CODE_INVALID_ARGUMENTS)
        except DataSetInvalidDataFileForMetadata:
            raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS)+"  Nevalidní datový soubor pro daná metadata.", 
                        ErrorMessenger.CODE_INVALID_ARGUMENTS)
            
        return dataSet
        
    def __getData(self, args, targets=True, useConfig=None, useDataset=None):
        """
        Získání dat a cílů.
        
        :param args: Argumenty z argument manažéru. Jsou použity pro získání souborů data a metadat.
        :param targets: True => získat cíle. 
        :param useConfig: Místo self.configAll použije daný config.
        :param useDataset: Pokud není None ignoruje ostatní parametry a použije předvytvořený dataset.
        :returns: Dvojici (data, cíle). Pokud targets=False => data.
        """
        
        dataSet=useDataset
        
        if not dataSet:
            if not useConfig:
                useConfig=self.configAll
            
            if targets and not useConfig[ConfigManager.sectionGetData]["TARGET_FIELD"]:
                raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Je nutné vyplnit cíl pro klasifikaci TARGET_FIELD.",
                    ErrorMessenger.CODE_INVALID_CONFIG)
            
            
            dataSet=self.__getDataSet(args,useConfig)
            
        try:
            if targets:
                gData, gTargets =dataSet.getTrainData()
            else:
                gData=dataSet.getData()
            
        except DataSetInvalidTarget:
            raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Neexistující cíl pro trénování.", 
                        ErrorMessenger.CODE_INVALID_CONFIG)
            
        
        if not gData or len(gData[next(iter(gData))])==0:
            raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_NO_INPUT_DATA), 
                        ErrorMessenger.CODE_NO_INPUT_DATA)
        if targets:
            return (gData, gTargets)
        else:
            return gData
        
        
    def __performFeaturesExtracting(self, dataToExtract, targets, retFeatTool=True):
        """
        Provede extrakci příznaků.
        
        :param args: Argumenty z argument manažéru.
        :param dataToExtract: Data pro extrakci
        :param retFeatTool: True
        :returns: Dvojici (extrahované příznaky, Features). Pokud retFeatTool false => extrahované příznaky.
        """
        lemmatizer=None
        try:
            allVecNames=[]
            if self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS"]:
                allVecNames+=[x for _, x in self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS"].items()]
                              
            if self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER"]:
                allVecNames+=[self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER"]]
                
                
            if Features.matchTargetVectorizer in allVecNames:
                lemmatizer=Lemmatizer(self.configAll[ConfigManager.sectionPreprocessing]["DICT"])
                
        except LemmatizerException:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Problém s dict.",
                                               ErrorMessenger.CODE_INVALID_CONFIG)   
        
        try:
            features=Features(
                getFulltext=self.configAll[ConfigManager.sectionGetData]["GET_FULLTEXT"], 
                getMetaFields=self.configAll[ConfigManager.sectionGetData]["GET_META_FIELDS"], 
                fullTextVectorizer=self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER"], 
                fullTextAnalyzer=self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_ANALYZER"], 
                metaVectorizers=self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS"], 
                metaAnalyzers=self.configAll[ConfigManager.sectionFeatures]["META_ANALYZERS"], 
                hashingVectorizer=self.configAll[ConfigManager.sectionHashingVectorizer], 
                doc2Vec=self.configAll[ConfigManager.sectionDOC2VEC], 
                lemmatizer=lemmatizer,
                fulltextName=ConfigManager.fulltextName,
                markEmpty=self.configAll[ConfigManager.sectionFeatures]["SKIP_EMPTY"])
        except FeaturesNoData:
            raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_NO_INPUT_DATA)+" Žádná data pro extrakci příznaků.", 
                        ErrorMessenger.CODE_NO_INPUT_DATA)
            
        

        if self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"] or \
            self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
            
            dataForVoca={}
            if self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"]:
                dataForVoca[ConfigManager.fulltextName]=dataToExtract[self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"]]
                
            if self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
                
                for fieldName, fieldNameForVocaBuilding in self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
                    dataForVoca[fieldName]=dataToExtract[fieldNameForVocaBuilding]
            
            features.learnVocabularies(dataForVoca)

        extracted=features.extractAndLearn(dataToExtract, targets, self.partSize, 
                                           workers=self.configAll[ConfigManager.sectionFeatures]["WORKERS"])
        
        
        if retFeatTool:
            return (extracted, features)
        
        return extracted
    
    def __makeStats(self, data):
        """
        Počet výskytů k dané položce.
        
        :param data: list -- obsahující položky
        """
        
        dataStats={}
        for ite in data:
            if ite not in dataStats:
                dataStats[ite]=0
            dataStats[ite]=dataStats[ite]+1
            
        return dataStats
    
    def __logStats(self, data):
        """
        Loguje počet výskytů k dané položce.
        
        :param data: list -- obsahující položky
        """
        
        dataStats=self.__makeStats(data)
        
        for name, val in sorted(dataStats.items(), key=lambda x: x[1], reverse=True):
            logging.info(name+"\t"+str(val))
        
    def classification(self, args):
        """
        Trénování klasifikátoru.
        
        :param args: Argumenty z argument manažéru.
        """    
        
        self.__initCheck(args)
        
        trainData=None
        trainTargets=None
        extracted=None
        featuresTool=None

        if args.features:
            
            loadData=CPKclassifierDataDump()
            
            try:
                loadData.loadBasicData(args.features)
                loadData.loadExtractedFeatures(args.features)
            except CPKclassifierDataDumpInvalidFile:
                raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_INPUT_FILE),
                                           ErrorMessenger.CODE_INVALID_INPUT_FILE)

            if loadData.features is None or loadData.targets is None or loadData.configFea is None:
                raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_INPUT_FILE),
                                           ErrorMessenger.CODE_INVALID_INPUT_FILE)
                
            trainTargets=loadData.targets
            extracted=loadData.features
            config=loadData.configFea
            
        else:
            
            config=self.configAll

        if self.__invalidExtractingMethod(config, self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"]):
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INV_COMB_EXT_FEAT_METHOD_AND_CLS),
                                           ErrorMessenger.CODE_INV_COMB_EXT_FEAT_METHOD_AND_CLS)
            
            
        if not args.features:
            #na tomto místě je to kvůli chybě ErrorMessenger.CODE_INV_COMB_EXT_FEAT_METHOD_AND_CLS
            
            actUseConfig=self._getConfigForVocaBuilding()
            
            dSet=self.__getDataSet(args, useConfig=actUseConfig)
            
            
            trainData, trainTargets=self.__getData(args, useDataset=dSet)
            extracted, featuresTool=self.__performFeaturesExtracting(trainData, trainTargets)
            
        for dataName, dFeat in extracted.items():
            logging.info("Velikost vektoru pro "+dataName+": "+str(dFeat.vectorSize()))
            
        saveTo=CPKclassifierDataDump(
            targets=trainTargets,
            features=extracted, 
            configFea=config, 
            configCls=self.configAll)
            
        if args.features:
            copyfile(args.features+CPKclassifierDataDump.featuresToolExtension, args.saveTo+CPKclassifierDataDump.featuresToolExtension)
        else:
            saveTo.addFeaturesTool(featuresTool)
            
        saveTo.save(args.saveTo)
        
        if featuresTool:
            featuresTool.flush()
                 
        
        cls=Classification(self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"], self.createParamsForClassifiers(),
                           self.configAll[ConfigManager.sectionClassification]["WEIGHT_AUTO_CV"])
        
        if self.configAll[ConfigManager.sectionClassification]["BALANCING"]:
            logging.info("začátek vyvažování trénovací množiny")
            bal=Balancing(self.configAll[ConfigManager.sectionClassification]["BALANCING"])
            
            extracted, trainTargets=bal.balance(extracted, trainTargets)
            logging.info("konec vyvažování trénovací množiny")
            
        cls.train(extracted, trainTargets, workers=self.configAll[ConfigManager.sectionClassification]["WORKERS"])
        
        logging.info("Vypisuji počet dokumentů na kategorii.")
        
        self.__logStats(trainTargets)
        
        
        
        saveTo.addClassificator(cls)
        saveTo.saveClassificator(args.saveTo)
        
                
            
    def predict(self, args):
        """
        Predikování cílů na naučeném klasifikátoru.
        
        :param args: Argumenty z argument manažéru.
        """
        self.__initCheck(args)
        
        loadData=CPKclassifierDataDump()
            
        try:
            loadData.loadClassificator(args.classifiers)
            loadData.loadFeaturesTool(args.classifiers)
            loadData.loadBasicData(args.classifiers)
        except CPKclassifierDataDumpInvalidFile:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_INPUT_FILE),
                    ErrorMessenger.CODE_INVALID_INPUT_FILE)
                
                
        if loadData.featuresTool is None or loadData.classificator is None or loadData.configFea is None:
                raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_INPUT_FILE),
                                           ErrorMessenger.CODE_INVALID_INPUT_FILE)
                
        if loadData.configFea[ConfigManager.sectionGetData]["GET_FULLTEXT"] and not args.data:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS)+" Je nutné přidat datový soubor s plnými texty.",
                                           ErrorMessenger.CODE_INVALID_ARGUMENTS)
            
        if loadData.configFea[ConfigManager.sectionGetData]["GET_META_FIELDS"] and not args.metadata:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS)+" Je nutné přidat metadatový soubor.",
                                           ErrorMessenger.CODE_INVALID_ARGUMENTS)
                
        
        
        
        dSet=self.__getDataSet(args)
        
        allKinds=set(loadData.configFea[ConfigManager.sectionGetData]["GET_META_FIELDS"])
        if loadData.configFea[ConfigManager.sectionGetData]["GET_FULLTEXT"]:
            allKinds.add(ConfigManager.fulltextName)
            
        testData=self.__getData(args, False, useDataset=dSet)

        
        if set(testData.keys())!=allKinds:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Pro predikci musí být pouze tato data: "+", ".join(allKinds),
                                           ErrorMessenger.CODE_INVALID_CONFIG)

        
        
        predictResult=Prediction(self.configAll[ConfigManager.sectionPredict]["PREDICTED_FIELD_NAME"], 
                                 self.configAll[ConfigManager.sectionPredict]["N_BEST"], 
                                 self.configAll[ConfigManager.sectionPredict]["N_BEST_PREFIX"], 
                                 sys.stdout, self.configAll[ConfigManager.sectionGetData]["ITEM_DELIMITER"],
                                 self.configAll[ConfigManager.sectionPredict]["CONF_POSTFIX"])
        
        
        metaForWrite={}
        if self.configAll[ConfigManager.sectionPredict]["WRITE_META_FIELDS"]:

            try:
                #upravíme původní dataset, tak abychom mohli získat dodatečná metadata a nemuseli vytvářet nový dSet.
                dSet.metaFields=self.configAll[ConfigManager.sectionPredict]["WRITE_META_FIELDS"]
                #chceme jen metadata a nepotřebujeme plný text
                dSet.disableFulltext()
            except DataSetInvalidMetadataFields:
                raise ExceptionMessageCode(
                ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" V konfiguračním souboru u pole WRITE_META_FIELDS v sekci "+ConfigManager.sectionPredict+" je název pole metadat, který není ve vstupní souboru s metadaty.",
                    ErrorMessenger.CODE_INVALID_CONFIG)
                
                            
            metaForWrite=self.__getData(args, targets=False, useDataset=dSet)
        
        
        if self.configAll[ConfigManager.sectionPredict]["USE_PROB"]:
            predicted=loadData.classificator.predictAuto(loadData.featuresTool.extract(testData, self.partSize, 
                                           workers=self.configAll[ConfigManager.sectionFeatures]["WORKERS"]), self.partSize, 
                                                         self.configAll[ConfigManager.sectionPredict]["THRESHOLD"],
                                                         self.configAll[ConfigManager.sectionPredict]["WORKERS"])
        else:
            predicted=loadData.classificator.predict(loadData.featuresTool.extract(testData, self.partSize, 
                                           workers=self.configAll[ConfigManager.sectionFeatures]["WORKERS"]), self.partSize, 
                                                         self.configAll[ConfigManager.sectionPredict]["WORKERS"])
            
        targetsNames=[]
        
        if self.configAll[ConfigManager.sectionPredict]["USE_PROB"] and loadData.classificator.couldGetNBest():
            targetsNames=loadData.classificator.targets
            
        predictResult.write(predicted, 
                            metaForWrite,
                            targetsNames, True)
        
        logging.info("Počet zpracovaných dokumentů: "+str(len(predicted)))
    
        
        
    def testing(self, args):
        """
        Testování klasifikátoru.
        
        :param args: Argumenty z argument manažéru.
        """
        
        
        self.__initCheck(args)
        
        if self.__invalidExtractingMethod(self.configAll, self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"]):
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INV_COMB_EXT_FEAT_METHOD_AND_CLS),
                                           ErrorMessenger.CODE_INV_COMB_EXT_FEAT_METHOD_AND_CLS)
            

        actUseConfig=self._getConfigForVocaBuilding()
        
        dSet=self.__getDataSet(args, useConfig=actUseConfig)
        
        allData, allTargets=self.__getData(args, useDataset=dSet)

        additionalData={}
        if self.configAll[ConfigManager.sectionPredict]["WRITE_META_FIELDS"]:
           
            try:
                #upravíme původní dataset, tak abychom mohli získat dodatečná metadata a nemuseli vytvářet nový dSet.
                dSet.metaFields=self.configAll[ConfigManager.sectionPredict]["WRITE_META_FIELDS"]
                #chceme jen metadata a nepotřebujeme plný text
                dSet.disableFulltext()
            except DataSetInvalidMetadataFields:
                raise ExceptionMessageCode(
                ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" V konfiguračním souboru u pole WRITE_META_FIELDS v sekci "+ConfigManager.sectionPredict+" je název pole metadat, který není ve vstupní souboru s metadaty.",
                    ErrorMessenger.CODE_INVALID_CONFIG)
        
            additionalData=self.__getData(args, targets=False, useDataset=dSet) #s těmito daty netrénujeme, tedy nepotřebujeme cíle
        
        predictResult=None
        writeResults=None
        if args.writeResults:
            writeResults=open(args.writeResults, "w")
            predictResult=Prediction(self.configAll[ConfigManager.sectionPredict]["PREDICTED_FIELD_NAME"], 
                                     self.configAll[ConfigManager.sectionPredict]["N_BEST"], 
                                     self.configAll[ConfigManager.sectionPredict]["N_BEST_PREFIX"], 
                                     writeResults, self.configAll[ConfigManager.sectionGetData]["ITEM_DELIMITER"],
                                     self.configAll[ConfigManager.sectionPredict]["CONF_POSTFIX"])
        
        testStats=Testing(self.configAll[ConfigManager.sectionDefault]["HIER_DELIMITER"], self.configAll[ConfigManager.sectionPredict]["N_BEST"])
        
        writeConfMat=None
        
        if args.writeConfMetrix:
            writeConfMat=open(args.writeConfMetrix, "w")
        
        actPart=0
        
        
        writteHeader=True
        
        if args.consistency:
            #delame test konzistence
            spliter=[(allData, allTargets, allData, allTargets, additionalData)]
            numOfPats=1
        else:
            
            spliter=SplitTestSet(allData, allTargets, additionalData)
        
            if self.configAll[ConfigManager.sectionTesting]["SPLIT_METHOD"]==SplitTestSet.stratifiedKFoldName:
                spliter.useMethodSKF(self.configAll[ConfigManager.sectionTesting]["SPLITS"])
                
            elif  self.configAll[ConfigManager.sectionTesting]["SPLIT_METHOD"]==SplitTestSet.stratifiedShuffleSplitName:
                spliter.useMethodSSS(self.configAll[ConfigManager.sectionTesting]["SPLITS"], 
                    self.configAll[ConfigManager.sectionTesting]["TEST_SIZE"])  
                
            elif  self.configAll[ConfigManager.sectionTesting]["SPLIT_METHOD"]==SplitTestSet.kFoldName:
                spliter.useMethodKF(self.configAll[ConfigManager.sectionTesting]["SPLITS"])
                
            numOfPats=spliter.numOfIteratations()
        
        allKindsOfData=set(self.configAll[ConfigManager.sectionGetData]["GET_META_FIELDS"])
        if self.configAll[ConfigManager.sectionGetData]["GET_FULLTEXT"]:
            allKindsOfData.add(ConfigManager.fulltextName)
            
        for trainData, trainTargets, testData, testTargets, testAddData in spliter:
            actPart=actPart+1

            if not args.consistency:
                logging.info("Křížová validace: "+str(actPart)+"/"+str(numOfPats))
            logging.info("Dokumentů pro trénování: "+str(len(trainTargets)))
            logging.info("Dokumentů pro testování: "+str(len(testTargets)))

            extracted, featuresTool=self.__performFeaturesExtracting(trainData, trainTargets)
            
            for dataName, dFeat in extracted.items():
                logging.info("Velikost vektoru pro "+dataName+": "+str(dFeat.vectorSize()))
    
            logging.info("začátek příznaky pro testování")
            if args.consistency:
                #Děláme test konzistence, nemusíme znovu extrahovat všechny příznaky.
                extractedTest=extracted

                logging.info("\tPřebírám z trénovacích příznaků.")
            else:                       
                extractedTest=featuresTool.extract(testData, self.partSize, 
                                           workers=self.configAll[ConfigManager.sectionFeatures]["WORKERS"])
                
            logging.info("konec příznaky pro testování")
            
            del testData
            del featuresTool
            
            clsT=Classification(self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"], self.createParamsForClassifiers(),
                           self.configAll[ConfigManager.sectionClassification]["WEIGHT_AUTO_CV"])
            

            if self.configAll[ConfigManager.sectionClassification]["BALANCING"]:
                logging.info("začátek vyvažování trénovací množiny")
                bal=Balancing(self.configAll[ConfigManager.sectionClassification]["BALANCING"])
                
                extracted, trainTargets=bal.balance(extracted, trainTargets)

                logging.info("konec vyvažování trénovací množiny")
            
            
            clsT.train(extracted, trainTargets, workers=self.configAll[ConfigManager.sectionClassification]["WORKERS"])
            del extracted

            if self.configAll[ConfigManager.sectionPredict]["USE_PROB"]:
                predicted=clsT.predictAuto(extractedTest, self.partSize, 
                                        self.configAll[ConfigManager.sectionPredict]["THRESHOLD"],
                                        self.configAll[ConfigManager.sectionPredict]["WORKERS"])
            else:
                predicted=clsT.predict(extractedTest, self.partSize, 
                                        self.configAll[ConfigManager.sectionPredict]["WORKERS"])
            
            targetsNames=[]

            if self.configAll[ConfigManager.sectionPredict]["USE_PROB"] and clsT.couldGetNBest():
                targetsNames=clsT.targets
            
            
            del extractedTest
            del clsT
            
            predictedTargets=Testing.selectBest(predicted, targetsNames)
            
            
            if args.writeResults:
                if self.configAll[ConfigManager.sectionTesting]["PREDICT_MATCH_FIELD_NAME"]:
                    testAddData[self.configAll[ConfigManager.sectionTesting]["PREDICT_MATCH_FIELD_NAME"]]=Testing.matchPredictTarget(predictedTargets, testTargets)
                
                predictResult.write(predicted, testAddData, targetsNames, writteHeader)
                
                
                if args.sepResults and actPart<numOfPats:
                    writeResults.close()
                    
                    bNameSplit=os.path.splitext(os.path.basename(args.writeResults))
                    
                    writeResults=open(os.path.join(os.path.dirname(args.writeResults), bNameSplit[0]+str(actPart+1)+bNameSplit[1]), "w")
                    predictResult.writeTo=writeResults
                    
                else:
                    writteHeader=False
                
            if args.writeConfMetrix:
                Testing.writeConfMatTo(predictedTargets, testTargets, writeConfMat)
                
            
            testStats.processResults(predictedTargets, predicted, testTargets, targetsNames, trainTargets)
        
            crossValScore=testStats.crossValidationScores()
            print("\nHotovo "+str(int(round(actPart/numOfPats, 2)*100))+"% | Aritmetický průměr metrik z kroků křížové validace ("+self.configAll[ConfigManager.sectionTesting]["SPLIT_METHOD"]+").")
            print("\nCíle (Pokud nějaký cíl v kroku chybí, tak je odstraněn z průměrování):")
            testStats.printTargetsCVScore(crossValScore)
            
            print("\nPočet dokumentů, které se nepodařilo klasifikovat:")
            print("\tPrůměrný: "+str(testStats.unclassified/actPart))
            print("\tCelkový: "+str(testStats.unclassified))
            
            print("\nSprávnosti pokud bereme v úvahu alespoň jeden dobrý cíl z n nejlepších:")
            testStats.printNBestCVScore(crossValScore)
            
            print("\nPrůměry pro precision, recall a fscore:")
            testStats.printAVGCVScore(crossValScore)
            
            sys.stdout.flush()
        
        if args.writeResults:
            writeResults.close()
        
        if args.writeConfMetrix:
            writeConfMat.close()
            
    def stats(self, args): 
        self.__initCheck(args)       
        
        try:
            docReader=DocReaderMetadata(metadataFile=args.input, 
                nonEmpty=self.configAll[ConfigManager.sectionGetData]["NON_EMPTY"], 
                empty=self.configAll[ConfigManager.sectionGetData]["EMPTY"], 
                fieldRegex={dataName: (re.compile(x[0]),x[1]) for dataName, x in self.configAll[ConfigManager.sectionGetData]["FIELD_REGEX"].items()}, 
                minPerField=self.configAll[ConfigManager.sectionGetData]["MIN_PER_FIELD"], 
                maxPerField=self.configAll[ConfigManager.sectionGetData]["MAX_PER_FIELD"],
                itemDelimiter=self.configAll[ConfigManager.sectionGetData]["ITEM_DELIMITER"], 
                selectItems=self.configAll[ConfigManager.sectionGetData]["SELECT_ITEMS"])
            
        except DocReaderNeedDataFile:
            raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS)+" Je nutné zadat cestu k datovému souboru s dokumenty.", 
                        ErrorMessenger.CODE_INVALID_ARGUMENTS)
        except DocReaderInvalidMetadataFields:
            raise ExceptionMessageCode(
                ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" V konfiguračním souboru je název pole metadat, který není ve vstupní souboru.",
                ErrorMessenger.CODE_INVALID_CONFIG)

        numberOfDocuments=0
        
        
        tarFieldName=self.configAll[ConfigManager.sectionGetData]["TARGET_FIELD"]
        preFieldName=self.configAll[ConfigManager.sectionPredict]["PREDICTED_FIELD_NAME"]
        
        getTar=tarFieldName in docReader.fieldNames
        getPre=preFieldName in docReader.fieldNames
        
        targets=[]
        predicted=[]
        

        logging.info("začátek získávání dat pro statistiku")
        for doc in docReader:
            numberOfDocuments+=1
            
            if getTar:
                targets.append(doc[self.configAll[ConfigManager.sectionGetData]["TARGET_FIELD"]][0])
            if getPre:
                predicted.append(doc[self.configAll[ConfigManager.sectionPredict]["PREDICTED_FIELD_NAME"]][0])
                
        logging.info("konec získávání dat pro statistiku")
        
        
        logging.info("začátek vypisování počtů")
        print("Počet dokumentů: "+str(numberOfDocuments))
        if getTar or getPre:
            print("Kategorie: ")
        if getTar:
            print("\tPočet pravých kategorií: "+str(len(set(targets))))
            print("\tPočty dokumentů v pravých kategoriích: ")
            dataStats=self.__makeStats(targets)
            for name, val in sorted(dataStats.items(), key=lambda x: x[1], reverse=True):
                print("\t\t"+name+"\t"+str(val))
                
        if getPre:
            print("\tPočet predikovaných kategorií: "+str(len(set(predicted))))
            print("\tPočty dokumentů v predikovaných kategoriích: ")
            dataStats=self.__makeStats(predicted)
            for name, val in sorted(dataStats.items(), key=lambda x: x[1], reverse=True):
                print("\t\t"+name+"\t"+str(val))
        
        
        logging.info("konec vypisování počtů")
        
        
            
        if tarFieldName in docReader.fieldNames and preFieldName in docReader.fieldNames:
            logging.info("začátek vyhodnocování úspěšnosti")
            test=Testing(self.configAll[ConfigManager.sectionDefault]["HIER_DELIMITER"])
            test.processResults(predicted, predicted, targets, [], targets)
            
            crossValScore=test.crossValidationScores()
            print("Cíle (Pokud nějaký cíl v kroku chybí, tak je odstraněn z průměrování):")
            test.printTargetsCVScore(crossValScore)
            
            print("\nPočet dokumentů, které se nepodařilo klasifikovat (celkový počet):")
            print("\t"+str(test.unclassified))
            
            print("\nSprávnosti pokud bereme v úvahu alespoň jeden dobrý cíl z n nejlepších:")
            test.printNBestCVScore(crossValScore)
            
            print("\nPrůměry pro precision, recall a fscore:")
            test.printAVGCVScore(crossValScore)
            
            logging.info("konec vyhodnocování úspěšnosti")
            
            
        
        
    def _getConfigForVocaBuilding(self):
        """
        Získání configu pro vytvážení slovníků při extrakci příznaků.
        Je tvořen zohledněním polí FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON a META_VECTORIZERS_BUILD_VOCABULARY_ON
        k dosavadnímu self.configAll

        :return: konfigurace
        """
        
        actUseConfig=self.configAll
  
        if self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"] or \
            self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
            actUseConfig=copy.deepcopy(self.configAll)
            
            if self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"]:
                if ConfigManager.fulltextName != self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"] and \
                    self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"] not in actUseConfig[ConfigManager.sectionGetData]["GET_META_FIELDS"]:
                    actUseConfig[ConfigManager.sectionGetData]["GET_META_FIELDS"]+=[self.configAll[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON"]]
            
            if self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
                for _, mData in self.configAll[ConfigManager.sectionFeatures]["META_VECTORIZERS_BUILD_VOCABULARY_ON"]:
                    if ConfigManager.fulltextName == mData:
                        actUseConfig[ConfigManager.sectionGetData]["GET_FULLTEXT"]=True
                         
                    elif mData not in actUseConfig[ConfigManager.sectionGetData]["GET_META_FIELDS"]:
                        actUseConfig[ConfigManager.sectionGetData]["GET_META_FIELDS"]+=[mData]
                        
        return actUseConfig
    
    def __invalidExtractingMethod(self, config, classifiers):
        """
        Určí zda-li se jedná o nevalidní kombinaci klasifikátoru a metody pro extrakci příznaků.
        Pokud je zvolen MultinomialNB a pro extrakci příznaků je použito Doc2Vec nebo HashingVectorizer s NON_NEGATIVE=false
        
        :param config: Konfigurace použitá pro získávání příznaků.
        :param classifiers: Struktura obsahující trojice (název dat, název klasifikátoru, váha)
        :returns: True -> nevalidní. False ->  validní
        """
        
        allClsNames=set([ x[1] for x in classifiers])
        
        if Classification.multinomialNBName not in allClsNames:
            return False
        
        invVec=[Features.doc2VecName, Features.hashingVectorizerName]
        

        if config[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER"] and \
            config[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER"].lower() in invVec and \
                Classification.multinomialNBName in [x[1] for x in classifiers if x[0]==ConfigManager.fulltextName]:
            
            if (config[ConfigManager.sectionFeatures]["FULL_TEXT_VECTORIZER"].lower()!=Features.hashingVectorizerName or 
                    not config[ConfigManager.sectionHashingVectorizer]["NON_NEGATIVE"]):
                return True

            
        if config[ConfigManager.sectionFeatures]["META_VECTORIZERS"]:

            for dataName, vectName in config[ConfigManager.sectionFeatures]["META_VECTORIZERS"].items():
                if vectName in invVec and \
                    Classification.multinomialNBName in [x[1] for x in classifiers if x[0]==dataName]:
                
                    if vectName in invVec and (vectName!=Features.hashingVectorizerName or 
                        not config[ConfigManager.sectionHashingVectorizer]["NON_NEGATIVE"]):
                        return True
        
        return False
    
    def createParamsForClassifiers(self):
        """
        Na základě konfigurace, vytvoří parametry klasifikátorů.
        :returns: Parametry pro klasifikátory. Pro parametr classifierParams konstruktoru Classification.
        """
        
        clsPar={}
            
            
        if Classification.KNeighborsClassifierName in [x[1] for x in self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"]]:
            clsPar[Classification.KNeighborsClassifierName]={
                "n_neighbors":self.configAll[ConfigManager.sectionKNeighborsClassifier]["NUM_OF_NEIGHBORS"],
                "n_jobs":self.configAll[ConfigManager.sectionKNeighborsClassifier]["WORKERS"],
                "weights":self.configAll[ConfigManager.sectionKNeighborsClassifier]["WEIGHTS"]
            }
            
        if Classification.SGDClassifierName in [x[1] for x in self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"]]:
            
            clsPar[Classification.SGDClassifierName]={
                "loss":self.configAll[ConfigManager.sectionSGDClassifier]["LOSS"]
            }
            
        if Classification.matchTargetClassifierName in [x[1] for x in self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"]]:

            try:

                lemmatizer=Lemmatizer(self.configAll[ConfigManager.sectionPreprocessing]["DICT"])


                clsPar[Classification.matchTargetClassifierName]={
                    "lemmatizer":lemmatizer
                }
            except LemmatizerException:
                raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Problém s dict.",
                                               ErrorMessenger.CODE_INVALID_CONFIG)   
            
        return clsPar
    
    
def killChilds():
    """
    Zabije všechny potomky.
    """
    
    for p in multiprocessing.active_children():
        p.terminate()
        
if __name__ == "__main__":
    
    try:
        cpkClassifier=CPKclassifier(os.path.dirname(os.path.realpath(__file__))+'/config/config.ini');
        args = ArgumentsManager.parseArgs(cpkClassifier)

        if args is not None:
            if args.log:
                logging.basicConfig(filename=args.log,format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
                
            logging.info("začátek")

            args.func(args)
            logging.info("konec")
    except ExceptionMessageCode as e:
        killChilds()
        ErrorMessenger.echoError(e.message, e.code)
    except IOError as e:
        killChilds()
        ErrorMessenger.echoError(ErrorMessenger.getMessage(ErrorMessenger.CODE_COULDNT_WORK_WITH_FILE)+"\n"+str(e), 
                                 ErrorMessenger.CODE_COULDNT_WORK_WITH_FILE)
    except KeyboardInterrupt:
        killChilds()
    except SystemExit:
        killChilds()
        raise
    except Exception as e: 
        killChilds()
        print("--------------------", file=sys.stderr)
        print("Detail chyby:\n", file=sys.stderr)
        traceback.print_tb(e.__traceback__)
        
        print("--------------------", file=sys.stderr)
        print("Text: ", end='', file=sys.stderr)
        print(e, file=sys.stderr)
        print("--------------------", file=sys.stderr)
        ErrorMessenger.echoError(ErrorMessenger.getMessage(ErrorMessenger.CODE_UNKNOWN_ERROR), ErrorMessenger.CODE_UNKNOWN_ERROR)
    