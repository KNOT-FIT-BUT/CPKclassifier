#!/usr/bin/python3
# -*- coding: UTF-8 -*-
"""
DocClassifier -- Systém pro klasifikaci dokumentů.
Obsahuje také nástroje pro předzpracování, vyvažování a testování.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""
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

from DocClassifierPack.utils.DataSet import DataSet, DataSetInvalidMetadataFields, DataSetNoDataPath, DataSetInvalidTarget
from DocClassifierPack.DocClassifierDataDump import DocClassifierDataDumpInvalidFile, DocClassifierDataDump

from DocClassifierPack.preprocessing.Preprocessing import Preprocessing, LemmatizerTaggerException
from DocClassifierPack.features.Features import Features, FeaturesNoData
from DocClassifierPack.classification.Classification import Classification
from DocClassifierPack.balancing.Balancing import Balancing
from DocClassifierPack.prediction.Prediction import Prediction
from DocClassifierPack.testing.SplitTestSet import SplitTestSet
from DocClassifierPack.testing.Testing import Testing


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
    Manažér argumentů pro DocClassifier.
    """
    
    @classmethod
    def parseArgs(cls, docClassifier):
        """
        Parsování argumentů a výpis chybové zprávy.
        
        :param cls: arguments
        :param docClassifier: Object třídy DocClassifier.
        
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
                help="Vezme pouze slova, která mají počet znaků větší nebo roven MINWORDLENGTH (po případné lemmatizací/extrakci/sepSigns).")
        
        parserPreprocessing.add_argument("--maxWordLength", type=int,
                help="Vezme pouze slova, která mají počet znaků menší nebo roven MAXWORDLENGTH (po případné lemmatizací/extrakci/sepSigns).")
        
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
        parserPreprocessing.set_defaults(func=docClassifier.preprocessing)

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
        parserGetData.set_defaults(func=docClassifier.getDataArgs)

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
        parserFeatures.set_defaults(func=docClassifier.featuresExtracting)
        
        
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
        parserClassification.set_defaults(func=docClassifier.classification)
        
        
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
        parserPredict.set_defaults(func=docClassifier.predict)
        
        
        parserTesting = subparsers.add_parser('testing', help='Testování klasifikátoru. Řídí se nastavením v konfiguračním souboru. Statistiku píše do stdout.')
        parserTesting.add_argument("--data", type=str,
                help="Vstupní datový soubor.")
        parserTesting.add_argument("--metadata", type=str,
                help="Vstupní metadatový soubor. (povinné)", required=True)
        
        parserTesting.add_argument("--writeResults", type=str,
                help="Cesta k souboru, kde má být uložen výsledek predikcí v jednotlivích iteracích. Nastavení na základě sekce PREDICTION v konfiguračním souboru.")
        
        
        parserTesting.add_argument("--writeConfMetrix", type=str,
                help="Cesta k souboru kde budou uloženy metice záměn.")
        
        parserTesting.add_argument("--consistency", action='store_true',
                help="Natrénuje na všech dostupných datech a na těchto datech i natrénovaný klasifikátor otestuje.")
        
        
        parserTesting.add_argument("--config", type=str,
                help="Tento konfigurační soubor přenastaví parametry z defaultního konfiguračního souboru. (Pouze uvedené)")
        parserTesting.add_argument("--log", type=str,
                help="Kam uložit logovací soubor.")
        parserTesting.set_defaults(func=docClassifier.testing)
        
        
        subparsersForHelp={
            'preprocessing':parserPreprocessing,
            'getData':parserGetData,
            'features':parserFeatures,
            'classification':parserClassification,
            'prediction':parserPredict,
            'testing':parserTesting
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
            "TAGGER":None,
            "TAGGER_POS":None,
            "STOP_WORDS":None
            }
        
        if preprocessing["TAGGER"]: 
            
            if preprocessing["TAGGER"][0]!="/":
                result["TAGGER"]=os.path.dirname(os.path.realpath(__file__))+"/"+preprocessing["TAGGER"]
            else:
                result["TAGGER"]=preprocessing["TAGGER"]
                
            if not os.path.isfile(result["TAGGER"]):
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionPreprocessing+\
                                           " v parametru TAGGER není cesta k existujícímu souboru.",
                                               ErrorMessenger.CODE_INVALID_CONFIG)
                
           
        if preprocessing["TAGGER_POS"]:
            if preprocessing["TAGGER_POS"][0]!="/":
                result["TAGGER_POS"]=os.path.dirname(os.path.realpath(__file__))+"/"+preprocessing["TAGGER_POS"]
            else:
                result["TAGGER_POS"]=preprocessing["TAGGER_POS"]
            
            if not os.path.isfile(result["TAGGER_POS"]):
                raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionPreprocessing+\
                                           " v parametru TAGGER_POS není cesta k existujícímu souboru.",
                                               ErrorMessenger.CODE_INVALID_CONFIG)
            
        if preprocessing["STOP_WORDS"]:
            result["STOP_WORDS"]=shlex.split(preprocessing["STOP_WORDS"])
        
        return result
    
    def __transformGetDataVals(self):
        """
        Převede hodnoty pro sekci GET_DATA a validuje je.
        
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        result={
            "TARGET_FIELD":None,
            "ITEM_DELIMITER": None
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
                    result["FIELD_REGEX"][k]=(re.compile(value),0)
                except re.error:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. Nevalidní hodnota regulárního výrazu ve FIELD_REGEX "+k+":"+value,
                                               ErrorMessenger.CODE_INVALID_CONFIG)
        
        result["MIN_PER_FIELD"]=self.__createDict(getData["MIN_PER_FIELD"], numValues="int")
        result["MAX_PER_FIELD"]=self.__createDict(getData["MAX_PER_FIELD"], numValues="float")

        if getData["ITEM_DELIMITER"]:
            result["ITEM_DELIMITER"]=shlex.split(getData["ITEM_DELIMITER"])[0]
        
        result["SELECT_WORDS"]=self.__parseSelectorVal(getData["SELECT_WORDS"])
        result["SELECT_ITEMS"]=self.__createDict(getData["SELECT_ITEMS"], convertSelectors=True)
        
        
        result["GET_FULLTEXT"]=getData["GET_FULLTEXT"].lower()=="true"
        
        result["GET_META_FIELDS"]=shlex.split(getData["GET_META_FIELDS"])
        
        if self.fulltextName in result["GET_META_FIELDS"]:
            raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. GET_META_FIELDS nesmí obsahovat název "+self.fulltextName+", který je vyhrazen pouze pro data: "+getData["GET_META_FIELDS"],
                                       ErrorMessenger.CODE_INVALID_CONFIG)

        
        return result
    
    def __transformFeaturesVals(self, transformedGetData):
        """
        Převede hodnoty pro sekci FEATURES a validuje je.
        
        :param transformedGetData: Pro validaci je nutné, poskytnout převedenou sekci get data.
        :returns: dict -- ve formátu jméno prametru jako klíč a k němu hodnota parametru
        """
        
        result={}
        
        features=self.configParser[self.sectionFeatures]
        
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
            "BALANCING":None
            }
            
        clsFromConfig=self.configParser[self.sectionClassification]["CLASSIFIER"]
        if clsFromConfig:
            for dataName, clsName, clsWeight in self.__createTriplet(clsFromConfig):
                clsName=clsName.lower()
                
            
                if clsName not in Classification.classifiersNames:
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionClassification+" u parametru CLASSIFIER je uveden neznámý klasifikátor: "+ clsName,
                            ErrorMessenger.CODE_INVALID_CONFIG)
                    
                if dataName not in transformedGetData["GET_META_FIELDS"]+[self.fulltextName]:
                    raise ExceptionMessageCode(
                        "Nevalidní hodnota v konfiguračním souboru. V sekci "+self.sectionClassification+" u parametru CLASSIFIER je uveden neznámý název dat: "+ dataName,
                            ErrorMessenger.CODE_INVALID_CONFIG)
                    
                
                try:
                    clsWeight = float(clsWeight)
                except ValueError:
                    raise ExceptionMessageCode("Nevalidní hodnota v konfiguračním souboru. V sekci CLASSIFICATION u CLASSIFIER (musí být jako váha uvedeno číslo): "+str(clsWeight),
                                ErrorMessenger.CODE_INVALID_CONFIG)
                    
                
                    
                
                result["CLASSIFIER"].append((dataName, clsName, clsWeight))
                
        allDataNames=[xName for xName, _, _ in result["CLASSIFIER"]]
        
        if transformedGetData["GET_FULLTEXT"] and self.fulltextName not in allDataNames:
            result["CLASSIFIER"].append((self.fulltextName,Classification.linearSVCName, 1))
            
        if transformedGetData["GET_META_FIELDS"]:
            for mName in transformedGetData["GET_META_FIELDS"]:
                if mName not in allDataNames:
                    result["CLASSIFIER"].append((mName,Classification.linearSVCName, 1))
                    
                    
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
            "USE_PROB":True
            }
        
        
        predict=self.configParser[self.sectionPredict]
        if predict["PREDICTED_FIELD_NAME"]:
            result["PREDICTED_FIELD_NAME"]=shlex.split(predict["PREDICTED_FIELD_NAME"])[0]
            
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
        :returns: list -- trojic (klíč, hodnota)
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
        
        
        

class DocClassifier(object):
    """
    Nástroj pro klasifikaci dokumentů, hledání podobných dokumentů a předzpracování dokumentů.
    """

    
    
    def __init__(self, config, logAfterLines=100):
        """
        Inicializace.
        
        :param config: Cesta k souboru se základní konfigurací.
        :param logAfterLines: Udává po kolika přečtených řádcích má dojit k logování.
        """
        self.logAfterLines=logAfterLines
        self.initialConfig=config
        
        self.partSize=500    #používá se pro části u predikce nebo u hledání podobných dokumentů
    
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
         
        if self.configAll[ConfigManager.sectionPreprocessing]["TAGGER"] is None:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Chybí TAGGER.",
                                           ErrorMessenger.CODE_INVALID_CONFIG)
            
        if self.configAll[ConfigManager.sectionPreprocessing]["TAGGER_POS"] is None:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Chybí TAGGER_POS.",
                                           ErrorMessenger.CODE_INVALID_CONFIG)   
        
        try:
            Preprocessing(args, self.configAll[ConfigManager.sectionPreprocessing]["STOP_WORDS"], 
                      self.configAll[ConfigManager.sectionPreprocessing]["TAGGER"], 
                      self.configAll[ConfigManager.sectionPreprocessing]["TAGGER_POS"], self.logAfterLines).start()
        except LemmatizerTaggerException:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" Problém s taggerem.",
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
        
        dSet=self.__getDataSet(args)        
            
        trainData, trainTargets=self.__getData(args, useDataset=dSet)
        
        
        extracted, features=self.__performFeaturesExtracting(trainData)
        
        for dataName, dFeat in extracted.items():
            logging.info("Velikost vektoru pro "+dataName+": "+str(dFeat.shape[1]))
        
        DocClassifierDataDump(targets=trainTargets, featuresTool=features, features=extracted, configFea=self.configAll).save(args.saveTo)


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
                            fieldRegex=useConfig[ConfigManager.sectionGetData]["FIELD_REGEX"], 
                            minPerField=useConfig[ConfigManager.sectionGetData]["MIN_PER_FIELD"], 
                            maxPerField=useConfig[ConfigManager.sectionGetData]["MAX_PER_FIELD"], 
                            itemDelimiter=useConfig[ConfigManager.sectionGetData]["ITEM_DELIMITER"], 
                            selectWords=useConfig[ConfigManager.sectionGetData]["SELECT_WORDS"], 
                            selectItems=useConfig[ConfigManager.sectionGetData]["SELECT_ITEMS"], 
                            fulltextName=ConfigManager.fulltextName)
            
        except DataSetInvalidMetadataFields:
            raise ExceptionMessageCode(
                ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_CONFIG)+" V konfiguračním souboru je název pole metadat, který není ve vstupní souboru s metadaty.",
                ErrorMessenger.CODE_INVALID_CONFIG)
        except DataSetNoDataPath:
            raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS)+" Je nutné zadat cestu k datovému souboru s dokumenty.", 
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
        
        
    def __performFeaturesExtracting(self, dataToExtract, retFeatTool=True):
        """
        Provede extrakci příznaků.
        
        :param args: Argumenty z argument manažéru.
        :param dataToExtract: Data pro extrakci
        :retFeatTool retFeatTool: True
        :returns: Dvojici (extrahované příznaky, Features). Pokud retFeatTool false => extrahované příznaky.
        """
        
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
                fulltextName=ConfigManager.fulltextName)
        except FeaturesNoData:
            raise ExceptionMessageCode(
                    ErrorMessenger.getMessage(ErrorMessenger.CODE_NO_INPUT_DATA)+" Žádná data pro extrakci příznaků.", 
                        ErrorMessenger.CODE_NO_INPUT_DATA)

        extracted=features.extractAndLearn(dataToExtract, self.partSize)
        
        
        if retFeatTool:
            return (extracted, features)
        
        return extracted
    
    def __logStats(self, data):
        """
        Loguje počet výskytů k dané položce.
        
        :param data: list -- obsahující položky
        """
        
        dataStats={}
        for ite in data:
            if ite not in dataStats:
                dataStats[ite]=0
            dataStats[ite]=dataStats[ite]+1
        
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
            
            loadData=DocClassifierDataDump()
            
            try:
                loadData.loadBasicData(args.features)
                loadData.loadExtractedFeatures(args.features)
            except DocClassifierDataDumpInvalidFile:
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
            dSet=self.__getDataSet(args)
            
            
            trainData, trainTargets=self.__getData(args, useDataset=dSet)
            extracted, featuresTool=self.__performFeaturesExtracting(trainData)
            
        for dataName, dFeat in extracted.items():
            logging.info("Velikost vektoru pro "+dataName+": "+str(dFeat.shape[1]))
            
        saveTo=DocClassifierDataDump(
            targets=trainTargets,
            features=extracted, 
            configFea=config, 
            configCls=self.configAll)
            
        if args.features:
            copyfile(args.features+DocClassifierDataDump.featuresToolExtension, args.saveTo+DocClassifierDataDump.featuresToolExtension)
        else:
            saveTo.addFeaturesTool(featuresTool)
            
        saveTo.save(args.saveTo)
        
        if featuresTool:
            featuresTool.flush()
            
        if Classification.KNeighborsClassifierName in [x[1] for x in self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"]]:
            cls=Classification(self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"],
                                        {
                                        Classification.KNeighborsClassifierName:{
                                            "n_neighbors":self.configAll[ConfigManager.sectionKNeighborsClassifier]["NUM_OF_NEIGHBORS"],
                                            "n_jobs":self.configAll[ConfigManager.sectionKNeighborsClassifier]["WORKERS"],
                                            "weights":self.configAll[ConfigManager.sectionKNeighborsClassifier]["WEIGHTS"]
                                            }
                                                                                             })
        else:
            cls=Classification(self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"])
        
        if self.configAll[ConfigManager.sectionClassification]["BALANCING"]:
            logging.info("začátek vyvažování trénovací množiny")
            bal=Balancing(self.configAll[ConfigManager.sectionClassification]["BALANCING"])
            
            extracted, trainTargets=bal.balance(extracted, trainTargets)
            logging.info("konec vyvažování trénovací množiny")
            
        cls.train(extracted, trainTargets)
        
        logging.info("Natrénováno cílů: "+str(len(set(trainTargets))))
        logging.info("Natrénováno na "+str(len(trainTargets))+" dokumentech.")
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
        
        loadData=DocClassifierDataDump()
            
        try:
            loadData.loadClassificator(args.classifiers)
            loadData.loadFeaturesTool(args.classifiers)
            loadData.loadBasicData(args.classifiers)
        except DocClassifierDataDumpInvalidFile:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_INPUT_FILE),
                    ErrorMessenger.CODE_INVALID_INPUT_FILE)
                
                
        if loadData.featuresTool is None or loadData.classificator is None or loadData.configFea is None:
                raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_INPUT_FILE),
                                           ErrorMessenger.CODE_INVALID_INPUT_FILE)
                
        if loadData.configFea[ConfigManager.sectionGetData]["GET_FULLTEXT"] and not args.data:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS+" Je nutné přidat datový soubor s plnými texty."),
                                           ErrorMessenger.CODE_INVALID_ARGUMENTS)
            
        if loadData.configFea[ConfigManager.sectionGetData]["GET_META_FIELDS"] and not args.metadata:
            raise ExceptionMessageCode(ErrorMessenger.getMessage(ErrorMessenger.CODE_INVALID_ARGUMENTS+" Je nutné přidat metadatový soubor."),
                                           ErrorMessenger.CODE_INVALID_ARGUMENTS)
                
        testData=self.__getData(args, False)
        
        allKinds=set(loadData.configFea[ConfigManager.sectionGetData]["GET_META_FIELDS"])
        if loadData.configFea[ConfigManager.sectionGetData]["GET_FULLTEXT"]:
            allKinds.add(ConfigManager.fulltextName)

        
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
            configForWrite=self.configAll.copy()
            configForWrite[ConfigManager.sectionGetData]["GET_META_FIELDS"]=self.configAll[ConfigManager.sectionPredict]["WRITE_META_FIELDS"]
            configForWrite[ConfigManager.sectionGetData]["GET_FULLTEXT"]=False
            
            metaForWrite=self.__getData(args, targets=False, useConfig=configForWrite)
        
        
        if self.configAll[ConfigManager.sectionPredict]["USE_PROB"]:
            predicted=loadData.classificator.predictAuto(loadData.featuresTool.extract(testData, self.partSize), self.partSize)
        else:
            predicted=loadData.classificator.predict(loadData.featuresTool.extract(testData, self.partSize), self.partSize)
            
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
            

        allData, allTargets=self.__getData(args)

        additionalData={}
        if self.configAll[ConfigManager.sectionPredict]["WRITE_META_FIELDS"]:
            configForWrite=copy.deepcopy(self.configAll)
            configForWrite[ConfigManager.sectionGetData]["GET_META_FIELDS"]=self.configAll[ConfigManager.sectionPredict]["WRITE_META_FIELDS"]
            configForWrite[ConfigManager.sectionGetData]["GET_FULLTEXT"]=False
        
            additionalData=self.__getData(args, targets=False, useConfig=configForWrite)
        
        
        
        
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
        
        for trainData, trainTargets, testData, testTargets, testAddData in spliter:
            actPart=actPart+1

            if not args.consistency:
                logging.info("Křížová validace: "+str(actPart)+"/"+str(numOfPats))
            logging.info("Dokumentů pro trénování: "+str(len(trainTargets)))
            logging.info("Dokumentů pro testování: "+str(len(testTargets)))
            
            extracted, featuresTool=self.__performFeaturesExtracting(trainData)
            
            logging.info("začátek příznaky pro testování")
            extractedTest=featuresTool.extract(testData, self.partSize)
            logging.info("konec příznaky pro testování")
            
            for dataName, dFeat in extracted.items():
                logging.info("Velikost vektoru pro "+dataName+": "+str(dFeat.shape[1]))
                
            del featuresTool
            
            
            if Classification.KNeighborsClassifierName in [x[1] for x in self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"]]:
                clsT=Classification(self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"],
                                        {
                                        Classification.KNeighborsClassifierName:{
                                            "n_neighbors":self.configAll[ConfigManager.sectionKNeighborsClassifier]["NUM_OF_NEIGHBORS"],
                                            "n_jobs":self.configAll[ConfigManager.sectionKNeighborsClassifier]["WORKERS"],
                                            "weights":self.configAll[ConfigManager.sectionKNeighborsClassifier]["WEIGHTS"]
                                            }
                                                                                             })
            else:
                clsT=Classification(self.configAll[ConfigManager.sectionClassification]["CLASSIFIER"])
            

            if self.configAll[ConfigManager.sectionClassification]["BALANCING"]:
                logging.info("začátek vyvažování trénovací množiny")
                bal=Balancing(self.configAll[ConfigManager.sectionClassification]["BALANCING"])
                
                extracted, trainTargets=bal.balance(extracted, trainTargets)

                logging.info("konec vyvažování trénovací množiny")
            
            
            clsT.train(extracted, trainTargets)
            del extracted
            
            if self.configAll[ConfigManager.sectionPredict]["USE_PROB"]:
                predicted=clsT.predictAuto(extractedTest, self.partSize)
            else:
                predicted=clsT.predict(extractedTest, self.partSize)
            
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
                writteHeader=False
                
            if args.writeConfMetrix:
                Testing.writeConfMatTo(predictedTargets, testTargets, writeConfMat)
                
            
            testStats.processResults(predictedTargets, predicted, testTargets, targetsNames, trainTargets)
        
            crossValScore=testStats.crossValidationScores()
            print("\nHotovo "+str(int(round(actPart/numOfPats, 2)*100))+"% | Aritmetický průměr metrik z kroků křížové validace ("+self.configAll[ConfigManager.sectionTesting]["SPLIT_METHOD"]+").")
            print("\nCíle (Pokud nějaký cíl v kroku chybí, tak je odstraněn z průměrování):")
            testStats.printTargetsCVScore(crossValScore)
            
            print("\nPřesnosti pokud bereme v úvahu alespoň jeden dobrý cíl z n nejlepších:")
            testStats.printNBestCVScore(crossValScore)
            
            print("\nPrůměry pro precision, recall a fscore:")
            testStats.printAVGCVScore(crossValScore)
            
            sys.stdout.flush()
        
        if args.writeResults:
            writeResults.close()
        
        if args.writeConfMetrix:
            writeConfMat.close()
            
    
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
        
if __name__ == "__main__":
    try:
        docClassifier=DocClassifier(os.path.dirname(os.path.realpath(__file__))+'/config/config.ini');
        args = ArgumentsManager.parseArgs(docClassifier)
        if args is not None:
            if args.log:
                logging.basicConfig(filename=args.log,format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
                
            logging.info("začátek")
            args.func(args)
            logging.info("konec")
    except ExceptionMessageCode as e:
        ErrorMessenger.echoError(e.message, e.code)
    except IOError as e:
        ErrorMessenger.echoError(ErrorMessenger.getMessage(ErrorMessenger.CODE_COULDNT_WORK_WITH_FILE)+"\n"+str(e), 
                                 ErrorMessenger.CODE_COULDNT_WORK_WITH_FILE)
    except SystemExit:
        raise
    except Exception as e: 
        
        print("--------------------")
        print("Detail chyby:\n")
        traceback.print_tb(e.__traceback__)
        
        print("--------------------")
        print("Text: ", end='')
        print(e)
        print("--------------------")
        ErrorMessenger.echoError(ErrorMessenger.getMessage(ErrorMessenger.CODE_UNKNOWN_ERROR), ErrorMessenger.CODE_UNKNOWN_ERROR)
    