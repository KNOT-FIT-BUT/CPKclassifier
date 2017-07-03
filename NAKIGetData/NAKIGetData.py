#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Uses config.ini for configuration.
"""
Contains class for getting data from NAKI project to suitable format for DocClasifier.

@author:     Martin Dočekal
@contact:    xdocek09@stud.fit.vubtr.cz
"""
import os
import sys
import csv
import re
import logging
import configparser
import copy

from argparse import ArgumentParser, ArgumentTypeError
from asyncore import write

from ufal.morphodita import *

class LemmatizerException(Exception):
    pass

class Lemmatizer(object):
    """
    Třída pro lemmatizaci slov a extrakci vybraných slovních druhů. Používá nástroj morphodita.

    Značky slovních druhů:
        N - 1
        A - 2
        P - 3
        C - 4
        V - 5
        D - 6
        R - 7
        J - 8
        T - 9
        I - 10
        Z - Symboly
        X - Neznámé
    """
    __POSTranslaterNum={
        1 : "N", 
        2 : "A", 
        3 : "P", 
        4 : "C", 
        5 : "V", 
        6 : "D", 
        7 : "R", 
        8 : "J", 
        9 : "T", 
        10 : "I"
        }
    
    def __init__(self, dict):
        """
        Konstrukce objektu.
        
        :param dict: Cesta k souboru pro dict morphodity.
        :raises LemmatizerException: Když není definovaný tokenizer pro dodaný model. Nebo nevalidní slovník.
        """

        self.morpho=Morpho.load(dict)
        if self.morpho is None:
            raise LemmatizerException("Chybný DICT.")
        self.tokenizer = self.morpho.newTokenizer()
        if self.tokenizer is None:
            raise LemmatizerException("Není definovaný tokenizer pro dodaný model.")

        
        self.forms = Forms()
        self.tokens = TokenRanges()
        self.lemmas = TaggedLemmas()
        self.converter=TagsetConverter.newPdtToConll2009Converter()
        
        
        

    def lemmatize(self, text):
        """
        Vrací lemmatizovanou formu slov.
        
        :param text: Text pro zpracování.
        :returns:  list -- obsahující lemmatizovaná slova
        """
        self.tokenizer.setText(text)
        words=[]
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            for word in self.forms:
                self.morpho.analyze( word , self.morpho.GUESSER, self.lemmas)
                self.converter.convert(self.lemmas[0])
                words.append(self.lemmas[0].lemma)
            
        return words
    
    def getWordsPOS(self, text):
        """
        Získá slovní druhy k jednotlivým slovům v parametru text.
        
        :param text: Text pro analýzu.
        :returns:  list -- of tuples, kde (slovo, POS)
        """
        
        self.tokenizer.setText(text)
        words=[]
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            for word in self.forms:
                self.morpho.analyze( word , self.morpho.GUESSER, self.lemmas)
                words.append((word, self.lemmas[0].tag[0]))
                    
        return words
        
        
    def extract(self, text, only, lemmatized=True):
        """
        Extrahuje pouze slova, která mají definovaný slovní druh v parametru only.
        
        :param text: Text pro extrakci.
        :param only: set -- povolených slovních druhů
        :param lemamatized: True -> vrátí lemmatizovanou formu slova.
        :returns:  list -- obsahující slova
        """
        words=[]             
        self.tokenizer.setText(text)
        
        while self.tokenizer.nextSentence(self.forms, self.tokens):
            for word in self.forms:
                self.morpho.analyze( word , self.morpho.GUESSER, self.lemmas)
                if self.lemmas[0].tag[0] in only:
                    if lemmatized:
                        self.converter.convert(self.lemmas[0])
                        words.append(self.lemmas[0].lemma)
                    else:
                        words.append(word)
 
        return words
    
    def translateNumericPOS(self, POS):
        """
        Přeloží slovní druh z číselné reprezentace do příslušné značky slovního druhu.
        
        :param POS: list -- obsahující číselnou reprezentaci slovních druhů
        :returns:  list -- obsahující značky slovních druhů
        """
        translated=[]
        for x in POS:
            try: 
                x=int(x)
            except ValueError:
                pass
            
            if x in self.__class__.__POSTranslaterNum:
                translated.append(self.__class__.__POSTranslaterNum[x])
            else:
                translated.append(x)
                
        return translated


class ArgumentParserError(Exception): pass
class ExceptionsArgumentParser(ArgumentParser):
    def error(self, message):
        raise ArgumentParserError(message)
    
class ArgumentsManager:
    """
    Manager for script arguments.
    """
    
    @classmethod
    def parseArgs(cls,):
        """
        Arguments parser. Also prints help.

        :param cls: arguments
        """
        
        parser = ExceptionsArgumentParser(description="Tool for getting data from NAKI project to suitable format for DocClasifier.")
        
        parser.add_argument("--docFolder", type=str,
                help="Folder where documents fulltexts are saved.")
        parser.add_argument("--saveDataTo", type=str,
                help="Path where to save data.")
        parser.add_argument("--saveMetaDataTo", type=str,
                help="Path where to save metadata.", required=True)
        parser.add_argument("--hierarchyFile", type=str,
                help="Maps targets into defined hierarchy.", required=True)
        parser.add_argument("--metadataFilesMap", type=str,
                help="Path to file with metadata files map.", required=True)
        parser.add_argument("--enableEmptyFulltext", action='store_true',
                help="Enables documents with empty fulltext or no fulltext at all.")
        
        

        parser.add_argument("--log", type=str,
                help="Where to save the log.")

        if len(sys.argv)<2:
            parser.print_help()
            return None
        try:
            parsed=parser.parse_args()

            if not parsed.enableEmptyFulltext and not parsed.docFolder:
                raise ArgumentParserError("Invalid combination of arguments (enableEmptyFulltext, docFolder).")


        except ArgumentParserError as e:
            parser.print_help()
            print(str(e), file=sys.stderr)
            return None

        return parsed

class FieldItem(object):
    """
    Represents one item in metadata field.
    """
    def __init__(self, value, position, priority):
        """
        Item initialization
        :param value:
            Value of item. Is used for hash.
        :param position:
            Position in field.
        :param priority:
            Priority of item. Depends on the priority of the metadata source.
        """
        self.value=value
        self.position=position
        self.priority=priority
        
    def __eq__(self, other):
        return other and self.value == other.value
    def __hash__(self):
        return hash(self.value)
    def __str__(self):
        return str(self.value)
    def __repr__(self):
        return str((self.value, self.position, self.priority))

class NAKIGetData( object):
    """
    Class for getting data from NAKI project to suitable format for DocClasifier. 
    """
    
    fieldsFlags=["008", "072", "080", "600", "610", "611", "630", "648", "100", "650", "651", "653", "655", "670", "678", "695", "964","245"]
    multiItemsFields=['080', '650', '611', '630', '653', '610', '655', '600', '964', '072', '651', '648']
    
    searchNumberInFirstPosition=re.compile(r'^[0-9].*')

    def __init__(self, config, docFolder, saveMetaDataTo, metadataFilesMap, hierarchyFile=None, saveDataTo=None):
        self.codeToNamesMap={}
        self.metaDataFiles=None
        
        self.docFolder=docFolder
        
        self.config=config["DEFAULT"]
        self.saveDataTo=saveDataTo
        self.saveMetaDataTo=saveMetaDataTo
        self.metadataFilesMap=metadataFilesMap
        self.__getMetadataFiles()
        
        self.itemDelimiter=self.config["ITEM_DELIMITER"]
        self.hierarchyDelimiter=self.config["HIERARCHY_DELIMITER"]
        
        self.hierarchyField=self.config["HIERARCHY_FIELD"]
        self.__getHierMap(hierarchyFile)
        
        self.targetsField=self.config["TARGETS_FIELD"]
        
        
        
        dictMorpho=self.config["DICT"]
        if not os.path.isabs(self.config["DICT"]):
            dictMorpho=os.path.dirname(os.path.realpath(__file__))+"/"+self.config["DICT"]

        self.lemmatizer=Lemmatizer(dictMorpho)
        self.fieldsForLematization=self.config["FIELDS_FOR_LEMMATIZATION"].split(",")
        self.lemmFieldExtension=self.config["LEMMATIZED_FIELD_EXTENSION"]

    def __getHierMap(self, hierFile):
        """
        Extracts hierarchy map from file.
        :param hierFile: Path to file with hierarchy map.
        """
        self.hierarchyFile=hierFile
        if not self.hierarchyFile:
            return
        
        logging.info("Start of getting hierarchy from file.")
        self.hierMap={}
        
        with open(self.hierarchyFile, "r") as mapFile:
            for line in mapFile:
                parts=[ x.strip() for x in line[:-1].split("\t")]
                self.hierMap[parts[0].lower()]=self.hierarchyDelimiter.join(parts[1:])
        logging.info("End of getting hierarchy from file.")
        
        
    def __getMetadataFiles(self):
        """
        Reads metadata files Map.
        """
        logging.info("Start of getting metadata files.")
        with open(self.metadataFilesMap, "r") as mapF:
            reader = csv.DictReader(mapF)
            self.metaDataFiles={}
            for row in reader:
                if not os.path.isfile(row["file"]) :
                    print("Metadata file: "+row["file"]+" does not exist.", file=sys.stderr)
                    continue
                if os.stat(row["file"]).st_size == 0:
                    print("Metadata file: "+row["file"]+" is empty.", file=sys.stderr)
                    continue
            
                self.metaDataFiles[row["file"]]=(row["name"], int(row["priority"]))
                
                
            
        logging.info("End of getting metadata files.")
        
    def write(self, metaOnly=False, enableEmptyFulltext=False):
        """
        Writes data and metadata.
        
        :param metaOnly:
            True=> writes only metadata.
        :param enableEmptyFulltext:
            Enables documents with empty fulltext or no fulltext at all.
            
        """
        
        metaData=self.getMetaData(not(enableEmptyFulltext))
        
        logging.info("Start of writting.")

        cnt=0
        


        with open(self.saveMetaDataTo, "w") as sMetaDataTo:
            fieldnames = ["dedup_record_id"]+self.fieldsFlags
            if self.hierarchyFile:
                fieldnames.insert( fieldnames.index(self.targetsField)+1, self.hierarchyField)
                
            for xName in self.fieldsForLematization:
                fieldnames.append(xName+self.lemmFieldExtension)
            
            writerMeta = csv.DictWriter(sMetaDataTo, fieldnames=fieldnames)
            writerMeta.writeheader()
            
            if metaOnly:
                for mD in metaData:
                    cnt=cnt+1
                    if cnt%100==0:
                        logging.info("Writting "+str(cnt)+". document.")
                    
                    writerMeta.writerow(self.__convertMetadataRowToWrittableFormat(mD)) 
            else:
                with open(self.saveDataTo, "w") as sDataTo:
                    for mD in metaData:
                        dLine=self.getData(list(mD["dedup_record_id"])[0].value)
                        if dLine is not None or enableEmptyFulltext:
                            if dLine is None:
                                dLine=""
                                
                            cnt=cnt+1
                            if cnt%100==0:
                                logging.info("Writting "+str(cnt)+". document.")
    
                            sDataTo.write(dLine+"\n")
    
                            writerMeta.writerow(self.__convertMetadataRowToWrittableFormat(mD))
                    

        logging.info("Total number of documents:"+str(cnt)+".")
        logging.info("End of writting.")
        
    
    def __convertMetadataRowToWrittableFormat(self, row):
        """
        Converts metadata row to writtable format
        :param row:
            Metadata row
        """
        wrRow=dict([(x,None) for x in row])
        for fieldName, items in row.items():
            wrRow[fieldName]=self.itemDelimiter.join([ x.value for x in sorted(items, key = lambda x: (-x.priority, x.position))])
        
        return wrRow
        
    def getData(self, dri):
        """
        Gets content of document with dri.
        :param dri:    Document ID.
        :returns: string -- Document with concatenated lines.
        """
        path=os.path.join(self.docFolder, str(dri)+".lemm")
        if not self.__docHaveContent(dri):
            return None
        
        with open(path, "r") as f:
            return " ".join(line.strip() for line in f)
        
    def __docHaveContent(self, dri):
        """
        Checks if document have content. Prints info messages.
        :param dri:Document ID.
        :returns: True document have content.
        """
        path=os.path.join(self.docFolder, str(dri)+".lemm")
        if not os.path.isfile(path) :
            print("File: "+path+" does not exist.", file=sys.stderr)
            return False
        
        if os.stat(path).st_size == 0:
            print("File: "+path+" is empty.", file=sys.stderr)
            return False
        return True
        
    def getMetaData(self, controlContent=True):
        """
        Gets metadata.
        
        :param controlContent: False no control if document has fulltext content.
        :returns: dict -- with metadata.
        
        """
        logging.info("Start of getting metadata.")
        
        meta=[]
        idRecords={}    #its here becasuse of search optimalization

        for metaDataFile, val in sorted(self.metaDataFiles.items(), key = lambda x: x[1][1], reverse=True):
            name, priority=val
            logging.info("Reading "+name+": "+ metaDataFile)
            with open(metaDataFile, "r") as metaFile:
                for line in metaFile:
                    mData = self.__parseMetaLine(line[:-1], priority)
                    if not mData or len(mData["dedup_record_id"])==0:
                        continue

                    dri=list(mData["dedup_record_id"])[0].value
                    if controlContent and not self.__docHaveContent(dri):
                        continue
                    
                    if dri in idRecords:
                        recordIndex=idRecords[dri]
                        self.__concatenateMetaRecords(meta[recordIndex], mData)
                        continue

                        
                    idRecords[dri]=len(meta)
                    meta.append(mData)
                
        if self.hierarchyFile:
            logging.info("Start of hierarchy mapping.")
    
            for m in meta:
                m[self.hierarchyField]=self.__hierarchyMap(m[self.targetsField])
                
            logging.info("End of hierarchy mapping.")
        
        if self.fieldsForLematization:
            logging.info("Start of metadata lemmatization.")
            for m in meta:
                for xName in self.fieldsForLematization:
                    m[xName+self.lemmFieldExtension]=set()
                for name, items in m.items():
                    if name in self.fieldsForLematization:

                        for item in items:
                            lemm_item=copy.deepcopy(item)
                            lemm_item.value=" ".join(self.lemmatizer.lemmatize(re.sub(r'\W', ' ', item.value)))
                            m[name+self.lemmFieldExtension].add(lemm_item)
            
            logging.info("End of metadata lemmatization.")
        logging.info("End of getting metadata.")
        return meta
    
    def __hierarchyMap(self, targets):
        """
        Performs code to names mapping.
        :param targets:
            For code extracting.
        :returns:  list -- With mapped targets into hierarchy.
        """
        mappedTo=[]
        
        
        for target in targets:
            tmpVal=target.value.strip().lower()
            if tmpVal not in self.hierMap:
                print("Target: "+tmpVal+" doesn't have mapping in hierarchy. This target will be skipped.", file=sys.stderr)
                continue
            
            targetHier=copy.deepcopy(target)
            targetHier.value=self.hierMap[tmpVal]
            mappedTo.append(targetHier)
            
        return mappedTo
    
    def __concatenateMetaRecords(self, first, second):
        """
        Concatenates two meta recods.
        :param first:
            First record for concatenating. This record will be supplemented.
        :param second:
            Second record for concatenating
        """
        for fName, f in first.items():
            if f==[] and second[fName]!="":
                #fill empty
                first[fName]=second[fName]
            elif fName in self.multiItemsFields and second[fName]!="":
                #multi items fields
                
                self.__unionItems(f, second[fName])
        
        
    def __getFieldItems(self, field, priority):
        """
        Get items from field.
        :param field: field containing items.
        :param priority: Defines items priority.
            Usefull for multiple source data.
        :returns: set -- Set of FieldItem.
        """
        if not field:
            return set()
        return set([FieldItem(x.strip(), i, priority) for i, x in enumerate(field.split(self.itemDelimiter))])
    
    @staticmethod
    def __unionItems(oldItems, newItems):
        """
        Concatenates two sets of items.
        :param oldItems:
            This record will be supplemented.
        :param newItems:
            Second record for concatenating
        """
        for item in newItems:
            if item in oldItems:
                alreadyIn=list(oldItems)
                alreadyIn=alreadyIn[alreadyIn.index(item)]
                
                if item.priority>=alreadyIn.priority and item.position<alreadyIn.position:
                    oldItems.remove(alreadyIn)
                    oldItems.add(item)
            else:
                oldItems.add(item)

    
    def __parseMetaLine(self, line, priority):
        """
        Parsing data from line in format:
        :param line: Line in format:
            dedup_record_id \t 008[28] \t 072 \t 080 \t 600 \t 610 \t 611 \t 630 \t 648 \t 100 \t 650
 \t 651 \t 653 \t 655 \t 670 \t 678 \t 695 \t 964 \t 245 \n
        :param priority: Defines items priority.
            Usefull for multiple source data.
        :returns:  dict - metadata
        """

        parts=line.split("\t")
        
        flags=["dedup_record_id"]+self.fieldsFlags
        
        lData={};
        if len(parts)>=len(flags):
            for i, x in enumerate(flags):
                lData[x]=self.__getFieldItems(parts[i], priority)
        
        return lData;
        
        
            
    

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(os.path.dirname(os.path.realpath(__file__))+'/config.ini')
    
    args = ArgumentsManager.parseArgs()
    if args is not None:
        if args.log:
            logging.basicConfig(filename=args.log,format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        else:
            logging.basicConfig(stream=sys.stdout,format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            
        logging.info("start")

        writer=NAKIGetData(config, args.docFolder, args.saveMetaDataTo, args.metadataFilesMap, args.hierarchyFile, args.saveDataTo)
        writer.write(args.saveDataTo is None, args.enableEmptyFulltext)
        logging.info("end")
            
    
    
    
    
    