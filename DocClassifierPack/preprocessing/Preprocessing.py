# -*- coding: UTF-8 -*-
"""
Obsahuje třídu pro předzpracování vstupních dat.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

import logging
import re
from unidecode import unidecode

from ufal.morphodita import *


class Preprocessing(object):
    """
    Třída pro předzpracování vstupních dat.
    """

    posSigns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Z", "X"]
    def __init__(self, args, stopWords, tagger, taggerPOS, logAfterLines=100):
        """
        Inicializace předzpracování
        
        :param args:    Argumenty pro preprocessing z ArgumentsManager pro DocClassifier.
        :param stopWords: list -- obsahující stop slova
        :param tagger: Cesta k souboru pro tagger morphodity.
        :param taggerPOS: Cesta k souboru pro tagger morphodity. Který bude použit pro extrakci slovních druhů.
        :param logAfterLines: Udává po kolika přečtených řádcích má dojit k logování.
        """
        
        self.args=args
        self.stopWords=stopWords
        self.tagger=tagger
        self.taggerPOS=taggerPOS
        self.logAfterLines=logAfterLines
        
        
    def setParams(self, args=None, stopWords=None, tagger=None, taggerPOS=None, logAfterLines=None):
        """
        Nastavení nových argumentů, které slouží jako parametry pro předzpracování.
        
        :param args:    Argumenty pro preprocessing z ArgumentsManager pro DocClassifier.
        :param stopWords: list -- obsahující stop slova
        :param tagger: Cesta k souboru pro tagger morphodity.
        :param taggerPOS: Cesta k souboru pro tagger morphodity. Který bude použit pro extrakci slovních druhů.
        :param logAfterLines: Udává po kolika přečtených řádcích má dojit k logování.
        """
        if args is not None:
            self.args=args
        if stopWords is not None:
            self.stopWords=stopWords
        if tagger is not None:
            self.tagger=tagger
        if taggerPOS is not None:
            self.taggerPOS=taggerPOS
        if logAfterLines is not None:
            self.logAfterLines=logAfterLines
        
    def start(self):
        """
        Zahájí předzpracování. Parametry se nastavují v konstruktoru nebo pomocí setParams.
        Výsledek je zapisován do stdout.
        """
        logging.info("začátek předzpracování")
        
        with open(self.args.input, "r") as inFile:
            lemPosExt=None
            onlyPos=None
            
            
            #inicializace nástrojů pro předzpracování
            wordsToRem=[]
            if self.args.noSW:
                wordsToRem=self.stopWords
                    
            WordsRemover=RemoveWords(wordsToRem, self.args.minWordLength, self.args.maxWordLength)
            
            if self.args.lemmatize or self.args.pos or self.args.sepSigns:
                lemPosExt=Lemmatizer(self.tagger, self.taggerPOS)
                if self.args.pos:
                    onlyPos=lemPosExt.translateNumericPOS(self.args.pos)
            
            lineCnt=0
            
            
            for line in inFile:
                line=line.rstrip('\n')
                lineCnt+=1
                
                if lineCnt%self.logAfterLines==0:
                    logging.info("Předzpracovávám "+str(lineCnt)+". řádek.")
                                    
                if self.args.pos:
                    #extrakce slovních druhů
                    line=lemPosExt.extract(line, onlyPos, self.args.lemmatize)
                    
                elif self.args.lemmatize:
                    #lemmatizace
                    line=lemPosExt.lemmatize(line)
                        
                elif self.args.sepSigns:
                    #oddělení znaků od slov např. ,.:;
                    #lemmatizace a extrakce příznaků již znaky separuje
                    #neprovádíme tedy separaci znavu, když není potřeba
                    
                    
                    line=[ x[0] for x in lemPosExt.getWordsPOS(line)]
                
                if isinstance(line, list):
                    line=" ".join(line)
                    
                #Odstranění nežádaných slov
                line=" ".join(WordsRemover.removeWords(line))   
                
                if self.args.unidecode:
                    line=unidecode(line)
                    
                if self.args.uc:
                    line=line.upper()
                if self.args.lc:
                    line=line.lower()
                
                print(line)
                
            
            logging.info("Předzpracováno "+str(lineCnt)+" řádků.") 
                
        logging.info("konec předzpracování")
                
class RemoveWords(object):
    """
    Třída pro odstranění slov s definovnými parametry. Stop slova nebo na základě počtu znaků.
    """

    def __init__(self, removeStopWords=[], minWordLength=None, maxWordLength=None):
        """
        Konstrukce objektu.
        
        :param removeStopWords: list -- stop slov pro odstranění
        :param minWordLength: Minimální délka slova. Každé slovo s délkou menší než je tato bude odstraněno.
        :param maxWordLength: Maximální délka slova. Každé slovo s délkou větší než je tato bude odstraněno.
        """
        self.removeStopWords=removeStopWords
        self.minWordLength=minWordLength
        self.maxWordLength=maxWordLength
        
        
    
    def removeWords(self, txt):
        """
        Odstraní slova s definovanými vlastnostmi.
        
        :param txt: string/list -- pro odstranění slov.
        :returns:  list -- obsahující zbývající slova
        """
        
        if isinstance(txt, list):
            words=txt
        else:
            words=txt.split()
            
        if self.removeStopWords==[] and self.minWordLength is None and self.maxWordLength is None:
            return words
        
        notRemovedWords=[]
        for x in words:
            if (self.removeStopWords and x in self.removeStopWords) or \
                (self.minWordLength and len(x) <self.minWordLength) or\
                (self.maxWordLength and len(x) >self.maxWordLength):
                continue
                    
            notRemovedWords.append(x)
                
        return notRemovedWords                

class LemmatizerTaggerException(Exception):
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
    
    def __init__(self, tagger, taggerPOS):
        """
        Konstrukce objektu.
        
        :param tagger: Cesta k souboru pro tagger morphodity.
        :param taggerPOS: Cesta k souboru pro tagger morphodity. Který bude použit pro extrakci slovních druhů.
        :raises LemmatizerTaggerException: Když není definovaný tokenizer pro dodaný model.
        """
        self.tagger=Tagger.load(tagger)
        if self.tagger is None:
            raise LemmatizerTaggerException("Chybný TAGGER.")
        self.tokenizer = self.tagger.newTokenizer()
        if self.tokenizer is None:
            raise LemmatizerTaggerException("Není definovaný tokenizer pro dodaný model.")
        
        self.taggerPOS=Tagger.load(tagger)
        if self.taggerPOS is None:
            raise LemmatizerTaggerException("Chybný TAGGER.")
        self.tokenizerPOS = self.tagger.newTokenizer()
        if self.tokenizerPOS is None:
            raise LemmatizerTaggerException("Není definovaný tokenizer pro dodaný model.")
        
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
            self.tagger.tag(self.forms, self.lemmas)
            for i in range(len(self.lemmas)):
                lemma = self.lemmas[i]
                self.converter.convert(lemma)
                words.append(lemma.lemma)
            
        return words
    
    def getWordsPOS(self, text):
        """
        Získá slovní druhy k jednotlivým slovům v parametru text.
        
        :param text: Text pro analýzu.
        :returns:  list -- of tuples, kde (slovo, POS)
        """
        
        self.tokenizerPOS.setText(text)
        words=[]
        while self.tokenizerPOS.nextSentence(self.forms, self.tokens):
            self.taggerPOS.tag(self.forms, self.lemmas)
            for i in range(len(self.lemmas)):
                lemma = self.lemmas[i]
                token = self.tokens[i]
                self.converter.convert(lemma)
                POS=lemma.tag.split("|")[0].split("=")[1]

                words.append((text[token.start : token.start + token.length], POS))
                    
        return words
        
        
    def extract(self, text, only, lemmatized=True):
        """
        Extrahuje pouze slova, která mají definovaný slovní druh v parametru only.
        
        :param text: Text pro extrakci.
        :param only: list -- povolených slovních druhů
        :param lemamatized: True -> vrátí lemmatizovanou formu slova.
        :returns:  list -- obsahující slova
        """

        if lemmatized:
            text=" ".join(self.lemmatize(text))
             
        self.tokenizerPOS.setText(text)
        words=[]
        while self.tokenizerPOS.nextSentence(self.forms, self.tokens):
            self.taggerPOS.tag(self.forms, self.lemmas)
            for i in range(len(self.lemmas)):
                lemma = self.lemmas[i]
                token = self.tokens[i]
                self.converter.convert(lemma)
                POS=lemma.tag.split("|")[0].split("=")[1]
                if POS in only:
                    words.append(text[token.start : token.start + token.length])
                    
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