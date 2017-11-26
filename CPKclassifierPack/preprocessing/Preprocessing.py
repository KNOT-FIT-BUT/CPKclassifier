# -*- coding: UTF-8 -*-
"""
Obsahuje třídu pro předzpracování vstupních dat.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""
from multiprocessing import Process, Manager, Lock, Value, Queue, Condition
import queue

import multiprocessing
import logging
import traceback
import sys

from unidecode import unidecode

from ufal.morphodita import *
import struct

import datetime
import ctypes


class SharedDocumentCache(object):
    """
    Slouží jako sdílená cache dokumentů mezi procesy.
    Zabraňuje vypisování dokumentů, které nejsou v pořadí.
    Do cache není nutné ukládat prázdné dokumenty.
    
    Pokud například řeknu pomocí metody waitFor, že čekám na dokument s číslem 20 a 40.
    Tedy:
    waitFor(20)    #Tato metoda ukládá čísla dokumentů do listu
    waitFor(40)
    
    Tak v momentě, kdy se objeví v cache dokument s číslem 20, jsou vypsány i prázdné dokumenty
    před dokumentem 20 (počáteční dokument má číslo 0). Dále pokud se objeví v cache dokument 40. Vypíší se
    i prázdné dokumenty mezi, tedy 21-39.
    """
    
    #CONDITION_WAIT_TIME_OUT=1.0   #maximální počet sekund, při čekání na podmínku
    
    def __init__(self, manager=None):

        """
        Inicializace cache.
        
        :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
        """
        if manager is None:
            manager=Manager()
        self.__docCache=manager.list()  #Zde budeme ukladat dokumenty.
        self.__sharedLock=Lock()
        self.__shouldWrite=Value(ctypes.c_ulonglong, 0)
        self.__numOfWaitingForWrite=Value(ctypes.c_uint, 0) #počet čekajicích procesů na výpis
        
        
    
        #List neprázdných dokumentů, na které čekáme. 
        #Díky tomuto listu nemusíme do cache ukládat prázdné dokumenty.

        self.__waitQ=manager.list()
        
        self.__notifyNewActWait=Condition(self.__sharedLock)
        
    def gF(self):
        return self.__docCache[0][0] if len(self.__docCache) else None
        
    def getNumberOfDocumentsInCache(self):
        """
        Získání počtu dokumentu v cache.
        
        :return: Počet dokumentů v cache.
        """
        
        return len(self.__docCache)
    
    def getNumberOfWaitingForWrite(self):
        """
        Získání počtu procesů, které čekají na zápis.
        
        :return: Počet procesů čekajících na zápis.
        """
        
        return self.__numOfWaitingForWrite.value

    def getLastWritten(self):
        """
        Získání posledně vypsaného dokumentu.
        
        :return: Vrací číslo naposledy vypsaného dokumentu. -1 znamená, že doposud nebyl vypsán dokument.
        """
        
        return self.__shouldWrite.value-1
        
    def waitFor(self, docNumber):
        """
        Uložení čísla dokumentu, který chceme vypsat. Ukládáme takto dokumenty, abychom nemuseli cachovat i prázdné řádky.
        Vkládání se provádí do fronty.
        
        :param docNumber: Číslo dokumentu, na který čekáme pro vypsání.
        """
        self.__sharedLock.acquire()

        insertAt=0
        for x in self.__waitQ:
            if docNumber > x:
                insertAt+=1
            else:
                break

        self.__waitQ.insert(insertAt,docNumber)
        
        if len(self.__waitQ)==1:
            self.__changeActWaitFor(docNumber)

        self.__sharedLock.release()
        
        
    def actWaitFor(self):
        """
        Vrátí číslo dokumentu, na který aktuálně čeká.
        
        :return: Číslo dokumentu, na který aktuálně čekáme. Pokud nečekáme na žádný dokument vrací None.
        """

        if len(self.__waitQ)!=0:
            return self.__waitQ[0]
        
        return None
        
    def __changeActWaitFor(self, docNumber):
        """
        Změna dokumentu, na který aktuálně čekáme.
        
        Dává vědět případným čekajícím dokumentům, které se mají zapsat bez ukládání do cache,
        že došlo ke změně aktuálního dokumentu, na který čekáme.
        
        :param docNumber: Číslo dokumentu.
        """
        #dame vedet pripadnym cekajicim dokumentum, ktere se maji zapsat
        #bez ukladani do cache
        self.__notifyNewActWait.notify_all()
        
    def __nextWaitFor(self):
        """
        Nastavíme dalšího ve frontě jako aktuálního, na který čekáme.
        Starého z fronty odstraníme.
        
        Dává vědět případným čekajícím dokumentům, které se mají zapsat bez ukládání do cache,
        Že došlo ke změně aktuálního dokumentu, na který čekáme.
        
        Používat pokud vlastním zámek.
        
        """

        if len(self.__waitQ)!=0:
            
            del self.__waitQ[0]
            #změníme aktuální
            if len(self.__waitQ)>0:
                self.__changeActWaitFor(self.__waitQ[0])
        
        
        
    def cacheDoc(self, docNumber, docTxt, pEnd=''):
        """
        Uložení dokumentu do cache.
        
        :param docNumber: Číslo aktuálně nového dokumentu (číslo řádku).
        :param docTxt: Nový dokument pro uložení.
        :param pEnd: Jedná se o parametr end pro print. Tedy jak bude zakončen výpis řetězce.
        """

        self.__sharedLock.acquire()
        insertAt=0
        for x in self.__docCache:
            if docNumber > x[0]:
                insertAt+=1
            else:
                break

        self.__docCache.insert(insertAt,(docNumber, docTxt, pEnd))

        self.__sharedLock.release()
            
    def cacheWrite(self):
        """
        Vypisuje data z cache. Na stdout.
        Vypíše jen dokumenty, které neporuší pořadí dané číslem dokumentu.
        """

        self.__sharedLock.acquire()

        try:

            actWait=self.actWaitFor()

            if actWait is not None:
                    
                while len(self.__docCache) and actWait==self.__docCache[0][0]:
                    if self.__shouldWrite.value==actWait:

                        print(self.__docCache[0][1], end=self.__docCache[0][2], flush=True)
                        self.__shouldWrite.value+=1
                        
                        del self.__docCache[0]
                        

                        self.__nextWaitFor()
                        actWait=self.actWaitFor()
    
                    else:
                        #doplníme bílé řádky
                        print("\n"*(actWait-self.__shouldWrite.value), end="",flush=True)
                        self.__shouldWrite.value=actWait
        
        finally:
            self.__sharedLock.release()
            
        
    def writeWithoutCaching(self, docNumber, docTxt, pEnd=''):
        """
        Vypíše dokument ve správném pořadí, ale nebude ukládán do cache.
        Místo toho bude čekat do doby než na něj přijde řada.
        
        Je nutné si uvědomit, že v tomto případě při čekání máme obsazený jeden proces/vlákno,
        nemůžeme mezitím tedy konat užitečnou práci, pouze spíme.
        
        :param docNumber: Číslo dokumentu.
        :param docTxt: Obsah dokumentu.
        :param pEnd: Jedná se o parametr end pro print. Tedy jak bude zakončen výpis řetězce.
        """

        self.__sharedLock.acquire()

        self.__numOfWaitingForWrite.value+=1
        
        try:
            while True:
                
                actWait=self.actWaitFor()
                
                
                if actWait is not None and actWait==docNumber:
                    if self.__shouldWrite.value<actWait:
                        #doplníme bílé řádky
                        
                        print("\n"*(actWait-self.__shouldWrite.value), end=pEnd,flush=True)
                        self.__shouldWrite.value=actWait
                    
                    print(docTxt, end=pEnd,flush=True)
                    
                    self.__numOfWaitingForWrite.value-=1
                    self.__shouldWrite.value+=1
                    
                    self.__nextWaitFor()

                    break
                
                else:
                    #aktuálně nečekáme na předzpracování našeho dokumentu
                    #počkáme tedy
                    self.__notifyNewActWait.wait()
                    
        finally:
            self.__sharedLock.release()
            

        

class PreprocessingWorker(Process):
    """
    Třída reprezentující jeden pracující proces provádějící předzpracování.
    """
    
    def __init__(self, linesForPreprocessing, fileToRead, sharedCache, args, wordsRemover, lemPosExt, onlyPos, errorBoard):
        """
        Inicializace procesu.
        
        :param linesForPreprocessing: Queue obsahující řádky pro předzpracování.
        :param fileToRead: Cesta k souboru, ze kterého získáme řádky pro čtení. Používá se pokud je předán pouze offset.
        :type sharedCache: SharedDocumentCache
        :param sharedCache: Sdílená cache pro ukládání řádku/dokumentů, které nejsou v pořadí pro výpis.
        :param args: Argumenty pro preprocessing z ArgumentsManager pro CPKclassifier.
        :param wordsRemover: RemoveWords -- Používá se pro odstraňování nevhodných slov.
        :param lemPosExt: Lemmatizer -- Inicializovaný lemmatizer, který bude použit pro lemmatizaci a výběr slov 
            na základě slovního druhu.
        :param onlyPos: list|None -- obsahující slovní druhy pro extrakci
        :param errorBoard: Queue, kde se v případě chyby dá vědět rodičí.
        """
        super(PreprocessingWorker, self).__init__()

        self.linesForPreprocessing = linesForPreprocessing
        self.fileToRead=fileToRead
        self.sharedCache=sharedCache
        self.args=args
        self.wordsRemover=wordsRemover
        self.lemPosExt=lemPosExt
        self.onlyPos=onlyPos
        self.errorBoard=errorBoard
        
        
    def preprocess(self, line):
        """
        Předzpracování jednoho řádku.
        
        :param line: Řádek pro předzpracování.
        :returns: string -- předzpracovaný řádek
        """
        
        if line == "":
            #optimalizace pro prázdné řádky
            return line
                            
        
        if not self.args.lemmatize:
            line=" ".join(self.wordsRemover.safeSepRemoveWords(line))
        
        if self.args.pos:
            #extrakce slovních druhů
            line=self.lemPosExt.extract(line, self.onlyPos, self.args.lemmatize)
            
        elif self.args.lemmatize:
            #lemmatizace
            line=self.lemPosExt.lemmatize(line)
                
        elif self.args.sepSigns:
            #oddělení znaků od slov např. ,.:;
            #lemmatizace a extrakce příznaků již znaky separuje
            #neprovádíme tedy separaci znavu, když není potřeba
            
            
            line=[ x[0] for x in self.lemPosExt.getWordsPOS(line)]
        
        
        if self.wordsRemover.couldRemove():
            #Odstranění nežádaných slov
            line=self.wordsRemover.removeWords(line)   
        
        if isinstance(line, list):
            line=" ".join(line)
        
        if self.args.unidecode:
            line=unidecode(line)
            
        if self.args.uc:
            line=line.upper()
        if self.args.lc:
            line=line.lower()
        
        return line
            
    def run(self, once=False):
        """
        Předzpracování.
        
        :param once: Pokud je True. Zpracuje jeden řádek, pokud je ihned k dispozici.
        """
        
        try:
            while True:
                try:
                    msg=self.linesForPreprocessing.get(timeout=1 if once else None)
                except queue.Empty:
                    break
                else:
                    if msg == "EOF":
                        return
                    
                    lineNumber, lineTxt, pEnd=msg

                    if type(lineTxt) == str:
                        lineTxt=self.preprocess(lineTxt)
                    else:
                        #dostali jsme pouze offset do souboru (zřejmě velký dokument)
                        #musime dokument tedy prvne nacist do pameti
                        with open(self.fileToRead, "r") as reading:
                            reading.seek(lineTxt)
                            lineTxt=self.preprocess(reading.readline().rstrip('\n'))
                            
                    if pEnd==" " and len(lineTxt)<1:
                        pEnd=""
                        
                    if sys.getsizeof(lineTxt)>Preprocessing.MAX_SIZE_OF_SHARED_OBJECT:
                        #příliš velké pro cache, počkáme na přímé vypsání
                        self.sharedCache.writeWithoutCaching(lineNumber, lineTxt, pEnd)
                    else:
                        #dáme do cache a místo čekání budeme dělat užitečnější práci
                        self.sharedCache.cacheDoc(lineNumber, lineTxt, pEnd)
                
                if once:
                    return
                    
        except:
            self.errorBoard.put("ERROR")
            print(traceback.format_exc(), file=sys.stderr)
            
        


class Preprocessing(object):
    """
    Třída pro předzpracování vstupních dat.
    """
    
    MAX_SIZE_OF_SHARED_OBJECT=2**(struct.calcsize("!i")*8-2)-1 #definuje maximalni velikost v bajtech, sdileneho objektu, na základě omezení v manageru


    posSigns=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Z", "X"]
    def __init__(self, args, stopWords, morphDic, workers=1, logAfterSec=30, maxNumberOfWordsPerLinePart=100000):
        """
        Inicializace předzpracování
        
        :param args:    Argumenty pro preprocessing z ArgumentsManager pro CPKclassifier.
        :param stopWords: list -- obsahující stop slova
        :param morphDic: Cesta k souboru pro slovník morphodity.
        :param workers: Počet pracujících procesů. Výchozí počet je 1 proces.
        :param logAfterSec: Udává po kolika sekundách nejdříve má dojit k logování.
        :param maxNumberOfWordsPerLinePart: Udává limitní počet slov, které budou zpracovávany společně.
                Rozděluje řádky na části.
        """
        
        self.args=args
        self.stopWords=stopWords
        self.dict=morphDic
        self.workers=workers
        if self.workers==-1:
            try:
                
                self.workers=multiprocessing.cpu_count()
            except:
                self.workers=1
        
        self.logAfterSec=logAfterSec
        self.errorBoard=None
        self.logType=0  #0 one process, 1 multiprocess
        self.lastLogTime=None
        
        self.maxNumberOfWordsPerLinePart=maxNumberOfWordsPerLinePart
        
    def start(self):
        """
        Zahájí předzpracování. Parametry se nastavují v konstruktoru nebo pomocí setParams.
        Výsledek je zapisován do stdout.
        """
        logging.info("začátek předzpracování")


        lemPosExt=None
        onlyPos=None
            
            
        #inicializace nástrojů pro předzpracování
        wordsToRem=set()
        if self.args.noSW:
            wordsToRem=set(self.stopWords)
                    
        WordsRemover=RemoveWords(wordsToRem, self.args.minWordLength, self.args.maxWordLength)

        if self.args.lemmatize or self.args.pos or self.args.sepSigns:
            lemPosExt=Lemmatizer(self.dict)
            if self.args.pos:
                onlyPos=set(lemPosExt.translateNumericPOS(self.args.pos))
                
                
        if self.workers==1:
            #jednoprocesorova varianta
            p=PreprocessingWorker(linesForPreprocessing=None, fileToRead=self.args.input,
                                  sharedCache=None, args=self.args, wordsRemover=WordsRemover, 
                                  lemPosExt=lemPosExt, onlyPos=onlyPos, errorBoard=None)
            
            self.lineNumber=0
            
            for lineCnt, isLastPart, lineTxt in self.readLineParts(self.args.input):
                
                if lineCnt!=self.lineNumber:
                    #další řádek
                    self.lineNumber+=1
                    self.logInfo()
                    self.lineNumber=lineCnt
                  
                pLine=p.preprocess(lineTxt)
                
                if isLastPart:
                    pEnd='\n'
                elif len(lineTxt)<1:
                    pEnd=''
                else:
                    pEnd=' '
                    
                
                
                if pEnd==" " and len(pLine)<1:
                    pEnd=""
                
                print(pLine, end=pEnd)

        else:
            #multiprocesorova varianta
            self.logType=1  #zmenime logovani na multiprocesove
            
            self.docCache=SharedDocumentCache()

            self.linesForPreprocessing=Queue()
            self.errorBoard=Queue()
            processes=[]
            
            
                
            #vytvoření procesů
            logging.info("Vytvářím nový počet procesů pro předzpracování: "+str(self.workers-1))
            
            for i in range(0,self.workers-1):
                p=PreprocessingWorker(linesForPreprocessing=self.linesForPreprocessing, fileToRead=self.args.input,
                                      sharedCache=self.docCache, args=self.args, wordsRemover=WordsRemover, 
                                      lemPosExt=lemPosExt, onlyPos=onlyPos, errorBoard=self.errorBoard)
                processes.append(p)
                logging.info("Proces "+str(i+1)+". start.")
                p.start()
                
            
            #Vytvoříme ještě jeden objekt pro případnou pomoc ostatním procesům se zpracováním.
            forHelpingP=PreprocessingWorker(linesForPreprocessing=self.linesForPreprocessing, fileToRead=self.args.input,
                                      sharedCache=self.docCache, args=self.args, wordsRemover=WordsRemover, 
                                      lemPosExt=lemPosExt, onlyPos=onlyPos, errorBoard=self.errorBoard)
            
            logging.info("procházím")
            
            #Budeme procházet plné texty, řádek po řádku.
            #Prázdné řádky jsou triviální záležitost a budeme je zpracovávat ve své vlastní režii.
            #Neprázdné budeme delegovat na ostatní procesy, pokud ale budeme mít dostatečný náskok,
            #tak pomůžeme ostatním s předzpracováním neprázdných řádků. Pokud bude řádek přiliš velký
            #pro meziprocesovou komunikaci, tak jej zpracujeme.
            
            self.lineNumber=0
            self.partsCounter=0
            
            for lineCnt, isLastPart, lineTxt in self.readLineParts(self.args.input):
                #procházíme všechny party
                
                self.controlMulPErrors()

                if lineTxt != "":
                    if isLastPart:
                        pEnd='\n'
                    elif len(lineTxt)<1:
                        pEnd=''
                    else:
                        pEnd=' '
                        
                    #prázdné řádky nedelegujeme, ale přímo zpracujeme.
                    if self.linesForPreprocessing.qsize()>(self.workers-1)*2:
                        logging.info("HELPING");
                        #Máme dostatečný náskok.
                        #Pomůžeme se zpracováním.                   
                            
                        self.docCache.waitFor(self.partsCounter)
                            
                        lineTxt=forHelpingP.preprocess(lineTxt)
                        
                        if pEnd==" " and len(lineTxt)<1:
                            pEnd=""
                            
                        if sys.getsizeof(lineTxt)>self.MAX_SIZE_OF_SHARED_OBJECT:
                            #přiliš velký pro cache
                            self.docCache.writeWithoutCaching(self.partsCounter, lineTxt, pEnd)
                        else:
                            self.docCache.cacheDoc(self.partsCounter, lineTxt, pEnd)
                                
                    elif sys.getsizeof(lineTxt)>self.MAX_SIZE_OF_SHARED_OBJECT:
                        logging.info("MUST HELPING");
                        #přiliš velký pro meziprocesorovou komunikaci
                        self.docCache.waitFor(self.partsCounter)
                            
                        lineTxt=forHelpingP.preprocess(lineTxt)
                        
                        if pEnd==" " and len(lineTxt)<1:
                            pEnd=""
                            
                        #přiliš velký pro cache
                        self.docCache.writeWithoutCaching(self.partsCounter, lineTxt, pEnd)
                            
                    else:
                        #delegujeme
                        self.docCache.waitFor(self.partsCounter)
                        self.linesForPreprocessing.put((self.partsCounter, lineTxt, pEnd))
                            
                    #vypíšeme data z cache
                    self.docCache.cacheWrite()
                        

                    
                if isLastPart:
                    #další řádek
                    self.lineNumber+=1
                    self.logInfo()
                    self.lineNumber=lineCnt

                #posuneme se na další part
                self.partsCounter+=1
            
                
            while self.linesForPreprocessing.qsize():
                #nemáme nic na práci, tak pomůžeme ostatním
                self.controlMulPErrors()
                forHelpingP.run(True)
                self.docCache.cacheWrite()
                self.logInfo()
                

            #vlozime priznak pro ukončení potomků
            for i in range(0,self.workers-1):
                self.linesForPreprocessing.put("EOF")
                
            #čekáme na ukončení
            for proc in processes:
                self.controlMulPErrors()
                self.logInfo()
                proc.join()           
                
            #vypíšeme z cache zbytek
            self.docCache.cacheWrite()
            
            if self.docCache.getLastWritten()!=self.partsCounter-1:
                #doplníme případné prázdné řádky na konci
                print("\n"*(self.partsCounter-1-self.docCache.getLastWritten()), end="",flush=True)
                
        logging.info("Předzpracováno "+str(self.lineNumber)+" řádků.") 
                
        logging.info("konec předzpracování")
        
    def logInfo(self):
        """
        Zobrazení logovacích informací.
        """
        if self.lastLogTime is None:
            self.lastLogTime=datetime.datetime.now()
            
        if (datetime.datetime.now()-self.lastLogTime).total_seconds()>=self.logAfterSec:
            if self.logType==0:
                self.lastLogTime=datetime.datetime.now()
                logging.info("Vypsáno "+str(self.lineNumber)+" řádků.")
            elif self.logType==1:
                self.lastLogTime=datetime.datetime.now()
                logging.info("Vypsáno "+str(self.docCache.getLastWritten()+1)+" řádků. Zpracovaných, ale nevypsaných částí: "+str(self.partsCounter-self.docCache.getLastWritten())+".")
                            
        
        
    def controlMulPErrors(self):
        """
        Kontrola chyb z ostatních procesů.
        """
        if self.errorBoard and not self.errorBoard.empty():
            print("Vznikla chyba. Ukončuji všechny procesy.", file=sys.stderr)
            for p in multiprocessing.active_children():
                p.terminate()
            exit()
            
            
    def readLineParts(self, filename, READ_SIZE=1000000):
        """
        Čte vstupní soubor a dělí jej po řádcích, pokud je vstupní řádek přiliš veliký
        je dále dělen dle počtu slov na jednotlivé části.
        
        :param filename: Cesta k souboru
        :type READ_SIZE: int 
        :param READ_SIZE: Volitelný. Kolik znaků bude maximálně naráz získáno ze souboru.
        :return: (lineCNT, isLastPartOnLine, part)
        """
        
        class AutomatonStates(object):
            START = 1
            WORD = 2
            SEARCH_NEXT_WORD=3

        lineCnt=0
      
        part=[""]
      
        state=AutomatonStates.START
        
        
        with open(filename, "r") as fInput:
            sr=fInput.read(READ_SIZE)

            while sr!='':
                i=0
                while i<len(sr):
                    c=sr[i]
                    i+=1
                    #čteme po znacích
                    if state==AutomatonStates.WORD:
                        #čteme slovo
                        
                        if c.isspace():
                            if c=="\n":
                                #konec slova vlivem konce řádku
                                state=AutomatonStates.START
                                yield(lineCnt, True, " ".join(part))
                                lineCnt+=1
                                part=[""]
                            else:
                                #konec slova vlivem bílého znaku
                                #zkusíme zjistit, zda-li za tímto slovem leží na tomto řádku i další slovo
                                state=AutomatonStates.SEARCH_NEXT_WORD
                        else:
                            #další znak slova
                            part[-1]+=c
                            
                    elif state==AutomatonStates.START:
                        #čekáme na slovo nebo konec řádku
                        if c=="\n":
                            #konec řádku
                            yield(lineCnt, True, " ".join(part))
                            lineCnt+=1
                            part=[""]
                        elif not c.isspace():
                            #začátek slova
                            state=AutomatonStates.WORD
                            part[-1]+=c

                    
                            
                    elif state==AutomatonStates.SEARCH_NEXT_WORD:
                        #skončili jsme se čtením slova a jsme zvědaví zda-li se na řádku nachází další
                        
                        if c.isspace():
                            if c=="\n":
                                #již žádné slovo na řádku není
                                state=AutomatonStates.START
                                yield(lineCnt, True, " ".join(part))
                                lineCnt+=1
                                part=[""]
                            else:
                                #vynecháme bílé znaky
                                pass
                        else:
                            state=AutomatonStates.WORD
                            #našli jsme slovo
                            if len(part)>=self.maxNumberOfWordsPerLinePart:
                                #máme již příliš mnoho slov
                                yield(lineCnt, False, " ".join(part))
                                #přidáme nové slovo
                                part=[c]
                            else:
                                #přidáme další nové slovo
                                part.append(c)
                
                #další data
                sr=fInput.read(READ_SIZE)
            
            #poslední part
            part=" ".join(part)
            if len(part)>0:
                yield(lineCnt, True, part)
        
            
        
                
class RemoveWords(object):
    """
    Třída pro odstranění slov s definovnými parametry. Stop slova nebo na základě počtu znaků.
    """

    def __init__(self, removeStopWords=set(), minWordLength=None, maxWordLength=None):
        """
        Konstrukce objektu.
        
        :param removeStopWords: set -- stop slov pro odstranění
        :param minWordLength: Minimální délka slova. Každé slovo s délkou menší než je tato bude odstraněno.
        :param maxWordLength: Maximální délka slova. Každé slovo s délkou větší než je tato bude odstraněno.
        """
        self.removeStopWords=removeStopWords
        self.minWordLength=minWordLength
        self.maxWordLength=maxWordLength
        
    

    def safeSepRemoveWords(self, txt):
        """
        Odstraní slova, která by byla odstraněna i po separaci znaků.
        Slouží pro optimalizaci předzpracování. Není zohledněna lemamtizaci.
        Používat pokud se nelemmatizuje.
        
        :param txt:string/list -- pro odstranění slov.
        :returns:  list -- obsahující zbývající slova
        """
        
        if isinstance(txt, list):
            words=txt
        else:
            words=txt.split()
            
        if self.removeStopWords==set() and self.minWordLength is None:
            return words
        
        notRemovedWords=[]
        

        for x in words:

            if (self.removeStopWords and x in self.removeStopWords) or \
                (self.minWordLength and len(x) <self.minWordLength):
                continue
                    
            notRemovedWords.append(x)
                
        return notRemovedWords

    
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
            
        if self.removeStopWords==set() and self.minWordLength is None and self.maxWordLength is None:
            return words
        
        notRemovedWords=[]
        for x in words:
            if (self.removeStopWords and x in self.removeStopWords) or \
                (self.minWordLength and len(x) <self.minWordLength) or\
                (self.maxWordLength and len(x) >self.maxWordLength):
                continue
                    
            notRemovedWords.append(x)
                
        return notRemovedWords          
    
    def couldRemove(self):
        """
        Zjištění, zda-li s aktuálním nastavením může dojít k nějakému odstranění slov
        :returns:  boolean -- True může dojit k odstranění. Jinak false.
        """
        
        return not(self.removeStopWords==set() and self.minWordLength is None and self.maxWordLength is None)

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
    
    def __init__(self, dictMorpho):
        """
        Konstrukce objektu.
        
        :param dictMorpho: Cesta k souboru pro dictMorpho morphodity.
        :raises LemmatizerException: Když není definovaný tokenizer pro dodaný model. Nebo nevalidní slovník.
        """

        self.morpho=Morpho.load(dictMorpho)
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