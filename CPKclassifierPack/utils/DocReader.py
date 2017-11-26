# -*- coding: UTF-8 -*-
"""
Obsahuje třídu pro čtení dokumentů z datového souboru a souboru metadat.

@author:     Martin Dočekal
@contact:    xdocek09@stud.fit.vubtr.cz
"""

from random import shuffle
import copy
import csv
import logging
import re

class DocReaderInvalidMetadataFields(Exception):
    """
    Některá pole nejsou ve vstupním souboru s metadaty.
    """
    pass

class DocReaderInvalidDataFileForMetadata(Exception):
    """
    Nevalidní datový soubor pro daná metadata.
    """
    pass

class DocReaderNeedDataFile(Exception):
    """
    Chybí soubor s plnými texty.
    """
    pass

class DocReader(object):
    """
    Třída pro čtení dokumentů ze souboru ve formátu: jeden dokument na řádek a k němu korespondující 
    řádek v metadatovém souboru ve formátu csv.
    Dokumenty mohou být čteny v řádkovém pořadí nebo náhodném.
    Umožňuje také filtrovat dokumenty na základě polí v metadatech.
    """
    def __init__(self, dataFile, metadataFile, lazyEvalData=False, nonEmpty=None, empty=None, fieldRegex=None, minPerField=None, maxPerField=None,
                  initMetadata=None, initLinesOffsets=None, itemDelimiter=None, selectWords=None, selectItems=None, fulltextName="fulltext"):
        """
        Konstruktor DocReader.
        
        :param dataFile: Cesta k datovému souboru s dokumenty (jeden dokument na řádek).
        :param metadataFile: Cesta k souboru s metadaty v csv formátu.
        :param lazyEvalData: Použít líné vyhodnocení. Namísto stringu bude vrácen DocReaderDataString s daty.
        :param nonEmpty: list -- polí z metadat, které nesmí být prázdné. 
        :param empty: list -- polí z metadat, které musí být prázdné. 
        :param fieldRegex: dict -- ve formátu: klíč jako jméno pole a hodnota je dvojice (regulární výraz, flagy).
            Položka v poli bude vynechána pokud regulární výraz neodpovídá.
        :param minPerField: dict -- ve formátu: klíč jako jméno pole a hodnota s minimálním počtem dokumentů
            Filtr definuje minimální počet dokumentů na položku v poli.
            Pro příklad: Pokud pole obsahuje kategorii a kategorie má méně než minimální počet dokumentů, celá kategorie bude vynechána.
        :param maxPerField: dict -- ve formátu: klíč jako jméno pole a hodnota s maximálním počtem dokumentů
            Filtr definuje maximální počet dokumentů na položku v poli.
            Pokud je hodnota float v intervalu <0,1>, maximálně x procent (int((numberOfDcuments*x)+0.5)) dokumentů v poli bude přečteno.
        :param initMetadata: Inicializuje metadata aniž by četl soubor s metadaty. Pomocí předaných hodnot v tomto parametru.
            Když je tento parametr použitý: nonEmpty, minPerField a maxPerField jsou ignorovány.
        :param initLinesOffsets: Inicializuje offsety datového souboru, aniž by jej četl. Pomocí předaných hodnot v tomto parametru.
        :param itemDelimiter: Oddělovač (řetězec), který separuje položky v poli.
        :param selectWords: slice/integer -- pro vybrání slov v každém dokumentu
        :param selectItems: dict -- klíč je název pole a hodnota je slice/integer pro vybrání položek.
        """
        
        self.fieldNames=None
        
        self.lazyEvalData=lazyEvalData
        self.nonEmpty=nonEmpty
        self.empty=empty
        self.fieldRegex=fieldRegex
        self.minPerField=minPerField

        self.maxPerField=maxPerField
        self.itemDelimiter=itemDelimiter
        
        self.selectWords=selectWords
        self.selectItems=selectItems
        
        self.__fulltextName=fulltextName
            
        #kvuli filtru je nutne prvne analyzovat plne texty
        self.dataFile = dataFile
        self.dataLinesOffsets=initLinesOffsets
        if initLinesOffsets is None and self.dataFile is not None:
            self.__dataFileLinesOffsets()
    
        self.metadataFile = metadataFile
        self.metadata=initMetadata
        if initMetadata is None:
            self.__filter()
        
        
        

        
    def __len__(self):
        """
        Vrací velikost.
        """
        return len(self.metadata)
        
    def __iter__(self):
        """
        Iteruje zkrze dokumenty.
        """
        for docInfoData in self.read():
            yield docInfoData
            
    def read(self, stringFormat=False):
        """
        Přečte další dokument a jeho metadata.
        
        :returns: list -- (data, metadata)
        """
        with open(self.dataFile, "r") as fileToRead:
            for docIndex, mData in self.metadata:
                line=None
                if self.lazyEvalData:
                    line=DocReaderDataString(self.dataFile, self.dataLinesOffsets[docIndex], None, None, self.selectWords)
                else:
                    fileToRead.seek(self.dataLinesOffsets[docIndex])
                    line=fileToRead.readline().rstrip('\n').split()
                    if self.selectWords:
                        line=line[self.selectWords]

                        
                yield (line, mData.copy())
        
    def toData(self):
        """
        Vytvoří DocReaderData pro čtení dat bez metadat.
        
        :returns:  DocReaderData -- Obalí tento DocReader.
        """
        return DocReaderData(
            dataFile=self.dataFile, 
            metadataFile=self.metadataFile, 
            lazyEvalData=self.lazyEvalData, 
            nonEmpty=self.nonEmpty, 
            empty=self.empty, 
            fieldRegex=self.fieldRegex, 
            minPerField=self.minPerField, 
            maxPerField=self.maxPerField,
            initMetadata=copy.deepcopy(self.metadata), 
            initLinesOffsets=copy.deepcopy(self.dataLinesOffsets), 
            itemDelimiter=self.itemDelimiter, 
            selectWords=self.selectWords, 
            selectItems=self.selectItems,
            fulltextName=self.__fulltextName)
            
    def toMetaData(self):
        """
        Vytvoří DocReaderMetadata pro čtení metadat bez dat.
        
        :returns:  DocReaderMetadata -- Obalí tento DocReader.
        """
        return DocReaderMetadata(
            metadataFile=self.metadataFile, 
            nonEmpty=self.nonEmpty, 
            empty=self.empty, 
            fieldRegex=self.fieldRegex, 
            minPerField=self.minPerField, 
            maxPerField=self.maxPerField,
            initMetadata=copy.deepcopy(self.metadata),
            itemDelimiter=self.itemDelimiter,
            selectItems=self.selectItems)
    
    def toList(self):
        """
        Převede dokumenty do listu.
        
        :returns:  list -- naplněný dokumenty (data,metadata).
        """
        return [doc for doc in self.read()]
        
    def __dataFileLinesOffsets(self):
        """
        Hledá offsety řádků
        
        :returns:  list -- řádkových offsetů.
        """
        logging.info("začátek hledání offsetů řádků v datovém souboru")
        if self.dataLinesOffsets is None:
            self.dataLinesOffsets=[]
            with open(self.dataFile, "r") as f:
                self.dataLinesOffsets.append(0)
                while f.readline():
                    self.dataLinesOffsets.append(f.tell())
                    
                del self.dataLinesOffsets[len(self.dataLinesOffsets)-1]
                
        logging.info("konec hledání offsetů řádků v datovém souboru")
        return self.dataLinesOffsets
    
    def __maxFilter(self):
        """
        Filtrování dokumentů na základě parametru maxPerField.
        """
        if not self.maxPerField:
            return
        
        #vytvoření statistik
        fieldsStats=self.fieldsStats(self.maxPerField.keys())
        restToRead={}

        for fName, maxV in self.maxPerField.items():
            if fName not in restToRead:
                restToRead[fName]={}
            
            for item in fieldsStats[fName]:
                restToRead[fName][item]=fieldsStats[fName][item]
                
                if isinstance(maxV, float) and maxV<=1:
                    restToRead[fName][item]=int((fieldsStats[fName][item]*maxV)+0.5)
                elif restToRead[fName][item]>maxV:
                    restToRead[fName][item]=maxV
                    
        
        #spuštění max filtru
        newMeta=[]
        for i, mData in self.metadata:
            skipDoc=False
            rowA=dict([ (x, None) for x in mData])
            
            for fName, fVal in mData.items():
                if not fVal:
                    #prázdné pole
                    continue
                #filtrování položek
                filteredItems=[]
                maxControl=(restToRead and (fName in self.maxPerField))
                for ite in fVal:
                    if maxControl:
                        if restToRead[fName][ite]<=0:
                            #přeskoč max
                            continue
                        #nepřeskočené dokumenty budou přečteny => sniž hodnotu
                        restToRead[fName][ite]=restToRead[fName][ite]-1
                        
                    filteredItems.append(ite)
                    
                if len(filteredItems)==0:
                    skipDoc=True
                    break;
                
                rowA[fName]=filteredItems
                
            if skipDoc:
                continue
                  
            newMeta.append((i, rowA))
            
        self.metadata=newMeta
        
    def __minFilter(self):
        """
        Filtrování dokumentů na základě parametru minPerField.
        """
        #začítek min filteru
        if not self.minPerField:
            return
        fieldsStats=self.fieldsStats(self.minPerField.keys())
        newMeta=[]
        
        for i, mData in self.metadata:
            skipDoc=False
            rowA=dict([ (x, None) for x in mData])
            
            for fName, fVal in mData.items():
                if not fVal:
                    #prázdné pole
                    continue
                
                #filtrování položek
                filteredItems=[]
                
                minControl=(self.minPerField and (fName in self.minPerField))
                for ite in fVal:
                    if minControl and (fieldsStats[fName][ite] < self.minPerField[fName]):
                        #přeskoč min
                        continue
                    filteredItems.append(ite)
                    
                if len(filteredItems)==0:
                    skipDoc=True
                    break;
                
                rowA[fName]=filteredItems
                
            if skipDoc:
                continue
                  
            newMeta.append((i, rowA))
            
        self.metadata=newMeta     
        
    def __nonEmptyRegexRowFilter(self, row, rowIndex, openedDataFile=None):
        """
        Filtr dokumentů na základě non empty filtru a regexRowFilter.
        
        :param row: Řádek metadat.
        :param rowIndex: Udává kolikáty je toto řádek.
        :param openedDataFile: Volitelný argument. Pokud chceme kontroloval prázdnost/plnost souboru s plnými texty.
        :returns: dict -- konvertovaný dokument | None => dokument by měl být vynechán
        """
        
        
        #kontrola pro plný text
        if openedDataFile is not None:
            openedDataFile.seek(self.dataLinesOffsets[rowIndex])
            words=openedDataFile.readline().rstrip('\n').split()
                
            if self.__fulltextName in self.nonEmpty and len(words)==0:
                #musí mít neprázdný plný text
                return None 
                
            if self.__fulltextName in self.empty and len(words)>0:
                #musí mít prázdný plný text
                return None 
        doc=dict([ (x, None) for x in row])
        for fName, fVal in row.items():
            fItems=self.__getFieldItems(fVal, fName)
            
            if self.fieldRegex and fName in self.fieldRegex:
                reg, fl =self.fieldRegex[fName]
                newItems=[]
                for ite in fItems:
                    
                    if re.match(reg, ite, fl):
                        newItems.append(ite)
                        
                fItems=newItems

            if self.nonEmpty and fName in self.nonEmpty and not fItems:
                #prázdné pole, dokument by měl být vynechán
                return None
            
            if self.empty and fName in self.empty and fItems:
                #Pole by mělo být prázdné, dokument by měl být vynechán
                return None
            
            doc[fName]=fItems

        
        return doc
        
        
        
    def __filter(self):
        """
        Filtr pro požadované dokumenty.
        
        :raise DocReaderInvalidDataFileForMetadata: 
        """
        
        logging.info("začátek čtení metadat a filtrování dokumentů")
        
        self.metadata=[]
        with open(self.metadataFile, "r") as metadata:
            reader = csv.DictReader(metadata)
            self.fieldNames=reader.fieldnames
            self.__validateFields()
            #prvně čti a použij non empty a regex filter 
            lineCnt=0
            openedDataFile=None
            if self.dataFile is not None and (self.__fulltextName in self.nonEmpty or  self.__fulltextName in self.empty):
                #budeme potřebovat pro filtr otevřit soubor
                openedDataFile=open(self.dataFile, "r")
            
            for row in reader:

                if self.dataLinesOffsets is not None and lineCnt>=len(self.dataLinesOffsets):
                    raise DocReaderInvalidDataFileForMetadata()
                
                rowA=self.__nonEmptyRegexRowFilter(row, lineCnt, openedDataFile)
                if rowA:
                    self.metadata.append((lineCnt, rowA))
                
                lineCnt=lineCnt+1
            
            if openedDataFile:
                openedDataFile.close()
            
            self.__maxFilter()  #prvně čti maximální počet dokumentů
            self.__minFilter()  #pak odsekni zbytek, který nedosáhne na minimální hodnoty
                
        logging.info("konec čtení metadat a filtrování dokumentů")
        
    def __validateFields(self):
        """
        Zjistí jestli daný soubor obshauje potřebná pole.
        
        :raises DocReaderInvalidMetadataFields:
        """
        needFields=set()
        
        if self.nonEmpty:
            needFields.update(set(self.nonEmpty)-set([self.__fulltextName]))
            if (self.__fulltextName is not None and self.__fulltextName in self.nonEmpty) and not self.dataFile:
                raise DocReaderNeedDataFile();
        
        if self.empty:
            needFields.update(set(self.empty)-set([self.__fulltextName]))
            
            if (self.__fulltextName is not None and self.__fulltextName in self.empty) and not self.dataFile:
                raise DocReaderNeedDataFile();
            
        if self.fieldRegex:
            needFields.update(self.fieldRegex)
            
        if self.minPerField:
            needFields.update(self.minPerField)
            
        if self.maxPerField:
            needFields.update(self.maxPerField)
            
        if self.selectItems:
            needFields.update(self.selectItems)
        
        if not needFields.issubset(set(self.fieldNames)):
            raise DocReaderInvalidMetadataFields()
            
        
    def fieldsStatsNonFiltered(self, fields):
        """
        Počítá dokumenty odpovídající položkám v polích. (bez filtrů)
        
        :param fields: list -- jmén polí
        :returns: dict -- se statistikami
        """

        with open(self.metadataFile, "r") as metadata:
            reader = csv.DictReader(metadata)
            
            fieldsCnt=dict([(f, {}) for f in fields])
            for row in reader:
                for f in fields:
                    for ite in self.__getFieldItems(row[f], f):
                        if ite not in fieldsCnt[f]:
                            fieldsCnt[f][ite]=0
                        fieldsCnt[f][ite]=fieldsCnt[f][ite]+1
                    
        return fieldsCnt
    
    def fieldsStats(self, fields):
        """
        Počítá dokumenty korespondující k položkám.
        
        :param fields: list -- jmén polí
        :returns: dict -- se statistikami
        """
        fieldsCnt=dict([(f, {}) for f in fields])
        
        for mData in self.metadata:
            for f in fields:
                if mData[1][f] is not None:
                    for ite in mData[1][f]:
                        if ite not in fieldsCnt[f]:
                            fieldsCnt[f][ite]=0
                        fieldsCnt[f][ite]=fieldsCnt[f][ite]+1
                    
        return fieldsCnt
    
    def __getFieldItems(self, field, name):
        """
        Získání položek z pole.
        
        :param field: Pole obsahující položky.
        :param name: Jméno pole.
        :returns: list/None -- s položkami
        """
        if not field:
            return []
        
        if self.itemDelimiter is not None:
            items=[ x.strip() for x in field.split(self.itemDelimiter)]
            if self.selectItems and name in self.selectItems:
                items=items[self.selectItems[name]]
                if type(items) is not list:
                    return [items]
                return items

            return items
            
        return [field]
    
    def shuffle(self):
        """
        Zamíchá dokumenty.
        """
        logging.info("začátek míchání dokumentů")
        shuffle(self.metadata)
        logging.info("konec míchání dokumentů")

class DocReaderData(DocReader):
    """
    Třída pro čtení dat dokumentů ze souboru.
    """  
        
    def __iter__(self):
        """
        Iteruje v datech dokumentů.
        """
        for doc in super().read():
            yield doc[0]
            
    def toList(self):
        """
        Konvertuje data do listu.

        :returns:  list -- Naplněný daty dokumentů.
        """
        return [doc[0] for doc in self.read()]
    
class DocReaderMetadata(DocReader):
    """
    Třída pro čtení metadat dokumentů ze souboru.
    """  
    def __init__(self, metadataFile, nonEmpty=None, empty=None, fieldRegex=None, minPerField=None, maxPerField=None,
                  initMetadata=None, itemDelimiter=None, selectItems=None):
        """
        Konstruktor DocReaderMetadata.
        
        :param metadataFile: Cesta k souboru s metadaty v csv formátu.
        :param nonEmpty: list -- polí z metadat, které nesmí být prázdné
        :param empty: list -- polí z metadat, které musí být prázdné. Parametr nonEmpty má prioritu před empty.
        :param fieldRegex: dict -- ve formátu: klíč jako jméno pole a hodnota je dvojice (regulární výraz, flagy).
            Položka v poli bude vynechána pokud regulární výraz neodpovídá.
        :param minPerField: dict -- ve formátu: klíč jako jméno pole a hodnota s minimálním počtem dokumentů
            Filtr definuje minimální počet dokumentů na položku v poli.
            Pro příklad: Pokud pole obsahuje kategorii a kategorie má méně než minimální počet dokumentů, celá kategorie bude vynechána.
        :param maxPerField: dict -- ve formátu: klíč jako jméno pole a hodnota s maximálním počtem dokumentů
            Filtr definuje maximální počet dokumentů na položku v poli.
            Pokud je hodnota float v intervalu <0,1>, maximálně x procent (int((numberOfDcuments*x)+0.5)) dokumentů v poli bude přečteno.
        :param initMetadata: Inicializuje metadata aniž by četl soubor s metadaty. Pomocí předaných hodnot v tomto parametru.
            Když je tento parametr použitý: nonEmpty, minPerField a maxPerField jsou ignorovány.
        :param itemDelimiter: Oddělovač (řetězec), který separuje položky v poli.
        :param selectItems dict -- klíč je název pole a hodnota je slice/integer pro vybrání položek.
        """
        
        super().__init__(None, metadataFile, nonEmpty=nonEmpty, empty=empty, fieldRegex=fieldRegex, minPerField=minPerField, maxPerField=maxPerField,
                  initMetadata=initMetadata, itemDelimiter=itemDelimiter,  selectItems=selectItems)
        
    def __iter__(self):
        """
        Iteruje v metadatech dokumentů.
        """
        for doc in self.read():
            yield doc   
            
    def read(self):
        """
        Čte další metadata dokumentu.
        
        :returns: list -- metadata
        """
        for _, mData in self.metadata:
            yield mData.copy()
            
    def toData(self):
        """
        S tímto nelze číst data.
        
        :returns:  None.
        """
        return None
        
    def toList(self):
        """
        Konvertuje metadata do listu.
        
        :returns:  list -- naplněný metadaty.
        """
        return [doc for doc in self.read()]

class DocReaderDataString(object):
    """
    Třída pro líné vyhodnocení. Je vhodná pro úsporu paměti.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Konstruktor.
        
        :param filename: string -- cesta k souboru pro čtení
        :param lineOffset: list -- offset řádku, který by se měl číst
        :param offsetOnLine: list -- offset na řádku, který by měl být přečtený
        :param readLength: list -- maximální délka řetězce
        :param selectWords: slice -- pro výběr slov v dokumentu
        """
        self.filename=args[0]
        self.lineOffset = args[1]
        
        self.offsetOnLine=args[2] if len(args)>2 else None
        self.readLength=args[3] if len(args)>3 else None
        
        self.selectWords=args[4] if len(args)>4 else None
            
        self.len=None
        
        
    def __len__(self):
        """
        Délka listu.
        
        :returns:  int -- Délka listu.
        """
        if self.len is None:
            self.len=len(self.__readMyWords())
        return self.len
    
    def __readMyLinePart(self):
        """
        Čte definovanou část z definovaného řádku.
        
        :returns: definovanou část řádku
        """
        with open(self.filename, "r") as fileToRead:
            fileToRead.seek(self.lineOffset)
            line=fileToRead.readline().rstrip('\n')
            
            if self.offsetOnLine and self.readLength:
                line=line[self.offsetOnLine:(self.offsetOnLine+self.readLength)]
            elif self.offsetOnLine:
                line=line[self.offsetOnLine:]
            elif self.readLength:
                line=line[:self.readLength]
                
            return line
        
    def __readMyWords(self):
        """
        Přečtení jenom vybraných slov.
        
        :returns: vybraná slova
        """
        if self.selectWords:
            return self.__readMyLinePart().split()[self.selectWords]
        return self.__readMyLinePart().split()
            
        
    def __getitem__(self, ind):
        """
        Získej položku na daném indexu.
        
        :param ind: index položky
        :returns: položka na indexu
        """
        
        return self.__readMyWords()[ind]
    
    def __iter__(self):
        """
        Iteruje přes list.
        """
        for item in self.__readMyWords():
            yield item
        
    def __str__(self):
        """
        Konverze tohoto listu na string.
        """
        
        return " ".join(self.__readMyWords())

    def __repr__(self):
        """
        Reprezentace tohoto listu.
        """
        return '%s(filename=%s, lineOffset=%s, offsetOnLine=%s, readLength=%s)' % (self.__class__.__name__, self.filename, self.lineOffset, self.offsetOnLine, self.readLength)    
    
    def makeChunks(self, maxWords):
        """
        Vyrobí části s deinovaným maximálním počtem slov.
        
        :param maxWords: Maximum slov.
        :returns: List s DocReaderDataString s částmi řádku
        """
        
        line=self.__readMyLinePart()
        words=[ (s.start(), len(s.group())) for s in re.finditer(r'\S+', line)]
        #chunking
        chunks=[]
        for x in range(0, len(words), maxWords):
            lastWord=words[x: x + maxWords][-1]
            chunks.append(DocReaderDataString(self.filename, self.lineOffset, words[x][0], lastWord[0]+lastWord[1]-words[x][0]))

        return chunks
    
    