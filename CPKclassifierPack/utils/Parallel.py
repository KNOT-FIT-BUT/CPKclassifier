# -*- coding: UTF-8 -*-
"""
Obsahuje nástroje pro paralelní zpracování.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

from multiprocessing import Process, Manager, Lock, Value, Condition
from ctypes import c_ulong
from enum import Enum
import pickle
import random
import string


from datetime import datetime

import struct


class EnhancedProcess(Process):
    
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        
        self._id=None
    
    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, newId):
        self._id=newId
        


class SharedStorage(object):
    """
    Uložiště synchronizované mezi procesy.
    """
    
    
    class StorageType(Enum):
        LIST = "list"   #SharedStorageList - wrapped manager.list()
        DICT = "DICT"   #SharedStorageDict - wrapped manager.dict()
        DICT_SIMPLE = "DICT_SIMPLE"   #manager.dict() Pokud nepotřebujeme pracovat s velkým objemem dat (nad 2GB), tak je vhodnější.
    
        
    
    def __init__(self, storageType, manager=None):
        """
        Inicializace uložiště.
        :type storageType: StorageType
        :param storageType: Druh uložiště. Všechny podporované druhy vymezuje StorageType.
        :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
        """
        if manager is None:
            manager=Manager()
        # Type checking
        if not isinstance(storageType, self.StorageType):
            raise TypeError('storageType musí být instancí StorageType')
        
        #Zde budou ukládány data,
        if storageType==self.StorageType.LIST:
            self._storage=SharedList(manager)
        elif storageType==self.StorageType.DICT:
            self._storage=SharedDict(manager)
        elif storageType==self.StorageType.DICT_SIMPLE:
            self._storage=manager.dict()
        else:
            raise ValueError('Neznámý druh uložiště (storageType).')
    
        self.__usedManager=manager
        #Sdílený zámek pro synchronizaci procesů
        self.__sharedLock=Lock()
        
        
        
        #počet uložených klasifikátorů
        self._numOfData=Value(c_ulong, 0)
        
        self.__waitForChange=Condition()
        
        self.acquiredStorage=False
        
    def __len__(self):
        """
        Zjištení počtu uložených dat.
        
        :return: Počet všech uložených dat.
        :rtype: int
        """
        return self._numOfData.value
    
    def _notifyChange(self):
        """
        Oznámí, že došlo ke změně připadným čekajícím.
        """
        self.__waitForChange.acquire()
        self.__waitForChange.notify_all()
        self.__waitForChange.release()
        
    def waitForChange(self, timeout=None):
        """
        Uspí proces dokud nenastane změna. Pokud měl proces přivlastněné uložiště, tak je uvolní
        a po probuzení zase přivlastní.
        
        :param timeout: Maximální počet sekund, které bude čekat. Může být None, pak čeká dokud nepřijde událost.
        """

        wasAcquiredBeforeSleeping=False
        
        if self.acquiredStorage:
            self.release()
            wasAcquiredBeforeSleeping=True
            
        self.__waitForChange.acquire()
        
        self.__waitForChange.wait(timeout)
        
        self.__waitForChange.release()
        
        if wasAcquiredBeforeSleeping:
            self.acquire()
            
        
    def acquire(self):
        """
        Přivlastní si uložiště pro sebe. Ostatní procesy musí čekat.
        """
        self.__sharedLock.acquire()
        self.acquiredStorage=True
        
    def release(self):
        """
        Uvolní uložiště pro ostatní procesy.
        """
        self.__sharedLock.release()
        self.acquiredStorage=False
    

    def _safeAcquire(self):
        """
        Přivlastnění si uložiště. V momentu, kdy chci měnit jeho stav.
        Zohledňuje případ, kdy je uložiště zamluveno pomocí acquire.
        """
        if not self.acquiredStorage:
            self.__sharedLock.acquire()
            
            
    def _safeRelease(self):
        """
        Uvolnění přístupu k uložišti.
        Zohledňuje případ, kdy je uložiště zamluveno pomocí acquire.
        """
        if not self.acquiredStorage:
            self.__sharedLock.release()
    

class SharedObject(object):
    MAX_SIZE_OF_SHARED_OBJECT=2**(struct.calcsize("!i")*8-2)-1 #definuje maximalni velikost v bajtech, sdileneho objektu, na základě omezení v manageru
    
    def _makeParts(self, item, maxSizeOfSharedObject=MAX_SIZE_OF_SHARED_OBJECT):
        """
        Rozložení položky na části.
        
        :param item: Položka
        :param maxSizeOfSharedObject: Položka bude serializována a rozdělena na položky o maximální velikosti maxSizeOfSharedObject.
        :rtype: list
        :return: Item divided into parts.
        """
        #provedeme serializaci a rozkouskujeme
        serializedItem=pickle.dumps(item)
        return [serializedItem[x:x+maxSizeOfSharedObject] for x in range(0, len(serializedItem), maxSizeOfSharedObject)]
    
class SharedObjectStartOfDataBlock(object):
    pass

class SharedObjectEndOfDataBlock(object):
    pass
        
class SharedList(SharedObject):
    """
    Sdílený list mezi procesy. Nejdná se ovšem o standardní list, tedy může se chovat odlišně.
        Používá manager.list.
            Narozdíl od manager.list přidává:
                Podporu pro ukládání velkých objektů.
    """
    
    def __init__(self, manager=None):
        """
        Vytvoří nový sdílený list.
        
        :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
        """
        if manager is None:
            manager=Manager()
            
        self.manager=manager
        
        self._list=self.manager.list()
        
        self._len=Value(c_ulong, 0)

        #Sdílený zámek pro synchronizaci procesů
        self.__sharedLock=Lock()

    def __repr__(self):
        try:
            self.__sharedLock.acquire()
            return repr(self._list)
        finally:
            self.__sharedLock.release()

    def __len__(self):
        try:
            self.__sharedLock.acquire()
            return self._len.value
        finally:
            self.__sharedLock.release()

    def __getitem__(self, ii):
        """
        Získání položky na daném indexu.
        
        :param ii: Index položky
        """
        #najdeme realný index
        i=self._searchIndex(ii)
        if isinstance(self._list[i], SharedObjectStartOfDataBlock):
            #jedná se o položku složenou z více částí.
            restored=b""
            i+=1
            while not isinstance(self._list[i], SharedObjectEndOfDataBlock):
                restored+=self._list[i]
                i+=1
            return pickle.loads(restored)
        else:
            return self._list[i]

    def __delitem__(self, ii):
        """
        Maže položku na daném umístění.
        
        :param ii: Umístění.
        """
        
        self.__sharedLock.acquire()
        
        try:
            #najdeme realný index
            i=self._searchIndex(ii)
            if isinstance(self._list[i], SharedObjectStartOfDataBlock):
                #jedná se o položku složenou z více částí.
                
                #mažeme po zarážku
                self._delUntilEndOfBlock(i)
                
            del self._list[i]
            self._len.value-=1
        finally:
            self.__sharedLock.release()

    def __setitem__(self, ii, item, maxSizeOfSharedObject=SharedObject.MAX_SIZE_OF_SHARED_OBJECT):
        """
        Přepsání položky na dané pozici
        
        :param ii: pozice
        :param item: Položka
        :param maxSizeOfSharedObject: Pokud dojde k nepříjmutí položky Managerem. 
        Bude položka serializována a rozdělena na položky o maximální velikosti maxSizeOfSharedObject.
        :raise struct.error: Nelze odeslat položku do Manageru and po rozdělení.
        """
        self.__sharedLock.acquire()
        
        #najdeme realný index
        i=self._searchIndex(ii)
        if isinstance(self._list[i], SharedObjectStartOfDataBlock):
            #mažeme do konce bloku
            self._delUntilEndOfBlock(i)
                
        try:
            #pokusíme se přepsat      
            self._list[i]=item
            
        except struct.error:
            #Pravděpodobně jsou data příliš velká.
            #Rozkouskujeme je a zkusíme štěstí znovu.
            parts=self._makeParts(item, maxSizeOfSharedObject)
            #uložíme do společného uložiště
            #Na začátek a konec vložíme zarážky
            
            #1.přepisujeme
            self._list[i]=SharedObjectStartOfDataBlock
            #další přidáváme
            for part in parts:
                i+=1
                self._list.insert(i, part)
            i+=1
            self._list.insert(i, SharedObjectEndOfDataBlock())
        finally:
            self.__sharedLock.release()

    def __str__(self):
        return str(self._list)

    def insert(self, ii, item, maxSizeOfSharedObject=SharedObject.MAX_SIZE_OF_SHARED_OBJECT):
        """
        Vložení položky na dané místo.
        
        :param ii: Kam chci vložit položku.
        :param item: Položka
        :param maxSizeOfSharedObject: Pokud dojde k nepříjmutí položky Managerem. 
        Bude položka serializována a rozdělena na položky o maximální velikosti maxSizeOfSharedObject.
        :raise struct.error: Nelze odeslat položku do Manageru and po rozdělení.
        """
        

        self.__sharedLock.acquire()
        
        #najdeme realný index
        i=self._searchIndex(ii)
        added=False
        #pokusíme vložit
        try:
            self._list.insert(i, item)
            added=True
        except struct.error:
            #Pravděpodobně jsou data příliš velká.
            #Rozkouskujeme je a zkusíme štěstí znovu.
            parts=self._makeParts(item, maxSizeOfSharedObject)
            #uložíme do společného uložiště
            #Na začátek a konec vložíme zarážky

            self._list.insert(i, SharedObjectStartOfDataBlock)
            for part in parts:
                i+=1
                self._list.insert(i, part)
            i+=1
            self._list.insert(i, SharedObjectEndOfDataBlock())
            added=True
        finally:
            if added:
                self._len.value+=1
            self.__sharedLock.release()

    def append(self, item, maxSizeOfSharedObject=SharedObject.MAX_SIZE_OF_SHARED_OBJECT):
        """
        Přidá položku na konec seznamu.
        
         
        :param item: Položka
        :param maxSizeOfSharedObject: Pokud dojde k nepříjmutí položky Managerem. 
        Bude položka serializována a rozdělena na položky o maximální velikosti maxSizeOfSharedObject.
        :raise struct.error: Nelze odeslat položku do Manageru and po rozdělení.
        """
        self.__sharedLock.acquire()
        
        added=False
        try:
            self._list.append(item)
            added=True
        except struct.error:
            #Pravděpodobně jsou data příliš velká.
            #Rozkouskujeme je a zkusíme štěstí znovu.
            
            parts=self._makeParts(item, maxSizeOfSharedObject)
            
            #uložíme do společného uložiště
            #Na začátek a konec vložíme zarážky

            self._list.append(SharedObjectStartOfDataBlock())
            for part in parts:
                self._list.append(part)
            self._list.append(SharedObjectEndOfDataBlock())
            added=True
        finally:
            if added:
                self._len.value+=1
            self.__sharedLock.release()
        
    def clear(self):
        """
        Odstranění všeh položek.
        """
        
        self.__sharedLock.acquire()
        try:
            del self._list[:]
        finally:
            self.__sharedLock.release()
    
    def _searchIndex(self, ii):
        """
        Searches real index for virtual index ii.
        
        :param ii: Virtual index.
        :return: Real index to list.
        """
        
        searchIndex=-1
        seekEndOfPart=False

        for i,x in enumerate(self._list):
            if isinstance(x, SharedObjectStartOfDataBlock):
                seekEndOfPart=True
                searchIndex+=1
            elif isinstance(x, SharedObjectEndOfDataBlock):
                seekEndOfPart=False
            elif not seekEndOfPart:
                searchIndex+=1

            if searchIndex==ii:
                break
              
        
        if searchIndex!=ii:
            i+=1
            
        return i
    
    def _delUntilEndOfBlock(self, i):
        """
        Maže od indexu i (včetně) po zarážku konce bloku.
        Samotný konec bloku není smazán!!!
        
        :param i: Od jakého indexu začne.
        """
        while not isinstance(self._list[i], SharedObjectEndOfDataBlock):
            del self._list[i]


class SharedObjectSplitedIntoBlocks(object):
    """
    Používá se k indikaci, že daný objekt byl rozdělen na bloky.
    Uchovává také klíče k těmto blokům.
    """
    def __init__(self, keys):        
        self.keys = keys
 
class SharedDict(SharedObject):
    """
    Sdílený dict mezi procesy. Nejdná se ovšem o standardní dict, tedy může se chovat odlišně.
    Používá 2x manager.dict.
        Narozdíl od manager.dict přidává:
            Podporu pro ukládání velkých objektů.
    """
    
    def __init__(self, manager=None):
        """
        Vytvoří nový sdílený dict.
        
        :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
        """
        if manager is None:
            manager=Manager()
            
        self.manager=manager
        
        self._dict=self.manager.dict()
        
        #uchování objektů, které musely být rozloženy na části
        self._blocks=self.manager.dict()


        #Sdílený zámek pro synchronizaci procesů
        self.__sharedLock=Lock()

    def __setitem__(self, key, item, maxSizeOfSharedObject=SharedObject.MAX_SIZE_OF_SHARED_OBJECT):
        """
        Nastavení položky s daným klíčem.
        
        :param key: Klíč do dict.
        :param item: Položka.
        :param maxSizeOfSharedObject: Pokud dojde k nepříjmutí položky Managerem. 
        Bude položka serializována a rozdělena na položky o maximální velikosti maxSizeOfSharedObject.
        :raise struct.error: Nelze odeslat položku do Manageru and po rozdělení.
        """
        self.__sharedLock.acquire()
        try:
            if key in self._dict and isinstance(self._dict[key], SharedObjectSplitedIntoBlocks):
                #odstraníme bloky původního objektu
                for tmpKey in self._dict[key].keys():
                    del self._keyMap[tmpKey]
                    
            #pokusíme se vložit
            self._dict[key]=item
            
        except struct.error:
            #Pravděpodobně jsou data příliš velká.
            #Rozkouskujeme je a zkusíme štěstí znovu.
            
            parts=self._makeParts(item, maxSizeOfSharedObject)
            
            #uložíme do společného uložiště
            blockKeys=[]
            
            for part in parts:
                k=self._generateKey()
                self._blocks[k]=part
                blockKeys.append(k)
                
            self._dict[key]=SharedObjectSplitedIntoBlocks(blockKeys)
            
        finally:
            self.__sharedLock.release()
    
    def __getitem__(self, key):
        """
        Získání položky na daném klíči.
        
        :param key: Daný klíč.
        """
        self.__sharedLock.acquire()
        try:
            item=self._dict[key]
            if isinstance(item, SharedObjectSplitedIntoBlocks):
                #máme položku rozloženou na více částí

                item=self._getBlockItem(item.keys())
                
            return item
        finally:
            self.__sharedLock.release()
        

    def __repr__(self):
        self.__sharedLock.acquire()
        try:
            return repr(self._dict)+repr(self._blocks)
        finally:
            self.__sharedLock.release()

    def __len__(self):
        return len(self._dict)

    def __delitem__(self, key):
        """
        Odstraní položku na klíči.
        
        :param key: Daný klíč.
        """
        self.__sharedLock.acquire()
        try:
            if isinstance(self._dict[key], SharedObjectSplitedIntoBlocks):
                #položka je rozdělena do bloků, které musíme nejprve odstranit
                for pKey in self._dict[key].keys():
                    del self._blocks[pKey]
                    
            del self._dict[key]
            
        finally:
            self.__sharedLock.release()

    def clear(self):
        """
        Odstranění všeh položek.
        """
        
        self.__sharedLock.acquire()
        try:
            self._dict.clear()
            self._blocks.clear()
        finally:
            self.__sharedLock.release()

    def copy(self):
        """
        Vytvoření
        """
        return self.__dict__.copy()

    def has_key(self, k):
        """
        Zjištění zda obsahuje daný klíč.
        
        :param k: Klíč
        :rtype: bool
        :return: True obsahuje. Flase jinak
        """
        
        self.__sharedLock.acquire()
        try:
            return k in self._dict
        finally:
            self.__sharedLock.release()
        

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        """
        Získání klíčů
        
        :return: klíče
        """
        return self._dict.keys()

    def values(self):
        """
        Získání hodnot.
        
        :rtype: list
        :return: hodnoty
        """
        values=[]
        self.__sharedLock.acquire()
        try:
            for x in self._dict.values():
                if isinstance(x, SharedObjectSplitedIntoBlocks):
                    #položka rozdělená do částí
                    values.append(self._getBlockItem(x.keys()))
                else:
                    values.append(x)
            
            return values
        finally:
            self.__sharedLock.release()

    def items(self):
        """
        Získání klíčů a hodnot.
        
        :rtype: list of pairs
        :return: Klíče a hodnoty. (key, value)
        """
        
        items=[]
        self.__sharedLock.acquire()
        try:
            for k, x in self._dict.values():
                if isinstance(x, SharedObjectSplitedIntoBlocks):
                    #položka rozdělená do částí
                    items.append((k, self._getBlockItem(x.keys())))
                else:
                    items.append((k, x))
            return items
        finally:
            self.__sharedLock.release()

    def pop(self, key, default=None):
        """
        Pokud je klíč v dict odstraní jej a vrátí jeho hodnotu, jinak vrátí default.
        Pokud default není předán a klíč není v dict, tak vyhodí KeyError
        
        :param key:
        :param default:
        :return: Hodnotu klíče, nebo default.
        """
        self.__sharedLock.acquire()
        try:
            if key not in self._dict:
                #klíč neexistuje
                if default is not None:
                    return default
                else:
                    raise KeyError(key)
                
            #klíč existuje
            
            val=self._dict[key]
            
            if isinstance(val, SharedObjectSplitedIntoBlocks):
                parts=b""
                for k in val.keys():
                    parts+=self._blocks[k]
                    del self._blocks[k]
                    
                val=pickle.loads(parts)
                    
            del self._dict[key]
            
            return val
        finally:
            self.__sharedLock.release()
        

    def __contains__(self, key):
        """
        Zjištění zda dict obsahuje klíč. (Používá has_key) 
        
        :param key: Klíč
        :rtype: bool
        :return: True obsahuje. Flase jinak
        """
        return self.has_key(key)

    def __iter__(self):
        """
        Iteruje přes všechny klíče.
        Neblokuje proti změnám od ostatních procesů!
        """
        for k in self._dict:
            yield k

    
    def _generateKey(self, controlWith=None):

        """
        Generuje náhodný řetězec použitelný jako klíč.
        
        :type controlWith: dict 
        :param controlWith: Volitelný parametr. Pokud je zadán, tak kontroluje výskyt vygenerovaného klíč
            v dict a snaží se získat klíč, který se v dic nevyskytuje.
        :return: Náhodný klíč.
        """
        k=str(random.random())+str(datetime.now())
        while k in controlWith:
            #pokud došlo k náhodnému vzniku stejných klíčů přidáme na konec klíče znak
            k+=random.choice(string.ascii_uppercase + string.digits)
                    
        return k
    
    def _getBlockItem(self, keys):
        """
        Vytvoří z jednotlivých částí položky jednu celkovou položku.
        
        :type keys: list
        :param keys: Klíče jednotlivých částí.
        :return: Složenou položku.
        """

        return pickle.loads(b"".join(self._blocks[k] for k in keys))



class SharedQueue(SharedDict):
    """
    Sdílená fronta využívající SharedDict.
    """
    
    def __init__(self, manager=None):
        """
        Vytvoří novou sdílenou frontu.
        
        :param manager: Volitelný parametr. Pokud chceme vnutit použití jiného multiprocessing.Manager.
        """
        super.__init__(manager)
        
        self.__waitForChange=Condition()
        
    def put(self, data):
        pass
    
    def qsize(self):
        """
        Aktuální délka fronty.
        """
        return len(self)
    
    def get(self, ):
        pass
        
    
    

    