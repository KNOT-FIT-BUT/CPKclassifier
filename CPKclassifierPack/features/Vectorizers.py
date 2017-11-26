# -*- coding: UTF-8 -*-
"""
Obsahuje třídy pro vektorizaci.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator 

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class MatchTargetVectorizer(BaseEstimator):
    """
    Vytvoří příznakový vektor, kde jednotlivé dimenze odrážejí jednotlivé cíle/třídy/y. Hodnoty v těchto dimenzích pak
    ukazují kolik shodných slov (k tomu danému cíli) je ve vzorku obsaženo.
    Příklad:
        Popis dimenzí:
            0.    Cíl obsahuje slova: A C D
            1.    Cíl obsahuje slova: B G F
        
        Vzorek:
            A A A B B A C
        
        Výsledný vektor:
            [5 2]

    """
    
    def __init__(self, analyzer=None, lemmatizer=None):
        """
        Vytvoří vektorizer.
        
        :param analyzer: Analyzátor dat. None => ngram/1
        :type lemmatizer: Lemmatizer
        :param lemmatizer: Lemmatizuje cíle před natrénováním.
        """
        
        self.analyzer=analyzer
        self.lemmatizer=lemmatizer
        
        self.classes_=None
        
        #Slouží k rychlému zjištění, které slovo patří ke kterým cílům z self.classes_
        #Obsahuje slova jako klíč a k ním indexi cílů. Indexy jsou získány z self.classes_.
        self.__vocabulary={}
        
    def __makeVocabulary(self, x):
        """
        Vyrobí slovník z dat x. Slovník slouží k rychlému 
        zjištění, které slovo patří ke kterým cílům z self.classes_
        
        :param x: List listů slov. Pořadí musí odpovídat pořadím z self.classes_.
        """
        
        for words, target in zip(x, range(len(self.classes_))):
            if self.lemmatizer is not None:
                words=self.lemmatizer.lemmatize(words)
                
                
            for word in words:
                word=word.upper()
                if word not in self.__vocabulary:
                    self.__vocabulary[word]=set()
                    
                self.__vocabulary[word].add(target)
    
    def fit_transform(self, X, y):
        """
        Trénuje vectorizer a vrací vektory dat.
        
        :param X: Data
        :param y: Cíle. Slouží pro vytvoření slovníku.
        :returns: Převedené vektory.
        """
        
        self.fit(X, y)

        return self.transform(X)
    
    def fit(self, X, y):
        """
        Trénuje vectorizer.
        
        :param X: Data
        :param y: Cíle. Slouží pro vytvoření slovníku.
        :returns: Sebe sama.
        """
        
        self.classes_=np.unique(y)
        
        
        self.__makeVocabulary(self.classes_)
        
        return self

    def transform(self, X):
        """
        Získává vektory dat.
        
        :param X: Vstupní data.
        :returns: Vstupní data. Popřípadě převedené ngramy.
        """

        
        if self.analyzer:
            docsIter=[self.analyzer(cX) for cX in  X]
        else:
            docsIter=X
        
        resVecs=[]
        
        for words in docsIter:
            
            matches=[0]*len(self.classes_)  #pro každou kategorii.
            for word in words:
                word=word.upper()
                #Procházíme slova a hledáme shody.
                if word in self.__vocabulary:
                    #Došlo ke shodě, pro některou (či více) z kategorií.
                    for tar in self.__vocabulary[word]:
                        #Projdeme všechny kategorie, kde došlo ke shodě.
                        matches[tar]+=1
                        
            resVecs.append(matches)

        return csr_matrix(resVecs)
    

class OmitVectorizer(BaseEstimator):
    """
    Slouží pro vynechání kroku vektorizace a ponechává data v původní podobě. Pouye použije daný analyzátor.
    """
    
    def __init__(self, analyzer=None):
        """
        
        :param analyzer: Analyzátor dat. None => ngram/1
        """
        
        self.analyzer=analyzer
        
    def fit_transform(self, X, y):
        """
        Pouze vrací vstupní data (může vytvářet ngramy, dle nastavení při konstrukci). 
        Slouží jen pro kompatibilitu.
        
        :param X: Data
        :param y: Cíle
        :returns: Vstupní data. Popřípadě převedené ngramy.
        """
        
        
        if self.analyzer:
            return [self.analyzer(cX) for cX in  X]
        
        return X
    
    def fit(self, X, y):
        """
        Pouze vrací sama sebe. Slouží jen pro kompatibilitu.
        
        :param X: Data
        :param y: Cíle
        :returns: Sebe sama.
        """
        
        return self

    def transform(self, X):
        """
        Pouze vrací vstupní data (může vytvářet ngramy, dle nastavení při konstrukci).
        Slouží jen pro kompatibilitu.
        
        :param X: Vstupní data.
        :returns: Vstupní data. Popřípadě převedené ngramy.
        """

        if self.analyzer:
            return [self.analyzer(cX) for cX in  X]
        
        return X
        

class D2VVectorizer(BaseEstimator):
    """
    Slouží jako wrapper pro Doc2Vec na použití s sklearn.
    """
    
    def __init__(self, size, alpha, window, minCount, workers, iterCnt, sample, dm, negative, analyzer=None):
        """
        Inicializace Doc2Vec.
        
        :param size: Počet dimenzí vektoru.
        :param alpha: Počáteční rychlost učení(learning rate) (bude lineárně klesat k nule).
        :param window: Maximální vzdálenost mezi predikovaným slovem a kontextovými slovy použita pro predikci v dokumentu.
        :param minCount: Ignoruje všechny slova s celkovou frekvencí menší než je tato.
        :param workers: Udává počet vláken pro trénování modelu.
        :param iterCnt: Počet iterací/epoch.
        :param sample: Prahová hodnota pro snížení vzorků slov s vysokou frekvencí.
        :param dm: Jaký má být použit algoritmus.
        :param negative: #if > 0, bude použito negativní vzorkování, celočíselná hodnota udává kolik “noise words” má být odstraněno (obvykle 5-20)
        :param analyzer: Analyzátor dat. None => ngram/1
        """

        self.modelD2v = Doc2Vec(size=size, alpha=alpha, window=window, min_count=minCount, workers=workers, iter=iterCnt, sample=sample, dm=dm, negative=negative)
        self.analyzer=analyzer
        
        self._buildVoc=True

    def buildVocab(self, X):
        """
        Vytvoří slovník z poskytnutých dat. A nastaví, aby se při fázi fit znovu nezískával.
        
        :param X: Data
        """
        if self.analyzer:
            d2vDocsTrain=[TaggedDocument(D2VApplyAnalyzer(doc, self.analyzer), [i]) for i, doc in  enumerate(X) if doc is not None]
        else:
            d2vDocsTrain=[TaggedDocument(doc, [i]) for i, doc in  enumerate(X) if doc is not None]
        
        self.modelD2v.build_vocab(d2vDocsTrain)
        
        self._buildVoc=False
        
    def buildVocabWhenFit(self):
        """
        Nastaví, aby se vytvořil slovník při fit fázi. Ve fázi fit se tvoři implicitně.
        Vhodné při použití například s buildVocab.
        """
        
        self._buildVoc=True

        
    def fit_transform(self, X, y):
        """
        Trénuje Doc2Vec a vrací vektory dat.
        
        :param X: Data
        :param y: Cíle
        :returns: sparse matrix -- Převedená vstupní data na vektory.
        """
        
        if self.analyzer:
            d2vDocsTrain=[TaggedDocument(D2VApplyAnalyzer(doc, self.analyzer), [i]) for i, doc in  enumerate(X)]
        else:
            d2vDocsTrain=[TaggedDocument(doc, [i]) for i, doc in  enumerate(X)]
            
        if self._buildVoc:
            self.modelD2v.build_vocab(d2vDocsTrain)
            
        self.modelD2v.train(d2vDocsTrain, total_examples=self.modelD2v.corpus_count, epochs=self.modelD2v.iter)
        
        vect=csr_matrix(self.modelD2v.docvecs)
        self.modelD2v.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)
        
        return vect
    
    def fit(self, X, y):
        """
        Trénuje Doc2Vec.
        
        :param X: Data
        :param y: Cíle
        :returns: Sebe sama.
        """
        if self.analyzer:
            d2vDocsTrain=[TaggedDocument(D2VApplyAnalyzer(doc, self.analyzer), [i]) for i, doc in  enumerate(X)]
        else:
            d2vDocsTrain=[TaggedDocument(doc, [i]) for i, doc in  enumerate(X)]
            
        if self._buildVoc:
            self.modelD2v.build_vocab(d2vDocsTrain)
            
        self.modelD2v.train(d2vDocsTrain, total_examples=self.modelD2v.corpus_count, epochs=self.modelD2v.iter)
        self.modelD2v.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)
        
        return self

    def transform(self, X):
        """
        Převede vstupní data na vektory. Dle modelu.
        
        :param X: Vstupní data.
        :returns: sparse matrix -- Převedená data na vektory.
        """
        if self.analyzer:
            tmp=csr_matrix([self.modelD2v.infer_vector(D2VApplyAnalyzer(doc, self.analyzer)) for doc in X])
        else:
            tmp=csr_matrix([self.modelD2v.infer_vector(doc) for doc in X])

        return tmp
    
class D2VApplyAnalyzer(list):
    """
    Třída pro aplikování analyzátoru na data. Tvoří ngramy.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Konstruktor.
        
        :param data: list -- se slovy
        :param analyzer: analyzátor, který se má použít
        """
        super().__init__()
        self.data=args[0]
        self.analyzer = args[1]

        self._numOfData=None

    def __analyze(self):
        """
        Použije analyzer na data.
        
        :returns: list -- analyzovaná data (ngramy)
        """
        return self.analyzer(self.data)
        
    def __len__(self):
        """
        Délka listu.
        
        :returns:  int -- Délka listu.
        """
        if self._numOfData is None:
            self._numOfData=len(self.__analyze())

        return self._numOfData

        
    def __getitem__(self, ind):
        """
        Získej položku na daném indexu.
        
        :param ind: index položky
        :returns: položka na indexu
        """
        return self.__analyze()[ind]
    
    def __iter__(self):
        """
        Iteruje přes list.
        """
        for item in self.__analyze():
            yield item
        
    def __str__(self):
        """
        Konverze tohoto listu na string.
        """
        return str(self.__analyze())

    def __repr__(self):
        """
        Reprezentace tohoto listu.
        """
        return repr(self.__analyze())
    
    

    