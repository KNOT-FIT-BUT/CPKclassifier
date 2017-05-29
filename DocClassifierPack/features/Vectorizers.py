# -*- coding: UTF-8 -*-
"""
Obsahuje třídu sloužící jako wrapper pro Doc2Vec na použití s sklearn.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""

import numpy as np
from sklearn.base import BaseEstimator 

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

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
        

                    
    def fit_transform(self, X, y):
        """
        Trénuje Doc2Vec a vrací vektory dat.
        
        :param X: Data
        :param y: Cíle
        :returns: numpy array -- Převedená vstupní data na vektory.
        """
        
        if self.analyzer:
            d2vDocsTrain=[TaggedDocument(D2VApplyAnalyzer(doc, self.analyzer), [i]) for i, doc in  enumerate(X)]
        else:
            d2vDocsTrain=[TaggedDocument(doc, [i]) for i, doc in  enumerate(X)]
        self.modelD2v.build_vocab(d2vDocsTrain)
        self.modelD2v.train(d2vDocsTrain)
        
        vect=np.array(self.modelD2v.docvecs)
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
            
        self.modelD2v.build_vocab(d2vDocsTrain)
        self.modelD2v.train(d2vDocsTrain)
        self.modelD2v.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)
        
        return self

    def transform(self, X):
        """
        Převede vstupní data na vektory. Dle modelu.
        
        :param X: Vstupní data.
        :returns: numpy array -- Převedená data na vektory.
        """
        if self.analyzer:
            tmp=np.array([self.modelD2v.infer_vector(D2VApplyAnalyzer(doc, self.analyzer)) for doc in X])
        else:
            tmp=np.array([self.modelD2v.infer_vector(doc) for doc in X])

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

        self.__len=None

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
        if self.__len is None:
            self.__len=len(self.__analyze())

        return self.__len

        
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
    
    

    