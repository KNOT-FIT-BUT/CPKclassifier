# -*- coding: UTF-8 -*-
"""
Tento modul obsahuje klasifikátory.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz

"""
import numpy as np
from sklearn.cluster import KMeans

from sklearn.neighbors import NearestNeighbors
from cProfile import label


class MatchTargetClassifier(object):
    """
    Klasifikuje na základě shody slov. Je vybrán cíl, který má nejvíce shodných slov v názvu s klasifikovaným dokumentem.
    Pracuje se slovy v běžné podobě. Tedy jako s řetězcem. Nikoliv například předzpracovanými do číselného vektoru.
    
    """
    
    def __init__(self, lemmatizer=None):
        """
        Konstruktor klasifikátoru.
        
        :type lemmatizer: Lemmatizer
        :param lemmatizer:Lemmatizuje cíle před natrénováním. 
            Pouze pro účely klasifikace. Původní tvar je uložen.
        """

        self.classes_=None
        self.lemmatizer=lemmatizer
        
        #Slouží k rychlému zjištění, které slovo patří ke kterým cílům z self.classes_
        #Obsahuje slova jako klíč a k ním indexi cílů. Indexy jsou získány z self.classes_.
        self.__vocabulary={}  
             
    
    def fit(self, X, y):
        """
        Natrénuje klasifikátor pomocí cílů trénovacích dat. Parametr X je zde
        pouze z důvodu kompatibility s ostatními klasifikátory z sklearn a není nijak použit.
        
        :param X: Trénovací data. (nepoužívá se)
        :param y: Cíle trénovacích dat. List obsahujicí jednotlivé cíle k trénovacím datům.
        """
        
        self.classes_=np.unique(y)

        
    def predict_proba(self, X):
        """
        Odhadne pravděpodobnost s jakou hodnotou patří daný exemplář do všech natrénovaných tříd.
        Získáváme tedy číslo v intervalu <0,1>, kde jedna je nejpravděpodobnější.
        Pořadí odhadů pravděpodobnosti je určeno pořadím v self.classes_.
        
        :param X: Data pro klasifikaci. List exemplářu, kde každý exemplář obsahuje list se slovy.
        :return: Vrací list, který obsahuje pro každý exemplář list, který obsahuje prvdepodobnost s
            jakou patří daný exemplář do každé z tříd z self.classes_. self.classes_ určuje pořadí těchto
            pravděpodobností.
        """
        
        predictedProba=[]
        for matches in X:
            if hasattr(matches, "todense"):
                matches=matches.todense()
                
            if hasattr(matches, "tolist"): 
                matches=matches.tolist()[0]

            matchSum=sum(matches)

            predictedProba.append([ tarMatches/matchSum if matchSum!=0 else 0 for tarMatches in matches])
            
        
        return predictedProba
    
    def predict(self, X):
        """
        Odhad cíle/třídy.
        
        :param X: Data pro klasifikaci. List exemplářu, kde každý exemplář obsahuje list se slovy.
        :return: Vrací list, který obsahuje pro každý exemplář název predikovaného cíle.
        """
        
        predictedLabels=[]
        
        for docProba in self.predict_proba(X):
            predictedLabels.append(self.classes_[docProba.index(max(docProba))])
        
        return predictedLabels
        
        
class KMeansClassifier(object):
    """
    Klasifikátor používající metodu k-means. Jedná se o učení s učitelem.
    Nejprve jsou pro každou kategorii/cíl vypočteny centroidy z trénovacích dat, které jsou použity jako počátační centroidy pro metodu k-means.
    K je odvozeno od počtu kategorií/cílů.
    """
    
    class LabelsClusters(object):
        """
        Slouží pro uložení clusterů, které jsou přiřazeny k dané kategorii/cíli.
        """
        
        def __init__(self, labelId, clusters):
            """
            Inicializace objektu.
            
            :param labelId: Název/identifikátor kategorie/cíle.
            :type clusters: dict 
            :param clusters: Klíč obsahuje identifikátor clusteru a hodnota udává počet přiřazení do tohoto clusteru.
            """
            
            self.__id=labelId
            self.__clusters=clusters
            self.__maxClusterRefresh()
            
            self.__nearestFreeClusters=None
            
        def __maxClusterRefresh(self):
            """
            Hledá id maximálního clusteru.
            """
            
            self.__maxClusterId=max(self.__clusters, key=self.__clusters.get)
            
        def addCluster(self, clusterId, value):
            """
            Přiřadí nový cluster. Pokud cluster s tímto id je již přiřazen, tak aktualizuje jeho hodnotu přiřazení.
            
            :param clusterId: Identifikátor clusteru.
            :param value: Počet přiřazení do tohoto clusteru.
            """
            self.__clusters[clusterId]=value
            self.__maxClusterRefresh()
            
        def addNearestFreeClusters(self, clusterIds):
            """
            Přiřazení nejbližšího volného clusteru.
            
            :type clusterIds: list 
            :param clusterIds: Id nejbližších volných clusterů.
            """
            
            self.__nearestFreeClusters=clusterIds
            
        def getNearestFreeClusters(self):
            """
            Získání id nejbližších volných clusterů.
            
            :return: Id nejbližšího volných clusterů. None => žádný nebyl přiřazen.
            """
            
            return self.__nearestFreeClusters
            
            
        def maxClusterId(self):
            """
            Získání id clusteru s nejvíce přiřazeními.
            
            :return: Id clusteru. None => nemá žádný cluster.
            """
            
            return self.__maxClusterId
        
        def maxClusterValue(self):
            """
            Získání počtu přiřazení pro cluster s nejvíce přiřazeními.
            
            :return: Počet přiřazení. None => nemá žádný cluster.
            """
            
            return self.__clusters[self.__maxClusterId] if self.__maxClusterId is not None else None
        
        def removeMaxCluster(self):
            """
            Odstraní cluster s nejvíce přiřazeními.
            """
            
            del self.__clusters[self.__maxClusterId]
            if len(self.__clusters)==0:
                self.__maxClusterId=None
            else:
                self.__maxClusterId=max(self.__clusters, key=self.__clusters.get)
            
        def getId(self):
            """
            Získání názvu/identifikátoru kategorie/cíle.
            
            :return: název/identifikátor
            """
            
            return self.__id
        
        def getClusters(self):
            """
            Získání clusterů včetně počtu přiřazení do nich.
            
            :return: Klíč obsahuje identifikátor clusteru a hodnota udává počet přiřazení do tohoto clusteru.
            """
            
            return self.__clusters
        
        def __len__(self):
            """
            Počet clusterů.
            
            :return: Vrací počet přiřazených clusterů.
            """
            return len(self.__clusters)
            
    
    def __init__(self):
        """
        Inicializace klasifikátoru.
        """
        
        self.classes_=None
        self.initCentroids_=None    #počáteční centroidy vstupující do k-means
        self.clusterer=None
        self.clustersTranslator=None
        self.clusterIndexPositionInClasses=None

        
    def fit(self, X, y):
        """
        Natrénuje klasifikátor pomocí cílů trénovacích dat. Parametr X je zde
        pouze z důvodu kompatibility s ostatními klasifikátory z sklearn a není nijak použit.
        
        :param X: array-like | sparse matrix trénovací data
        :param y: cíle
        """
        #nejprve vyfiltrujeme dokumenty k jednotlivým kategoriím/cílům.
        yIndexes={}
        
        for i,actY in enumerate(y):
            if actY not in yIndexes:
                yIndexes[actY]=[]
            yIndexes[actY].append(i)
        
        #získáme centroidy pro cíle/kategorie v definovaném pořadi dle np.unique
        self.classes_=np.unique(y)
             
        self.initCentroids_=np.concatenate([X[yIndexes[t]].mean(axis=0) for t in self.classes_], axis=0)

        #spustíme KMeans
        self.clusterer=KMeans(n_clusters=len(self.classes_)).fit(X)
        
        #vytvoříme slovník pro překlad čísel clusterů na cíle/kategorie
        self.clustersTranslator={}

        #Budeme zjišťovat které clustery a v jakém množství se vyskytují pro trénovací cíle/kategorie.
        #Tedy zjistíme v jakém clusteru se nachází daný dokument a tento cluster přiřadíme k trénovacímu cíli tohoto dokumentu.
        #Zaznamenáváme si i množství takovýchto přiřazení.
         
        countsClustersInY={tY:{} for tY in self.classes_}
        
        for trueL, clusterL in zip(y, self.clusterer.labels_):
            if clusterL not in countsClustersInY[trueL]:
                countsClustersInY[trueL][clusterL]=0
                
            countsClustersInY[trueL][clusterL]+=1
            
        
        labels=[self.LabelsClusters(name, countsClustersInY[name]) for name in self.classes_]
            
        self.clustersTranslator={}
        
        #Vybereme cluster, který byl nejvíce přiřazován.
        allMaxClu=set()
        for label in labels:
            allMaxClu.add(label.maxClusterId())

        #musíme zjistit zda-li nebyl některý cluster přiřazen vícekrát, tedy že je maximální pro více cílů.
        if len(allMaxClu)!=len(labels):
            clustersMissing=set(range(0,len(self.classes_)))-allMaxClu
            
            sortedClustersMissing=sorted(clustersMissing)
            nearNeigh = NearestNeighbors(n_neighbors=len(sortedClustersMissing)).fit(self.clusterer.cluster_centers_[sortedClustersMissing])

            for labelIndex, clustersIds in enumerate(nearNeigh.kneighbors(self.initCentroids_, return_distance=False)):
                #labels[labelIndex[0]].addCluster(clustersIds, float('inf'))
                labels[labelIndex].addNearestFreeClusters([sortedClustersMissing[x] for x in clustersIds])
                
            
            alreadyInAt={}
            

            #projdeme všechny stejné a ponecháme cluster u toho, kde se vyskytuje častěji
            #V připadě, že už u něj další není dostane přednost, ikdyž je méně početný a získá cluster.
            #pokud se vyskytuje stejně necháme jej u jednoho z nich a přiřadíme mu další, který se u něj vyskytoval.
            
            for label in labels:
                self.__findCluster(label, alreadyInAt)

            
        
        self.clustersTranslator={}
        self.clusterIndexPositionInClasses={}

        for label in labels:
            self.clustersTranslator[label.maxClusterId()]=label.getId()
            
            for i, clsName in enumerate(self.classes_):
                if clsName==label.getId():
                    self.clusterIndexPositionInClasses[label.maxClusterId()]=i
                    break

        
        

    def __findCluster(self, actLabelsClusters, alreadyInAt):
        """
        Najde vhodný cluster pro kategorii/cíl.
        
        :type actLabelsClusters: self.LabelsClusters
        :param actLabelsClusters: Aktuální kategorie/cíl a jeho clustery.
        :type alreadyInAt: dict
        :param alreadyInAt: Klíč cluster a hodnota je přiřazený self.LabelsClusters. Slouží pro uchování
            přiřazených labelů ke clusterům a zjištění, které clustery jsou již přiřazené.
        """

        while len(actLabelsClusters)>0: #dokud máme ještě nějaké clustery
            if actLabelsClusters.maxClusterId() not in alreadyInAt:
                #doposud se nevyskystl
                #tedy jej zaregistrujeme
                alreadyInAt[actLabelsClusters.maxClusterId()]=actLabelsClusters
                
                return

            elif actLabelsClusters.maxClusterValue() > alreadyInAt[actLabelsClusters.maxClusterId()].maxClusterValue():
                #jiný label má již tento cluster přiřazený jako nejpočetnější, ale tento je početnější tak jej vyměníme.
                
                old=alreadyInAt[actLabelsClusters.maxClusterId()]
                alreadyInAt[actLabelsClusters.maxClusterId()]=actLabelsClusters
                
                #pro starý najdeme nový
                old.removeMaxCluster()
                self.__findCluster(old, alreadyInAt)
                
                return
            
            else:
                actLabelsClusters.removeMaxCluster()
                
                
        #Pokud jsme se dostali sem znamená, že nám již došly clustery a nemáme žádný přiřazený.
        #vybereme tedy ten, který je nejbližší k centroidu a neni zatím přiřazený.
        
        for clusterId in actLabelsClusters.getNearestFreeClusters():
            if clusterId not in alreadyInAt:
                actLabelsClusters.addCluster(clusterId, float('inf'))
                alreadyInAt[clusterId]=actLabelsClusters
                break
 
        
        
    def predict_proba(self, X):
        """
        Odhadne pravděpodobnost s jakou hodnotou patří daný exemplář do všech natrénovaných tříd.
        Získáváme tedy číslo v intervalu <0,1>, kde jedna je nejpravděpodobnější.
        Pořadí odhadů pravděpodobnosti je určeno pořadím v self.classes_.
        
        :param X:array-like | sparse matrix exempláře
        :return: Vrací list, který obsahuje pro každý exemplář list, který obsahuje prvdepodobnost s
            jakou patří daný exemplář do každé z tříd z self.classes_. self.classes_ určuje pořadí těchto
            pravděpodobností.
        """
        
        centroidsDistances=self.clusterer.transform(X)
        
        probabilities=[]
        
        for distances in centroidsDistances:
            if any(x==0 for x in distances):
                distancesInverse=[1 if d==0 else 0 for d in distances]
            else:
                distancesInverse=[1/d for d in distances]
                
            distancesInverseSum=sum(distancesInverse)
            
            #musíme vypičítat pravděpodobnost a dát ji na správné indexy
            prob=[0]*len(distancesInverse)
            
            for i,p in enumerate([ di/distancesInverseSum for di in distancesInverse]):
                prob[self.clusterIndexPositionInClasses[i]]=p
            
            
            probabilities.append(prob)
        
    
        return probabilities
        
    def predict(self, X):
        """
        Odhad cíle/třídy.
        
        :param X: Data pro klasifikaci. List exemplářu, kde každý exemplář obsahuje list se slovy.
        :return: Vrací list, který obsahuje pro každý exemplář název predikovaného cíle.
        """
        
        predicted=self.clusterer.predict(X)
        
        return [self.clustersTranslator[p] for p in predicted]
        

        