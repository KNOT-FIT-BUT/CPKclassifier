﻿# DocClassifier

Systém pro přiřazení tříd klasifikace Konspekt bibliografickým záznamům, které mohou mít přiřazen i soubor s plnotextovou podobou daného dokumentu. Řešení poskytuje mimo jiné také nástroje pro testování úspěšnosti a předzpracování vstupních textů, se zaměřením na český jazyk.


# Instalace

Zdrojové soubory pro DocClassifier si nahrajte do složky, kde budete chtít DocClassifier mít uložený.
Konkrétně se jedná o soubor DocClassifier.py a složky config, data a DocClassifierPack.

* Složka config obsahuje soubor s výchozí konfigurací.
* Složka data obsahuje taggery pro program MorphoDiTa a demonstrační příklady.
* DocClassifierPack obsahuje balíčky, které DocClassifier.py používá.

DocClassifier nevyžaduje instalaci. Je ovšem nutné nainstalovat závislosti a mít nainstalovaný Python řady 3.

## Instalace závislostí

Je nutné nainstalovat tyto závislosti:

* unidecode
* gensim
* numpy
* pandas
* scikit_learn
* scipy
* ufal.morphodita (MorphoDiTa)

Je možné použít nástroj pip (pip3). Tedy například:

    pip install unidecode

Pro informace k instalaci programu MorphoDiTa prosím navštivte oficiální stránky: http://ufal.mff.cuni.cz/morphodita/install. Je nutné nainstalovat balíček ufal.morphodita.


# Příklady

Předvedeme si použití na příkladech, které objasní, jakým způsobem lze systém používat. Budeme pracovat s triviální datovou sadou, která se nachází ve složce: data/priklady.

Více informací k jednotlivým volbám lze nalézt přímo v konfiguračním souboru. Další informace pak lze získat vypsáním nápovědy:

    ./DocClassifier.py

Nápovědu pro jednotlivé nástroje pak lze získat například pro předzpracování takto:

    ./DocClassifier.py preprocessing

U příkladů budeme často používat konfigurační soubory. Přídavné konfigurační soubory přepisují hodnoty, které jsou v nich uvedené, z výchozího konfiguračního souboru.

## Predikce na modelu NAKI CPK

Pro tuto část budeme používat naučený model a konfigurační soubor z: https://knot.fit.vutbr.cz/NAKI_CPK/naki_cpk_model.zip. Popíšeme si, jakým způsobem se predikují kategorie dokumentu z tohoto předem naučeného modelu.

Dále je nutné poskytnout data pro klasifikaci. Naučený model požaduje soubor s plnými texty a soubor s metadaty. Triviální příklady těchto souborů jsou dostupné v: data/priklady.

Metadata:

* dedup_record_id
  * ID dokumentu. Používá se jako dodatečná informace vypisovaná k výsledkům. Toto pole je možné zanedbat odstraněním z parametru WRITE_META_FIELDS v konfiguračním souboru.
* 072_HIER
    * Toto pole není nezbytné a je možné jej zanedbat odstraněním z parametru EMPTY v konfiguračním souboru. Slouží jen pro možnost mít klasifikované a neklasifikované dokumenty společně.
* 245
    * Používá se jako dodatečná informace vypisovaná k výsledkům. Toto pole je možné zanedbat odstraněním z parametru WRITE_META_FIELDS v konfiguračním souboru.
* 245_lemm
    * Lemmatizovaná verze 245. Používá se pro klasifikaci. Nelze zanedbat.
* 100
    * Používá se jako dodatečná informace vypisovaná k výsledkům. Toto pole je možné zanedbat odstraněním z parametru WRITE_META_FIELDS v konfiguračním souboru.
* 080
    * Používá se pro klasifikaci. Nelze zanedbat.

Pro více informací je vhodné se podívat přímo do konfiguračního souboru.

Pro názornost budeme předpokládat, že potřebná data se nachází v: data/NAKI.

Predikci spustíme následujícím příkazem:

    ./DocClassifier.py prediction --data data/NAKI/data.txt --metadata data/NAKI/meta.csv --classifiers data/NAKI/NAKI_cls.bin --config data/NAKI/config.ini --log data/NAKI/pred.log > data/NAKI/pred.csv

Předpokládáme vytvoření souboru data.txt s plnými texty a souboru meta.csv s metadaty. V souboru pred.log bude zaznamenán průběh operace a výsledky budou uloženy v pred.csv.

V této konfiguraci se klasifikátor pokusí odhadnout tři kategorie (seřazené od nejlepší).

## Předzpracování

V této sekci si uvedeme příklad použití pro:

    ./DocClassifier.py preprocessing

Budeme chtít lemmatizovat text, separovat znaky ,.:;?! , odstranit stopslova a převést všechny znaky na malé. Výsledek si uložíme do souboru data/priklady/data_p.txt
To vše lze provést následujícím příkazem:

    ./DocClassifier.py preprocessing --lemmatize --sepSigns --noSW --lc --input data/priklady/data.txt > data/priklady/data_p.txt

## Výběr dat

Budeme používat nástroj:

    ./DocClassifier.py getData

Chceme vybrat všechny dokumenty, které nemají přiřazenou kategorii, a uložit je do nových souborů.

Je nutné poskytnout soubor s metadaty, tedy: data/priklady/meta.csv. Tento metadatový soubor přísluší k plným textům data/priklady/data.txt, ale i data/priklady/data_p.txt. Jaký vybereme konkrétní soubor, je pro tento příklad nepodstatné.

Dále potřebujeme určit konfigurační soubor. Zde použijeme předem vytvořený soubor:  data/priklady/prep.ini.

Pro vybrání dokumentů, které nemají přiřazenou kategorii, nastavíme parametr EMPTY v sekci GET_DATA takto:

     EMPTY=TARGET

Také chceme mít ve výsledném novém metadatovém souboru všechna pole, nastavíme tedy:

     GET_META_FIELDS=ID TARGET TARGET_REAL

Teď již stačí spustit následující příkaz:

    ./DocClassifier.py getData --data data/priklady/data.txt --metadata data/priklady/meta.csv --saveDataTo data/priklady/data_bez.txt --saveMetadataTo data/priklady/meta_bez.csv --config data/priklady/prep.ini --log data/priklady/prep.log

Tímto si uložíme náš výběr dat do souborů data_bez.txt a meta_bez.csv. Také si necháme vypsat průběh do logovacího souboru.

## Extrakce příznaků

Budeme používat nástroj

    ./DocClassifier.py features

V tomto příkladu extrahujeme příznaky z dokumentů, které mají přiřazenou kategorii. Takto extrahované příznaky můžeme později použít pro natrénování klasifikátoru.

Tentokrát použijeme pro plné texty dokumentů soubor: data/priklady/data_p.txt. Příslušný metadatový soubor je: data/priklady/meta.csv.

Konfigurační soubor, který k tomuto budeme potřebovat, je již připravený v: data/priklady/ft.ini.

V sekci GET_DATA jsme nechali oproti výchozí konfiguraci nastavit pouze:

	NON_EMPTY=TARGET

Ve výchozí konfiguraci je již jinak vše vhodně nastaveno – pracujeme s plným textem a jméno pro pole s cílem trénování je nastaveno na TARGET.

Ve FEATURES jsme nastavili parametr FULL_TEXT_VECTORIZER: FULL_TEXT_VECTORIZER=CountVectorizer.

Nakonec spustíme systém pomocí:

    ./DocClassifier.py features --data data/priklady/data_p.txt --metadata data/priklady/meta.csv --saveTo data/priklady/ft.bin --config data/priklady/ft.ini --log data/priklady/ft.log

Výsledek bude uložen do tří souborů:

* data/priklady/ft.bin
	* Zde je uložena konfigurace a další informace.
* data/priklady/ft.ef
	* Zde jsou uloženy extrahované příznaky.
* data/priklady/ft.ft
	* Zde je uložen natrénovaný nástroj pro extrakci příznaků.

Mimo to si opět necháváme uložit i průběh do logovacího souboru.

## Trénování klasifikátoru

Budeme používat nástroj:

    ./DocClassifier.py classification

Nejprve budeme trénovat klasifikátor z předem extrahovaných a uložených příznaků. Poté si ukážeme, jak natrénovat klasifikátor přímo z dat.

Přídavný konfigurační soubor tentokrát nepoužijeme vůbec. Vystačíme s výchozí konfigurací.

Spustíme tedy pouze příkaz:

    ./DocClassifier.py classification --features data/priklady/ft.bin --saveTo data/priklady/cls.bin --log data/priklady/cls.log

Výsledek bude uložen do čtyř souborů:

* data/priklady/cls.bin
	* Zde je uložena konfigurace a další informace.
* data/priklady/cls.bin.cls
	* Zde je uložen natrénovaný klasifikátor.
* data/priklady/cls.bin.ef
	* Zde jsou uloženy extrahované příznaky.
* data/priklady/cls.bin.ft
	* Zde je uložen natrénovaný nástroj pro extrakci příznaků.

Mimo to si opět necháváme uložit i průběh do logovacího souboru.

Jako druhý příklad si uvedeme trénování přímo z dat. Bude se jednat pouze o sloučení postupu výše uvedeného a toho v příkladu Extrakce příznaků.

Protože chceme pro trénování použít výchozí konfiguraci a pro extrakci příznaků stejnou konfiguraci, která byla použita v části Extrakce příznaků, můžeme jako přídavný konfigurační soubor znovu použít: data/priklady/ft.ini.

Příkaz by tedy vypadal:

     ./DocClassifier.py classification --data data/priklady/data_p.txt --metadata data/priklady/meta.csv --config data/priklady/ft.ini --saveTo data/priklady/cls_d.bin --log data/priklady/cls_d.log

## Predikce

Budeme používat nástroj:

     ./DocClassifier.py prediction

Je nutné mít již předem natrénovaný klasifikátor (sekce: Trénování klasifikátoru).

Necháme si klasifikovat dokumenty, které nemají v našich datech přiřazenou kategorii.

Použijeme k tomu předem připravený konfigurační soubor: data/priklady/pred.ini.

Chceme pracovat pouze s dokumenty, které nemají kategorii. V sekci GET_DATA jsme tedy nastavili: EMPTY=TARGET.

Protože chceme vypsat k výsledkům i další metadata, nastavili jsme v sekci PREDICTION parametr: WRITE_META_FIELDS=ID TARGET_REAL.
Položka ID je uvedena, neboť chceme na výstupu i identifikátor klasifikovaného dokumentu. V případě vyhodnocování úspěšnosti na testovacích datech máme na výstupu i kategorii určenou v testovacím záznamu knihovníky (REAL).

Použijeme klasifikátor vzniklý procesem trénování popsaný v části Trénování klasifikátoru.


Příkaz:

     ./DocClassifier.py prediction --data data/priklady/data_p.txt --metadata data/priklady/meta.csv --classifiers data/priklady/cls.bin --config data/priklady/pred.ini --log data/priklady/pred.log

Dostaneme výstup ve formátu CSV, např.:

     PREDICTED,ID,TARGET_REAL
     pozdrav,9,pozdrav
     zvířata,10,zvířata
     geometrie,11,geometrie

## Testování

Budeme používat nástroj:

     ./DocClassifier.py testing

Otestujeme vytvořený klasifikátor. Opět použijeme již před připravený konfigurační soubor. Musíme v něm uvést, s jakými daty chceme pracovat, jak chceme extrahovat příznaky, jak se má trénovat klasifikátor, a v neposlední řadě i způsob testování (popřípadě i nastavení predikce).

Použijeme následující konfiguraci: data/priklady/test.ini.

Z velké části vystačíme s výchozím nastavením. Pouze nastavíme, že chceme jen dokumenty, které mají přiřazenou kategorii (NON_EMPTY=TARGET). Pro testování potřebujeme právě takové dokumenty. Nastavíme FULL_TEXT_VECTORIZER=CountVectorizer, stejně jako v předchozích příkladech. Dále, protože si necháme vypsat predikce z jednotlivých kroků, nastavíme WRITE_META_FIELDS=ID TARGET_REAL. Počet validačních kroků nastavíme: SPLITS=3.

Příkaz:

     ./DocClassifier.py testing --data data/priklady/data_p.txt --metadata data/priklady/meta.csv --writeResults data/priklady/test.res --writeConfMetrix data/priklady/test.cmat --config data/priklady/test.ini --log data/priklady/test.log

Výsledky z testování dostaneme na standardním výstupu. Dále si ukládáme matici záměn (data/priklady/test.cmat) a výsledky predikcí z jednotlivých kroků (data/priklady/test.res). Jako vždy ukládáme i log.