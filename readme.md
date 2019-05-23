# CPKclassifier

Systém pro přiřazení tříd klasifikace Konspekt bibliografickým záznamům, které mohou mít přiřazen i soubor s plnotextovou podobou daného dokumentu. Řešení poskytuje mimo jiné také nástroje pro testování úspěšnosti a předzpracování vstupních textů, se zaměřením na český jazyk.

Informace k nové verzi 2.0 jsou [zde](#verze-2).

# Instalace

Zdrojové soubory pro CPKclassifier si nahrajte do složky, kde budete chtít CPKclassifier mít uložený.
Konkrétně se jedná o soubor CPKclassifier.py a složky config, data a CPKclassifierPack.

* Složka config obsahuje soubor s výchozí konfigurací.
* Složka data obsahuje taggery pro program MorphoDiTa a demonstrační příklady.
* CPKclassifierPack obsahuje balíčky, které CPKclassifier.py používá.

CPKclassifier nevyžaduje instalaci. Je ovšem nutné nainstalovat závislosti a mít nainstalovaný Python řady 3.

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

    pip3 install unidecode

Pro informace k instalaci programu MorphoDiTa prosím navštivte oficiální stránky: http://ufal.mff.cuni.cz/morphodita/install. Je nutné nainstalovat balíček ufal.morphodita.

Pro instalaci závislostí je i možné použít requirements.txt:

	sudo pip3 install -r requirements.txt

	


# Příklady

Předvedeme si použití na příkladech, které objasní, jakým způsobem lze systém používat. Budeme pracovat s triviální datovou sadou, která se nachází ve složce: data/priklady.

Více informací k jednotlivým volbám lze nalézt přímo v konfiguračním souboru. Další informace pak lze získat vypsáním nápovědy:

    ./CPKclassifier.py

Nápovědu pro jednotlivé nástroje pak lze získat například pro předzpracování takto:

    ./CPKclassifier.py preprocessing

U příkladů budeme často používat konfigurační soubory. Přídavné konfigurační soubory přepisují hodnoty, které jsou v nich uvedené, z výchozího konfiguračního souboru.

## Predikce na modelu NKP MZK
Pro tuto část budeme používat naučený model (na datech z NKP a MZK) a konfigurační soubor z:   

    https://knot.fit.vutbr.cz/NAKI_CPK/NKP_MZK_model.zip

Popíšeme si, jakým způsobem se predikují kategorie dokumentu z tohoto předem naučeného modelu.

K modelu je nutné poskytnout data pro klasifikaci. Naučený model požaduje pouze soubor s metadaty. Plné texty nepoužívá.

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
* 6XX_964
    * Používá se pro klasifikaci. Nelze zanedbat. Jedná se o sloučení několika metadatových polí dohromady. Konkrétně toto pole bylo získáno takto: GET_META_FIELDS=600+610+611+630+648+650+651+653+655+695+964:6XX_964


Pro více informací je vhodné se podívat přímo do konfiguračního souboru.

Pro názornost budeme předpokládat, že potřebná data se nachází v adresáři: data.

Predikci spustíme následujícím příkazem:

    ./CPKclassifier.py prediction --metadata data/meta.csv --classifiers data/cls.bin --config data/config.ini --log data/pred.log > data/pred.csv

Předpokládáme předem vytvořený souboru meta.csv s metadaty.  Triviální příklady vstupních souborů jsou v: data/priklady. V souboru pred.log bude zaznamenán průběh operace a výsledky budou uloženy v pred.csv.

V této konfiguraci se klasifikátor pokusí odhadnout tři kategorie (seřazené od nejlepší).


## Předzpracování

V této sekci si uvedeme příklad použití pro:

    ./CPKclassifier.py preprocessing

Budeme chtít lemmatizovat text, separovat znaky ,.:;?! , odstranit stopslova a převést všechny znaky na malé. Výsledek si uložíme do souboru data/priklady/data_p.txt
To vše lze provést následujícím příkazem:

    ./CPKclassifier.py preprocessing --lemmatize --sepSigns --noSW --lc --input data/priklady/data.txt > data/priklady/data_p.txt

## Výběr dat

Budeme používat nástroj:

    ./CPKclassifier.py getData

Chceme vybrat všechny dokumenty, které nemají přiřazenou kategorii, a uložit je do nových souborů.

Je nutné poskytnout soubor s metadaty, tedy: data/priklady/meta.csv. Tento metadatový soubor přísluší k plným textům data/priklady/data.txt, ale i data/priklady/data_p.txt. Jaký vybereme konkrétní soubor, je pro tento příklad nepodstatné.

Dále potřebujeme určit konfigurační soubor. Zde použijeme předem vytvořený soubor:  data/priklady/prep.ini.

Pro vybrání dokumentů, které nemají přiřazenou kategorii, nastavíme parametr EMPTY v sekci GET_DATA takto:

     EMPTY=TARGET

Také chceme mít ve výsledném novém metadatovém souboru všechna pole, nastavíme tedy:

     GET_META_FIELDS=ID TARGET TARGET_REAL

Teď již stačí spustit následující příkaz:

    ./CPKclassifier.py getData --data data/priklady/data.txt --metadata data/priklady/meta.csv --saveDataTo data/priklady/data_bez.txt --saveMetadataTo data/priklady/meta_bez.csv --config data/priklady/prep.ini --log data/priklady/prep.log

Tímto si uložíme náš výběr dat do souborů data_bez.txt a meta_bez.csv. Také si necháme vypsat průběh do logovacího souboru.

## Extrakce příznaků

Budeme používat nástroj

    ./CPKclassifier.py features

V tomto příkladu extrahujeme příznaky z dokumentů, které mají přiřazenou kategorii. Takto extrahované příznaky můžeme později použít pro natrénování klasifikátoru.

Tentokrát použijeme pro plné texty dokumentů soubor: data/priklady/data_p.txt. Příslušný metadatový soubor je: data/priklady/meta.csv.

Konfigurační soubor, který k tomuto budeme potřebovat, je již připravený v: data/priklady/ft.ini.

V sekci GET_DATA jsme nechali oproti výchozí konfiguraci nastavit pouze:

	NON_EMPTY=TARGET

Ve výchozí konfiguraci je již jinak vše vhodně nastaveno – pracujeme s plným textem a jméno pro pole s cílem trénování je nastaveno na TARGET.

Ve FEATURES jsme nastavili parametr FULL_TEXT_VECTORIZER: FULL_TEXT_VECTORIZER=CountVectorizer.

Nakonec spustíme systém pomocí:

    ./CPKclassifier.py features --data data/priklady/data_p.txt --metadata data/priklady/meta.csv --saveTo data/priklady/ft.bin --config data/priklady/ft.ini --log data/priklady/ft.log

Výsledek bude uložen do tří souborů:

* data/priklady/ft.bin
	* Zde je uložena konfigurace a další informace.
* data/priklady/ft.ef
	* Zde jsou uloženy extrahované příznaky.
* data/priklady/ft.ft
	* Zde je uložen natrénovaný nástroj pro extrakci příznaků.

Mimo to si opět necháváme uložit i průběh do logovacího souboru.

## Trénování klasifikátoru

CPKclassifier umožňuje natrénovat jeden či více klasifikátorů. Je tedy možné každému druhu dat přiřadit klasifikátor, který je nejlépe klasifikuje. Výsledky jednotlivých klasifikátorů se agregují pomocí váženého průměru (dle vah klasifikátorů) a díky tomu se model navenek tváří jako jeden klasifikátor.

Nežli se vrhneme na praktické příklady, rozeberme si zde možnou podobu parametru CLASSIFIER z konfiguračního souboru, který slouží k přiřazení dat a vah klasifikátorům:

	CLASSIFIER=A:LinearSVC:0.8 B:KNeighborsClassifier:0.63 C:SGDClassifier:0.79 

Následující řádek definuje, že data A budou klasifikována pomocí LinearSVC s váhou 0.8, data B pomocí KneighborsClassifier s váhou 0.63 a C pomocí SGDClassifier 0.79. Nad každým druhem dat natrénujeme příslušný klasifikátor a při klasifikaci nejprve klasifikujeme každý druh dat zvlášť, tím dostaneme pro daný dokument jistoty od každého klasifikátoru. Tyto jistoty jsou nakonec agregovány pomocí váženého průměru.

### Příklady

Budeme používat nástroj:

    ./CPKclassifier.py classification

Nejprve budeme trénovat klasifikátor z předem extrahovaných a uložených příznaků. Poté si ukážeme, jak natrénovat klasifikátor přímo z dat.

Přídavný konfigurační soubor tentokrát nepoužijeme vůbec. Vystačíme s výchozí konfigurací.

Spustíme tedy pouze příkaz:

    ./CPKclassifier.py classification --features data/priklady/ft.bin --saveTo data/priklady/cls.bin --log data/priklady/cls.log

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

     ./CPKclassifier.py classification --data data/priklady/data_p.txt --metadata data/priklady/meta.csv --config data/priklady/ft.ini --saveTo data/priklady/cls_d.bin --log data/priklady/cls_d.log

## Predikce

Budeme používat nástroj:

     ./CPKclassifier.py prediction

Je nutné mít již předem natrénovaný klasifikátor (sekce: Trénování klasifikátoru).

Necháme si klasifikovat dokumenty, které nemají v našich datech přiřazenou kategorii.

Použijeme k tomu předem připravený konfigurační soubor: data/priklady/pred.ini.

Chceme pracovat pouze s dokumenty, které nemají kategorii. V sekci GET_DATA jsme tedy nastavili: EMPTY=TARGET.

Protože chceme vypsat k výsledkům i další metadata, nastavili jsme v sekci PREDICTION parametr: WRITE_META_FIELDS=ID TARGET_REAL.
Položka ID je uvedena, neboť chceme na výstupu i identifikátor klasifikovaného dokumentu. V případě vyhodnocování úspěšnosti na testovacích datech máme na výstupu i kategorii určenou v testovacím záznamu knihovníky (REAL).

Použijeme klasifikátor vzniklý procesem trénování popsaný v části Trénování klasifikátoru.


Příkaz:

     ./CPKclassifier.py prediction --data data/priklady/data_p.txt --metadata data/priklady/meta.csv --classifiers data/priklady/cls.bin --config data/priklady/pred.ini --log data/priklady/pred.log

Dostaneme výstup ve formátu CSV, např.:

     PREDICTED,ID,TARGET_REAL
     pozdrav,9,pozdrav
     zvířata,10,zvířata
     geometrie,11,geometrie

## Testování

Budeme používat nástroj:

     ./CPKclassifier.py testing

Otestujeme vytvořený klasifikátor. Opět použijeme již před připravený konfigurační soubor. Musíme v něm uvést, s jakými daty chceme pracovat, jak chceme extrahovat příznaky, jak se má trénovat klasifikátor, a v neposlední řadě i způsob testování (popřípadě i nastavení predikce).

Použijeme následující konfiguraci: data/priklady/test.ini.

Z velké části vystačíme s výchozím nastavením. Pouze nastavíme, že chceme jen dokumenty, které mají přiřazenou kategorii (NON_EMPTY=TARGET). Pro testování potřebujeme právě takové dokumenty. Nastavíme FULL_TEXT_VECTORIZER=CountVectorizer, stejně jako v předchozích příkladech. Dále, protože si necháme vypsat predikce z jednotlivých kroků, nastavíme WRITE_META_FIELDS=ID TARGET_REAL. Počet validačních kroků nastavíme: SPLITS=3.

Příkaz:

     ./CPKclassifier.py testing --data data/priklady/data_p.txt --metadata data/priklady/meta.csv --writeResults data/priklady/test.res --writeConfMetrix data/priklady/test.cmat --config data/priklady/test.ini --log data/priklady/test.log

Výsledky z testování dostaneme na standardním výstupu. Dále si ukládáme matici záměn (data/priklady/test.cmat) a výsledky predikcí z jednotlivých kroků (data/priklady/test.res). Jako vždy ukládáme i log.

# Verze 2

Jedná se o novou verzi systému pro klasifikaci, která vznikla z DocClassifier. Tato nová verze obohacuje starší systém o několik nových funkcí. Významnější změny oproti staršímu systému jsou uvedeny níže. Uvedené parametry, v textu níže, se odkazují na příslušné parametry konfiguračních souborů a vždy se vztahují k sekci, která je uvedena v hranatých závorkách u nadpisu.

## Předzpracování [PREPROCESSING]

Došlo k optimalizaci, paralelizaci a opravě známých chyb u předzpracování.

Nově je tedy možné pustit předzpracování v paralelním režimu. Je možné nastavit počet podílejících se procesů na převodu vstupních dat parametrem WORKERS. Samotná dělba práce mezi procesy probíhá tak, že jeden proces (hlavní) prochází vstupní soubor a čte jej řádek po řádku. Prázdné řádky nedeleguje, ale přímo zpracovává. Delegování prázdných řádku způsobovalo velké prodlevy při čekání u vstupu do kritické sekce (užitečná práce byla kratší než režie pro synchronizaci).
Neprázdné řádky jsou dále delegovány ostatním procesům.

Neprázdný řádek může být rozdělen na více částí, které budou zpracovávány zvlášť a to ze dvou důvodů.

Prvním důvodem je více homogenizovat zátěž mezi procesy, jelikož nějaký proces může získat velký dokument a dlouho jej zpracovávat zatímco ostatní procesy nemají žádnou užitečnou práci. Velikost části lze nastavit parametrem MAX_NUMBER_OF_WORDS_PER_LINE_PART.

Druhým důvodem je omezení, které klade použitá knihovna pro komunikaci mezi paralelními procesy, která si vynucuje rozdělení dokumentu na menší části.

## Získávání dat [GET_DATA]

Nově je možné sloučit několik metadatových polí do sebe či utvářet jejich kopie. Nastavení se provádí v poli GET_META_FIELDS.

Kopie umožňují například spustit trénování/klasifikaci nad stejným druhem dat, ale s použitím samostatného klasifikátoru pro originál a pro kopii.

Sloučení polí umožňuje začlenit více polí do jednoho nového pole a tím potenciálně zvětšit množinu příznaků.


## Příznaky [FEATURES]

V nové verzi lze spustit extrakci příznaků v paralelním režimu pomocí parametru WORKERS. Trénování jednoho extraktoru příznaků má vždy na zodpovědnost jeden proces. Samotná extrakce příznaků, pomocí natrénovaného extraktoru, pak probíhá tak, že se nejprve extrahuje pomocí jednoho extraktoru, procesy si dělí jednotlivé dokumenty mezi sebou, a pak se potenciálně pokračuje na další druh dat.

Extraktor může získat svůj slovník nad jiným druhem dat než, nad kterým bude provádět samotnou extrakci. Nastavení probíhá pomocí parametrů FULL_TEXT_VECTORIZER_BUILD_VOCABULARY_ON a META_VECTORIZERS_BUILD_VOCABULARY_ON.
V neposlední řadě byla rozšířena práce s prázdnými dokumenty, aby nebylo nutné mít pro klasifikaci či trénování klasifikátoru u daného dokumentu vždy všechny druhy dat, ale třeba jen některé. Nastavení se provádí pomocí parametru SKIP_EMPTY. Nicméně je nutné si uvědomit, že některé dokumenty nemusí být klasifikovány vůbec, pokud nemají přiřazeny ani jeden druh dat. V takovém případě by byl vrácen prázdný výsledek.

## Trénování klasifikátoru [CLASSIFICATION]

V této verzi je možné trénovat několik klasifikátorů paralelně. Nastavení se provádí, obdobně jako v předchozích případech, parametrem WORKERS.

Dále je možné nastavit práh pro klasifikaci u jednotlivých klasifikátorů, kdy je možné říct, že pokud si je klasifikátor s daným výsledkem klasifikace jist jenom na 30%, tak jej nebude systém započítávat k výsledkům ostatních klasifikátorů a nebude zahrnut v celkovém výsledku. Tím i automaticky dochází k snížení celkové jistoty. Teoreticky 100% jistotu můžeme získat jen u dokumentů, které obsahují všechny požadované druhy dat a všechny klasifikátory jsou si 100% jisty. Nastavení prahu je v parametru CLASSIFIER.

Nová verze zavádí automatické získávání vah pro klasifikátory. Získávání vah probíhá otestováním klasifikátoru na dané trénovací množině a to v několika křížově validačních krocích (počet kroků lze nastavit v WEIGHT_AUTO_CV). Průměrná úspěšnost je potom použita jako váha. Na rozdíl od fixní váhy jsou získávány váhy pro jednotlivé natrénované kategorie/třídy zvlášť. Můžeme tedy tímto způsobem říct, že daný klasifikátor je slabý pro určování jedné kategorie, ale naopak pro druhou je dobrý. Nastavení automatické váhy je v CLASSIFIER.

## Predikce [PREDICTION]

Predikce byla paralelizována. Příslušné nastavení, lze nalézt v WORKERS.

Dále je umožněno nastavit práh pro celkovou jistotu, tedy ne jen pro daný klasifikátor jako u sekce CLASSIFICATION. Pokud bude celková jistota klasifikace dokumentu pod úrovní prahu, tak jej má ponechat jako neklasifikovaný a vrátit prázdný výsledek. Nastavení je v THRESHOLD.

## Statistiky [STATS]
Tato sekce je zcela nová a umožňuje získat dodatečné statistiky z csv souborů, které vznikají například při klasifikaci. Vypisuje následující statistiky:

* počet dokumentů
* počet pravých kategorií
* počty dokumentů v pravých kategoriích
* počet predikovaných kategorií
* počty dokumentů v predikovaných kategoriích
