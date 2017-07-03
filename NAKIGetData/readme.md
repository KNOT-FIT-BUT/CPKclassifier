# Skript pro převod dat
Skript slouží pro převedení dat do vhodné podoby pro klasifikátor.

Ovládá se pomocí argumentů a také pomocí konfiguračního souboru config.ini.

Umožňuje převést jak plné texty, tak i metadata. Je možné slučovat metadata z více zdrojů. Pokud tedy uvedeme více zdrojů, metadata se budou navzájem doplňovat.

Očekává na vstupu soubor, který slouží jako seznam metadatových souborů, které se mají použít.
Jako příklad může posloužit soubor meta_map.csv. Je v něm možné vidět, že každý soubor má svoji prioritu. Položky metadatových polí v souborech s vyšší prioritou budou v rámci pole řazeny před ostatní položky. Toto je vhodné například pokud máme více zdrojů a každý uvádí jinou kategorii dokumentu. Zdroji, kterému věříme více přiřadíme vyšší prioritu.

Skript dále očekává soubor, který mapuje kategorie do hierarchie. Jako ukázkový soubor slouží hier.tsv. V prvním sloupci jsou uvedeny kategorie v původní podobě a v dalších sloupcích je k nim odpovídající podoba v hierarchickém formátu. Tedy například pro:

    Teorie čísel	Matematika	Teorie čísel

Dojde k převodu Teorie čísel na:

    Matematika->Teorie čísel.

V argumentech skriptu se také uvádí cesta k adresáři, kde jsou uloženy soubory, které obsahují plné texty dokumentů.

Je možné pomocí argumentu enableEmptyFulltext povolit začlenění dokumentů, které mají prázdný soubor s plným textem nebo takový soubor nemají vůbec. V opačném případě se takovéto dokumenty neuvažují.


## Příklad
Dejme tomu, že chceme převést data dokumentů, které mají i prázdný plný text do podoby pro klasifikátor a chceme ukládat jak plné texty, tak i metadata. Mimo to budeme chtít záznam o průběhu operace. Pro takovouto úlohy použijeme následující kombinaci argumentů:

    ./NAKIGetData.py --enableEmptyFulltext --saveDataTo data.txt --saveMetaDataTo meta.csv --metadataFilesMap meta_map.csv --hierarchyFile hier.tsv --log ./result.log --docFolder slozkasplnymitexty
