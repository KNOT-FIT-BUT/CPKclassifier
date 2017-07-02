# About

V tomto adresáry sú popísané a prezentované skripty, ktoré sa využívajú pre prípravu dát pre klasifikátor.
Skripty boli navrhnuté a vytvorené pre používanie s databázou dumpu, ktorý nám bol poskytnutý a náležite upravený.

# save_metadata_from_db.py

Získavanie a spracovanie fulltextov z databázy.

Pripojenie na databázu. 

Vytvorenie mapy knižníc. 

Príprava polí metadát, ktoré chceme získať.

Následne sa prechádzajú jednotlivé záznamy, z ktorých sa získavajú požadované metadáta a záznam sa priradzuje ku knižnici ku ktorej patrí.

Všetko sa zapisuje do súborov lib_XYZ.tsv, kde XZY predstavuje čislo knižnice podľa pridelenia v databáze. lib_map.tsv toto rozdelenie mapuje.

Priklad spustenia:
	
	python3 export_fields_to_files.py -sp /mnt/data-2/xkurak00/NAKI_CPK/data/lib_tsv_from_db_2017

# save_fulltexts_from_db.py

Získavanie a spracovanie fulltextov z databázy.

Pripojenie na databázu.

Získanie fulltextov podľa dedup_record_id.

Ich následne uloženie vo formáte .txt vo forme jedného riadku.

Príklad spustenia:
	
	python3 save_fulltexts_from_db.py -sp /mnt/data-2/xkurak00/NAKI_CPK/data/fulltext/fulltexts_from_db_201703/


# fulltext_parser.py

Spracovanie z fyzicky uložených fulltextov vo formáte .xml zabalených v tar.gz repozitároch.

Rozbaľovanie repozitára. 

Otváranie jednotlivých .xml a získavanie kľúčových vlastností. 

Výsledok v .txt súbory vo vertikalizovanom texte.

Priklad spustenia:
	
	python3 fulltext_parser.py input_fulltexts output_fulltexts 15
