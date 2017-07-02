#!/usr/bin/env python3
"""
@ name          save_metadata_from_db.py
@ old_name      export_fields_to_files.py
@ author        xkurak00, xormos00
@ date          June 2016
@ last_update   June 2017
@ license       VUT FIT UPGM
@ project       NAKI CPK
@ description   Skript na export metadat z databazy,
                 uklada jednotlive zaznamy podla kniznic
"""

import psycopg2, json, sys, re, argparse, os
import xml.etree.ElementTree as ET
import xml.dom.minidom as MM
import sys
import getpass
import time
from math import fmod

# Measure time of apllication run 1-is-on 0-is-off
debug_time = 1
if (debug_time):
    start_time = time.time()

# Priprava parseru a kontrola ciest
parser = argparse.ArgumentParser(description='Lemmatizer')
parser.add_argument('-sp', '--save_path', required=True,
                    help='Directory where created files should be saved')

args = vars(parser.parse_args())
save_path = os.path.abspath(args['save_path']) + '/'
if not os.path.exists(save_path):
    print("Wrong save directory", file=sys.stderr)
    sys.exit(1)

# Pripojenie na databazu
input_dbname = input("Please enter the name of database: ")
input_user = input("Please enter the user name of database: ")
input_host = input("Please enter host of database: ")
input_passwd = getpass.getpass('Password:')
print ("Connecting to: " + input_dbname + " | " + input_user + " | " + input_host + " | " + "**********")
try:
    conn = psycopg2.connect(dbname=input_dbname, user=input_user, host=input_host, password=input_passwd)
except:
    print("I am unable to connect to the database")
    sys.exit(1)

cur = conn.cursor()

# Nacitanie id a mien kniznic
# Zmapovanie import_conf pre kazdy record na jednotlive kniznice
# Ulozenie mapovacieho suboru lib_map.tsv
print("Loading lib tag", flush=True)
tsv_files_dir = save_path
cur.execute("""SELECT id, name FROM library""")
lib_rows = cur.fetchall()
lib_name = {}
lib_keys = {}
with open(tsv_files_dir + 'lib_map.tsv', 'w') as map_file:
    for i in lib_rows:
        code = int(i[0])
        lib_name[code] = i[1]
        map_file.write(str(code) + '\t' + i[1] + '\n')
        cur.execute("SELECT id FROM import_conf WHERE library_id=" + str(i[0]))
        info_rows = cur.fetchall()
        for a in info_rows:
            lib_keys[int(a[0])] = code

# Vytvorenie a otvorenie subor pre jednotlive kniznice
export_files = {}
for library in lib_name:
    export_files[library] = open(tsv_files_dir + 'lib_' + str(library) + '.tsv','w')

cur.execute("""SELECT id FROM dedup_record LIMIT 10""")

dedup_rows = cur.fetchall()

# Zoznam poli, ktore sa exportuju
fields = ["008", "072", "080", "600", "610", "611", "630", "648", "100",
      "650", "651", "653", "655", "670", "678", "695", "964", "245"]


# Prechadzanie jednotlivych dedup id
counter = 1
print('Exporting data', flush=True)
for i in dedup_rows:
    print("counter status: %d" % counter)
    counter = counter + 1
    d_id = str(i[0])
    cur = conn.cursor()
    # Select vsetkych records, ktore su priradene pre dany dedup_id
    cmd = "SELECT dedup_record_id, format, raw_record, import_conf_id from harvested_record where length(raw_record) > 0 and dedup_record_id=" + d_id
    cur.execute(cmd)
    rows = cur.fetchall()
    lib_split = {}
    for row in rows:
        # Odstranenie tagov na zaciatku xml suboru,
        # aby bolo mozne pouzivat xml parser pre hladanie datafieldov
        try:
            tmp = str(row[2].tobytes().decode('utf-8')).replace("marc:", '')
            text = re.sub(r"(?<=<record)[^>]*?(?=>)", "", tmp)
            tree = ET.fromstring(text)
        except Exception as e:
            print(e, file=sys.stderr)
            continue
        # Priradenie kniznice podla import_conf_id a vytvorenie
        # dict pre ukladanie zaznamov
        lib_key = lib_keys[int(row[3])]
        if lib_key not in lib_split:
            lib_split[lib_key] = {}
        for field in fields:
            lib_split[lib_key][field] = []

        # Prechadzanie controlfield kvoli polu 008
        for child in tree.findall('controlfield'):
            try:
                if child.attrib['tag'] == '008':
                    lib_split[lib_key]['008'].append(child.text)
            except:
                continue
        # Prechadzanie vsetkych datafields a vytahovanie vyznamnych poli
        for child in tree.findall('datafield'):
            try:
                tag = child.tag
                atr = child.attrib['tag']
                if atr not in fields:
                    continue
            except:
                continue
            if atr.startswith('6') and 'ind2' in child.attrib and child.attrib['ind2'] == '9':
                continue
            text = []
            # Prehladavanie jednotlivych "codov" v poliach
            for i in child:
                if atr != '245':
                    codes = ['a', 'x']
                else:
                    codes = ['a', 'b', 'c', 'x']
                if 'code' in i.attrib and i.attrib['code'] in codes:
                    if i.text != None:
                        text.append(i.text)
            lib_split[lib_key][atr].append(' '.join(text))

    # Ulozenie a zapis jednotlivych zaznamov pre kniznice
    for lib in lib_split:
        text = []
        for field in fields:
            text.append(' $|$ '.join(lib_split[lib][field]))
        if not ''.join(text).strip():
            continue
        export_files[lib].write('\t'.join([d_id]+text)+'\n')

# Zatvorenie suborov
[item.close() for key, item in export_files.items()]
print('Done')

if(debug_time):
    print("Time of run : --- %s seconds ---" % (time.time() - start_time))
