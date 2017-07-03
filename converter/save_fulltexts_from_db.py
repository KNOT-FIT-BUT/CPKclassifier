#!/usr/bin/env python3
"""
@ name          save_fulltexts_from_db.py
@ author        xkurak00, xormos00
@ date          June 2016
@ last_update   Jun 2016
@ license       VUT FIT UPGM
@ project       NAKI CPK
@ description   Skript na export fulltextov z databazy,

Skript si najprv nacita vsetky unikatne harvested_record_id z fulltext_kramerius
nasledne si ich zoradi podla deduplikacneho id a zacne exportovat fulltexty
"""


import psycopg2, sys, re, argparse, os

# Inicializacia parsera
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

# Nacitanie vsetkych unikatnych harvested_record_id z fulltext_kramerius
cur = conn.cursor()
command = "SELECT DISTINCT harvested_record_id FROM fulltext_kramerius"
cur.execute(command)
record_ids = [str(x[0]) for x in cur.fetchall()]
# Priradenie dedup_record_id pre kazdy record
dedup_ids = {}
for r_id in record_ids:
    try:
        command = "SELECT dedup_record_id FROM harvested_record WHERE id=" + r_id
        cur.execute(command)
        d_id = str(cur.fetchall()[0][0])
        if d_id not in dedup_ids:
            dedup_ids[d_id] = []
        dedup_ids[d_id].append(r_id)
    except Exception as e:
        print(e, file=sys.stderr)

# d_id je dedup_record_id h_ids je pole harvested_record_id
# Generovanie textu pre vsetky dedup id
for d_id, h_ids in dedup_ids.items():
    final_text = []
    # Exportovanie textu pre jednotlive harvested_record_id
    for h_id in h_ids:
        command = "SELECT order_in_document, fulltext FROM fulltext_kramerius WHERE harvested_record_id=" + h_id
        cur.execute(command)
        pack = cur.fetchall()
        pages = [(int(a[0]), a[1].tobytes().decode('utf-8')) for a in pack]
        text = ' '.join(x[1] for x in sorted(pages, key=lambda x: x[0]))
        text = re.sub(r'-\r?\n', '', text)
        text = re.sub(r'\r?\n', ' ', text)
        final_text.append(text)
    with open(save_path + d_id + '.txt', 'w') as save_file:
        save_file.write(' '.join(final_text) + '\n')
