#!/usr/bin/python3
#
# Klasifikátor poľa 072 z polí 080, 600, 610, 611, 630, 648, 008, 100,
#	                           650, 651, 653, 655, 670, 678, 695, 964, 245.
#
# autor: Ondrej Kurák
# email: xkurak00@stud.fit.vutbr.cz

import numpy as np
import time, pickle, re, os, sys, inspect, argparse, urllib.request
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from lemmatize import lemmatizer
import time
import hashlib
import urllib
from urllib.request import urlretrieve


class item_selector():

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


def filter_text(text):
	tmp = re.sub(r'\$.',' ', text)
	tmp = re.sub(r'\W+',' ', tmp)
	tmp = re.sub(r'\ +', ' ', tmp)
	return tmp.lower()


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        print(s, flush='True')
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

filename = inspect.getframeinfo(inspect.currentframe()).filename
my_dir = os.path.dirname(os.path.abspath(filename)) + '/'

# Arg parser initialization
parser = argparse.ArgumentParser(description='NAKI konspekt(072) classifier')
parser.add_argument('-rc', '--records_path', required=True,
					help='Path to the records.mrc file')
parser.add_argument('-sp', '--save_path', required=True, default=my_dir,
					help='Directory where created file should be saved')

args = vars(parser.parse_args())

if args['save_path'] == args['records_path']:
	print("ERROR: Incorrect output file", file=sys.stderr)
	sys.exit(0)

if not os.path.isfile(args['records_path']):
	print('EROOR: [records_path] is not a file')
	sys.exit(0)
try:
	with open(args['save_path'], 'w') as output_file:
		pass
except:
	print('EROOR: [save_path] incorrect save path', file=sys.stderr)
	sys.exit(1)

# Download
down_link = "http://knot.fit.vutbr.cz/NAKI_CPK/"
files = ['cat_data.pickle', 'classifier_data.jlib',
		 'czech-morfflex-160310.dict', 'md5sum.txt']
for file_name in files:
	if os.path.isfile( my_dir + 'data/' + file_name):
		continue
	start_time = time.time()
	print('Donwloading', file_name, end=' ... ', flush=True)
	urlretrieve(down_link + file_name, my_dir + 'data/' + file_name)
	print("Done(", round(time.time() -  start_time, 0),"s)", sep='', flush=True)

start_time = time.time()
print("md5sum check:",end = " ", flush = True)
with open(my_dir + 'data/' + 'md5sum.txt') as md5_file:
	for line in md5_file:
		check = line[:32]
		name = line[34:-1]
		try:
			if check != md5(my_dir + 'data/' + name):
				raise
		except:
				print('EROOR: md5sum of',name, file=sys.stderr)
				sys.exit(1)
print("Done(", round(time.time() -  start_time, 0),"s)", sep='', flush=True)

# Loading data
start_time = time.time()
print("Loading data:",end = " ", flush = True)
with open(my_dir + 'data/cat_data.pickle', 'rb') as config_file:
        cat_names = pickle.load(config_file)
clf = joblib.load( my_dir + 'data/classifier_data.jlib')

dict_path = my_dir + 'data/czech-morfflex-160310.dict'
pipeline = Pipeline([
        ('union', FeatureUnion(
        transformer_list = [
            ('meta_data', Pipeline([
                ('selector', item_selector(key='meta_data')),
                ('lemma', lemmatizer(dict_path=dict_path)),
                ('vect', HashingVectorizer(decode_error='ignore', n_features =2**18, non_negative=True)),
                ('tfdif', TfidfTransformer()),
            ])),
            ('fulltext', Pipeline([
                ('selector', item_selector(key='title')),
                ('lemma', lemmatizer(dict_path=dict_path)),
                ('vect', HashingVectorizer(decode_error='ignore', n_features =2**18, non_negative=True)),
                ('tfdif', TfidfTransformer()),
            ])),
        ],

        transformer_weights = {
            'meta_data': 1,
            'title': 0.7,
        }
        )),
])
print("Done(", round(time.time() -  start_time, 0),"s)", sep='', flush=True)

print("Processing data:", flush = True)
clas_fields = ["080", "600", "610", "611", "630", "648", "008", "100",
	  "650", "651", "653", "655", "670", "678", "695", "964"]


with open(args['records_path']) as records_file,\
	 open(args['save_path'], 'w') as output_file:
	text = ''
    title = ''
	konspekt = False
	num_of_records = 0;
	for line in records_file:
		if line != '\n':
			output_file.write(line)
			try:
				field = line[:line.index(' ')]
			except:
				continue
			if field == '072':
				konspekt = True
				continue
			if field in clas_fields:
				text += line
			elif field == '245':
				title = line
		else:
			if konspekt:
				konspekt = False
				output_file.write(line)
				continue
			text = filter_text(text)
			title = filter_text(title)
			data = {}
			data['meta_data'] = [text]
			data['title'] = [title + ' ' + text]
			X = pipeline.fit_transform(data)
			predicted = clf.predict(X)[0]
			predicted_proba = clf.predict_proba(X)[0]
			all_pred = []
			for index, item in enumerate(predicted_proba):
				if item != 0:
					all_pred.append([cat_names[index], item])
			for index, item in enumerate(sorted(all_pred, key=lambda x: x[1], reverse = True)):
				if index > 6:
					break
				output_file.write('072 c $a' + item[0] + '$b' + str(item[1]) + '\n')
			output_file.write(line)
			num_of_records += 1
			text = ''
            title = ''
