#!/usr/bin/env python3
from ufal.morphodita import *
import re

# Class for text lemmatization
# by MorphoDiTa (http://ufal.mff.cuni.cz/morphodita)
class lemmatizer():

	def __init__(self, dict_path):
		self.morpho = Morpho.load(dict_path)

	def lemmatize(self, text):
		lemma_text = ''
		lemmas = TaggedLemmas()
		for word in re.sub(r'[`|_|-]', ' ', text).split(' '):
			if word and not word.isspace():
				self.morpho.analyze( word , self.morpho.GUESSER, lemmas)
				lemma_text += re.sub( r'[`|_|-].*', '' ,lemmas[0].lemma) + ' '
		return lemma_text

	# Uprava pre pouzitie v FeatureUnion
	#ide o spracovanie pola
	def fit(self, x, y=None):
		return self

	def transform(self, x, y=None):
		return [self.lemmatize(a) for a in x]
