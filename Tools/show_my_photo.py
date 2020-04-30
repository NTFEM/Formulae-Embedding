import pickle
import gensim
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle

import numpy as np
import pandas as pd
from DataReader.wiki_data_reader import WikiDataReader

# data_reader = WikiDataReader('NTCIR-12_MathIR_Wikipedia_Corpus/MathTagArticles', 0, 'query_file')
# query_dict = data_reader.get_query()

# for list in query_dict.items():
#     print(list)
#
# with open('dict_tree_sif/flat_addbank_formula_treesif.pickle','rb') as file:
#     formula_dict = pickle.load(file)
#
# print(formula_dict['Transcendental_number:9'])
# print(formula_dict['AB_magnitude:7'])
# print(formula_dict['LTI_system_theory:109'])
with open('/home/aragorn/mywork/tangent/vector_file/vector_dict.pickle', 'rb') as file:
    vector_dick = pickle.load(file)
print(vector_dick)
# for vect in vector_dick.items():
#     print(vect)