import gensim
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import numpy as np
import pandas as pd
from DataReader.wiki_data_reader import WikiDataReader

# data_reader = WikiDataReader('NTCIR-12_MathIR_Wikipedia_Corpus/MathTagArticles', 0, 'query_file')

# # 字典判空
# with open('dict_tree_sif/flat_addbank_formula_treesif.pickle','rb') as file:
with open('dict_tree_sif/new_opt_formula_treesif.pickle','rb') as file:
    formula_dict = pickle.load(file)
# judge_table = pd.read_table('Retrieval_Results/judge.dat',sep = " ",header=None,engine = 'python')
# count_not = 0
# count_exit = 0
#
# #存储缺失公式
# miss_formula = []
# for i in range(1202):
#     if judge_table[3][i] == 4.0:
#         count_not += 1
#         if judge_table[2][i] not in formula_dict.keys():
#             miss_formula.append(judge_table[2][i])
#             print(judge_table[2][i])
#     else:
#         count_exit += 1
# print('4.0')
# print('--------------------------')
# # print(count_exit)
# # print(count_not)
#
# for i in range(1202):
#     if judge_table[3][i] == 3.0:
#         count_not += 1
#         if judge_table[2][i] not in formula_dict.keys():
#             miss_formula.append(judge_table[2][i])
#             print(judge_table[2][i])
#     else:
#         count_exit += 1
# print('3.0')
# print('--------------------------')
# print(count_exit)
# print(count_not)

# file = open('dict_tree_sif/miss_formula.pickle', 'wb')
# pickle.dump(miss_formula, file)
# file.close()
# print(miss_formula)


# judge_table = pd.read_table('Retrieval_Results/judge.dat',sep = " ",header=None,engine = 'python')
sif_table = pd.read_table('Retrieval_Results/res_tangent_cft',sep = " ",header=None,engine='python')

for i in range(20000):
    if sif_table[0][i] == 'NTCIR12-MathWiki-4':

#判空
count_match = 0
miss_match = 0
# for i in range(1202):
#     #观测相似度分数4的公式有几个没在检索结果中
#     if judge_table[3][i] == 3.0:
#         temp_flag = 0
#         for j in range(20000):
#             if judge_table[2][i] == sif_table[2][j] and judge_table[0][i] == sif_table[0][j]:
#                 temp_flag = 1
#                 count_match += 1
#
#         #flag没变说明当前公式没有出现在res中
#         if temp_flag == 0:
#             miss_match += 1
#             if judge_table[2][i] not in formula_dict.keys():
#                 print(judge_table[0][i], end = ':')
#                 print(judge_table[2][i], end = '--')
#                 print('not fund')
#                 continue
#             print(judge_table[0][i],end=':')
#             print(judge_table[2][i],end='--')
#             print(formula_dict[judge_table[2][i]])
#             print('\n')

print("匹配到公式个数:")
print(count_match)
print(miss_match)












