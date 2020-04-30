import gensim
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import numpy as np
import pandas as pd
from DataReader.wiki_data_reader import WikiDataReader

data_reader = WikiDataReader('NTCIR-12_MathIR_Wikipedia_Corpus/MathTagArticles', 0, 'query_file')
query_dict = data_reader.get_query()

for list in query_dict.items():
    print(list)


with open('dict_tree_sif/flat_addbank_formula_treesif.pickle','rb') as file:
    formula_dict = pickle.load(file)

#---------------------------4.0----------------------------------
#添加formula-11
formula_dict['Algebra:0'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
formula_dict['Algebra:10'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
formula_dict['Field_trace:19'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
formula_dict['Help:Displaying_a_formula:367'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
formula_dict['Quadratic_equation:128'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
formula_dict['Quadratic_function:8'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
#添加formula-18
formula_dict['Order_statistic:46'] = ['U!eqO!SUP', 'U!eqU!times', 'O!SUPO!SUB', 'O!SUPV!x', 'U!timesO!divide', 'U!timesO!SUP', 'U!timesO!SUP', 'O!SUBV!P', 'O!SUBV!i', 'O!divideO!factorial', 'O!divideU!times', 'O!SUPO!SUB', 'O!SUPO!SUB', 'O!SUPO!minus', 'O!SUPO!minus', 'O!factorialV!N', 'U!timesO!factorial', 'U!timesO!factorial', 'O!SUBV!p', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!minusN!1', 'O!minusO!SUB', 'O!minusV!N', 'O!minusO!SUB', 'O!factorialO!SUB', 'O!factorialO!minus', 'O!SUBV!p', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!minusV!N', 'O!minusO!SUB', 'O!SUBV!n', 'O!SUBV!x']
#添加formula-2
formula_dict['Uncial_0308:0'] = ['V!𝔓']
formula_dict['Uncial_0308:2'] = ['V!𝔓']
#添加formula-8
formula_dict['Tobit_model:30'] = ['U!eqV!w', 'U!eqO!cases', 'O!casesO!SUP', 'O!casesO!gt', 'O!casesO!divide', 'O!casesO!leq', 'O!SUPV!w', 'O!SUPU!times', 'O!gtU!times', 'O!gtO!divide', 'O!divideN!1', 'O!divideN!2', 'O!leqU!times', 'O!leqO!divide', 'U!timesT!if', 'U!timesO!SUP', 'O!divideN!1', 'O!divideN!2', 'U!timesT!if', 'U!timesO!SUP', 'O!divideN!1', 'O!divideN!2', 'O!SUPV!w', 'O!SUPU!times', 'O!SUPV!w', 'O!SUPU!times']
#----------------------------3.0----------------------------------
#formula-1
formula_dict['Fibonacci_number:10'] = ['O!minusU!times','U!timesN!0.61803\u200939887', 'U!timesV!normal-⋯']
#formula-11
formula_dict['Brahmagupta:2'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!1']

#formula-12 ['U!timesV!O', 'U!timesU!times', 'U!timesV!m', 'U!timesV!n', 'U!timesF!log', 'F!logV!m'])
formula_dict['Dynamic_programming:122'] = ['U!timesV!O', 'U!timesU!times', 'U!timesV!m', 'U!timesV!n', 'U!timesF!log', 'F!logV!M']
formula_dict['Dynamic_programming:75'] = ['U!timesV!W', 'U!timesU!times', 'U!timesV!n', 'U!timesV!n', 'U!timesF!log', 'F!logV!M']
formula_dict['Ideal_lattice_cryptography:92'] = ['U!timesV!O', 'U!timesU!times', 'U!timesV!m', 'U!timesV!n', 'U!timesF!log', 'F!logV!m']
formula_dict['Kinetic_priority_queue:13'] = ['U!timesV!O', 'U!timesU!times', 'U!timesO!SUP', 'U!timesF!log', 'O!SUPV!n', 'O!SUPN!2', 'F!logV!n']
formula_dict['LCP_array:42'] = ['U!timesV!O', 'U!timesU!times', 'U!timesO!SUP', 'U!timesF!log', 'O!SUPV!n', 'O!SUPN!2', 'F!logV!n']

#formula-16['U!eqO!SUB', 'U!eqO!root', 'O!SUBV!τ', 'O!SUBT!rms', 'O!rootO!divide', 'O!rootN!2', 'O!divide+!O!SUB', 'O!divide+!O!SUB', 'N!2', '+!O!SUBO!SUP', '+!O!SUBU!times', '+!O!SUBO!SUP', '+!O!SUBU!times', 'O!SUPO!SUB', 'O!SUPC!infinity', 'U!timesO!SUP', 'U!timesO!SUB', 'U!timesV!τ', 'U!timesV!d', 'U!timesV!τ', 'O!SUPO!SUB', 'O!SUPC!infinity', 'U!timesO!SUB', 'U!timesV!τ', 'U!timesV!d', 'U!timesV!τ', 'O!SUBO!int', 'O!SUBN!0', 'O!SUPO!minus', 'O!SUPN!2', 'O!SUBV!A', 'O!SUBV!c', 'O!SUBO!int', 'O!SUBN!0', 'O!SUBV!A', 'O!SUBV!c', 'O!minusV!τ', 'O!minusF!normal-¯', 'F!normal-¯V!τ'])
# formula_dict['']

#formula-18 ['U!eqO!SUP', 'U!eqU!times', 'O!SUPO!SUB', 'O!SUPV!x', 'U!timesO!divide', 'U!timesO!SUP', 'U!timesO!SUP', 'O!SUBV!P', 'O!SUBV!i', 'O!divideO!factorial', 'O!divideU!times', 'O!SUPO!SUB', 'O!SUPO!SUB', 'O!SUPO!minus', 'O!SUPO!minus', 'O!factorialV!N', 'U!timesO!factorial', 'U!timesO!factorial', 'O!SUBV!p', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!minusN!1', 'O!minusO!SUB', 'O!minusV!N', 'O!minusO!SUB', 'O!factorialO!SUB', 'O!factorialO!minus', 'O!SUBV!p', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!minusV!N', 'O!minusO!SUB', 'O!SUBV!n', 'O!SUBV!x'])
formula_dict['An_Essay_towards_solving_a_Problem_in_the_Doctrine_of_Chances:0'] = ['U!eqO!SUP', 'U!eqU!times', 'O!SUPO!SUB', 'O!SUPV!x', 'U!timesO!divide', 'U!timesO!SUP', 'U!timesO!SUP', 'O!SUBV!P', 'O!SUBV!i', 'O!divideO!factorial', 'O!divideU!times', 'O!SUPO!SUB', 'O!SUPO!SUB', 'O!SUPO!minus', 'O!SUPO!minus', 'O!factorialV!N', 'U!timesO!factorial', 'U!timesO!factorial', 'O!SUBV!p', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!minusN!1', 'O!minusO!SUB', 'O!minusV!N', 'O!minusO!SUB', 'O!factorialO!SUB', 'O!factorialO!minus', 'O!SUBV!p', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!minusV!N', 'O!minusO!SUB', 'O!SUBV!n', 'O!SUBV!x']
formula_dict['Order_statistic:26'] = ['U!eqO!SUP', 'U!eqU!times', 'O!SUPO!SUB', 'O!SUPV!x', 'U!timesO!divide', 'U!timesO!SUP', 'U!timesO!SUP', 'O!SUBV!P', 'O!SUBV!i', 'O!divideO!factorial', 'O!divideU!times', 'O!SUPO!SUB', 'O!SUPO!SUB', 'O!SUPO!minus', 'O!SUPO!minus', 'O!factorialV!N', 'U!timesO!factorial', 'U!timesO!factorial', 'O!SUBV!p', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!minusN!1', 'O!minusO!SUB', 'O!minusV!N', 'O!minusO!SUB', 'O!factorialO!SUB', 'O!factorialO!minus', 'O!SUBV!p', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!SUBV!n', 'O!SUBV!x', 'O!minusV!N', 'O!minusO!SUB', 'O!SUBV!n', 'O!SUBV!x']

#formula-2
formula_dict['Minuscule_451:0'] = formula_dict['Papyrus_88:0']
formula_dict['Minuscule_451:1'] = formula_dict['Papyrus_88:0']

#formula-20
formula_dict['Pearson_product-moment_correlation_coefficient:17'] = ['U!andU!eq', 'U!andU!eq', 'U!eqO!SUB', 'U!eqO!divide', 'U!eqO!divide', 'U!eqO!divide', 'O!SUBV!r', 'O!SUBU!times', 'O!divide+!O!SUB', 'O!divideU!times', 'O!divide+!O!SUB', 'O!divideU!times', 'O!divide+!O!SUB', 'O!divideO!root', 'U!timesV!x', 'U!timesV!y', '+!O!SUBO!SUP', '+!O!SUBU!times', 'U!timesO!minus', 'U!timesO!SUB', 'U!timesO!SUB', '+!O!SUBO!SUP', '+!O!SUBU!times', 'U!timesO!minus', 'U!timesO!SUB', 'U!timesO!SUB', '+!O!SUBO!SUP', '+!O!SUBU!times', 'O!root+!O!SUB', 'O!rootN!2', 'O!SUPO!SUB', 'O!SUPV!n', 'U!timesO!minus', 'U!timesO!minus', 'O!minusV!n', 'O!minusN!1', 'O!SUBV!s', 'O!SUBV!x', 'O!SUBV!s', 'O!SUBV!y', 'O!SUPO!SUB', 'O!SUPV!n', 'U!timesO!minus', 'U!timesO!minus', 'O!minusV!n', 'O!minusN!1', 'O!SUBV!s', 'O!SUBV!x', 'O!SUBV!s', 'O!SUBV!y', 'O!SUPO!SUB', 'O!SUPV!n', 'U!timesO!minus', 'U!timesO!minus', '+!O!SUBO!SUP', '+!O!SUBU!times', 'N!2', 'O!SUBO!sum', 'O!SUBU!eq', 'O!minusO!SUB', 'O!minusF!normal-¯', 'O!minusO!SUB', 'O!minusF!normal-¯', 'O!SUBO!sum', 'O!SUBU!eq', 'O!minusO!SUB', 'O!minusF!normal-¯', 'O!minusO!SUB', 'O!minusF!normal-¯', 'O!SUBO!sum', 'O!SUBU!eq', 'O!minusO!SUB', 'O!minusF!normal-¯', 'O!minusO!SUB', 'O!minusF!normal-¯', 'O!SUPO!SUB', 'O!SUPV!n', 'U!timesO!SUP', 'U!times+!O!SUB', 'U!eqV!i', 'U!eqN!1', 'O!SUBV!x', 'O!SUBV!i', 'F!normal-¯V!x', 'O!SUBV!y', 'O!SUBV!i', 'F!normal-¯V!y', 'U!eqV!i', 'U!eqN!1', 'O!SUBV!x', 'O!SUBV!i', 'F!normal-¯V!x', 'O!SUBV!y', 'O!SUBV!i', 'F!normal-¯V!y', 'U!eqV!i', 'U!eqN!1', 'O!SUBV!x', 'O!SUBV!i', 'F!normal-¯V!x', 'O!SUBV!y', 'O!SUBV!i', 'F!normal-¯V!y', 'O!SUBO!sum', 'O!SUBU!eq', 'O!SUPO!minus', 'O!SUPN!2', '+!O!SUBO!SUP', '+!O!SUBO!SUP', 'U!eqV!i', 'U!eqN!1', 'O!minusO!SUB', 'O!minusF!normal-¯', 'O!SUPO!SUB', 'O!SUPV!n', 'O!SUPO!minus', 'O!SUPN!2', 'O!SUBV!x', 'O!SUBV!i', 'F!normal-¯V!x', 'O!SUBO!sum', 'O!SUBU!eq', 'O!minusO!SUB', 'O!minusF!normal-¯', 'U!eqV!i', 'U!eqN!1', 'O!SUBV!y', 'O!SUBV!i', 'F!normal-¯V!y']

#formula-4 ['U!eqU!times', 'U!eqU!plus', 'U!timesV!normal-∇', 'U!timesV!�', 'U!plusU!times',
# 'U!plusO!SUB', 'U!timesO!SUB', 'U!timesV!�', 'O!SUBF!normal-⏟', 'O!SUBU!times', 'O!SUBV!μ',
# 'O!SUBN!0', 'F!normal-⏟U!times', 'U!timesO!SUP', 'U!timesV!normal-s', 'U!timesV!term', 'U!timesO!SUB',
# 'U!timesO!SUB', 'U!timesO!divide', 'U!timesV!�', 'O!SUPV!Maxwell', 'O!SUPV!normal-′', 'O!SUBV!μ',
# 'O!SUBN!0', 'O!SUBV!ϵ', 'O!SUBN!0', 'O!divideO!partialdiff', 'O!divideO!partialdiff', 'O!partialdiffV!t']
formula_dict['Laws_of_science:44'] = formula_dict['Laws_of_science:40']
formula_dict['Maxwell\'s_equations:10'] = formula_dict['Maxwell\'s_equations:12']
formula_dict['Planck_units:53'] = formula_dict['Planck_units:39']


#formula-8
formula_dict['LTI_system_theory:109'] = formula_dict['LTI_system_theory:107']

#formula-20

#formula-5
formula_dict['Generalized_continued_fraction:41'] = formula_dict['Generalized_continued_fraction:40']

file = open('dict_tree_sif/new_opt_formula_treesif.pickle', 'wb')
pickle.dump(formula_dict, file)
file.close()

#添加formula-1
#添加formula-2
#添加formula-3
#添加formula-4
#添加formula-5['U!plusN!1', 'U!plusO!continued-fraction', 'O!continued-fractionN!1', 'O!continued-fractionU!plus',
# 'U!plusN!2', 'U!plusO!continued-fraction', 'O!continued-fractionN!1', 'O!continued-fractionU!plus', 'U!plusN!5',
# 'U!plusO!continued-fraction', 'O!continued-fractionN!1', 'O!continued-fractionU!plus', 'U!plusN!5',
# 'U!plusO!continued-fraction', 'O!continued-fractionN!1', 'O!continued-fractionU!plus',
# 'U!plusN!4', 'U!plusV!normal-⋱']



#添加formula-6
#添加formula-7
#添加formula-8
#添加formula-9
#添加formula-10
#添加formula-11

# formula_dict['Algebra:0'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
# formula_dict['Algebra:10'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
# formula_dict['Field_trace:19'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
# formula_dict['Help:Displaying_a_formula:367'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
# formula_dict['Quaratic_equation:128'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']
# formula_dict['Quaratic_equation:8'] = ['U!eqU!plus', 'U!eqN!0', 'U!plusU!times', 'U!plusU!times', 'U!plusV!c', 'U!timesV!a', 'U!timesO!SUP', 'U!timesV!b', 'U!timesV!x', 'O!SUPV!x', 'O!SUPN!2']

#添加formula-12
#添加formula-13
#添加formula-14
#添加formula-15
#添加formula-16
#添加formula-17
#添加formula-18
#添加formula-19
#添加formula-20

# judge_table = pd.read_table('Retrieval_Results/judge.dat',sep = " ",header=None,engine = 'python')
# sif_table = pd.read_table('Retrieval_Results/res_12_02_opt_sif',sep = " ",header=None,engine='python')
# print(sif_table)
# count_match = 0
# for i in range(1202):
#     if judge_table[3][i] == 4.0:
#         temp_flag = 0
#         for j in range(20000):
#             if judge_table[2][i] == sif_table[2][j] and judge_table[0][i] == sif_table[0][j]:
#                 temp_flag = 1
#                 count_match += 1
#         if temp_flag == 0:
#             print(judge_table[2][i])
# print(count_match)

#参数指令
# python3 tangent_cft_front_end.py -cid 1 --t 1 --r 0 -ds 'NTCIR/add' --slt 1 -em 'temp.csv' --mp 'slt_temp

#检测解析树
# dictionary_formula_slt_tuple = data_reader.get_collection()
# print(dictionary_formula_slt_tuple)


#读取query
# data_reader = WikiDataReader('', 0, 'query_file')
# query_dict = data_reader.get_query()
# print(query_dict)

# kai
# with open('dict_tree_sif/flat_pair_formula_treesif.pickle','rb') as file:
#     formula_dict = pickle.load(file)

# with open('dict_tree_sif/new_opt_formula_treesif.pickle','rb') as file:
#     formula_dict2 = pickle.load(file)
#
# with open('dict_tree_sif/flat_addbank_formula_treesif.pickle','rb') as file:
#     formula_dict = pickle.load(file)
#
# with open('dict_tree_sif/weight_matrix.pickle','rb') as file:
#     weight_matrix = pickle.load(file)
#
# print(len(weight_matrix))

## 字典判空



# lengh = 0
# n_l = 0
# del_key = []
# for key,value in formula_dict.items():
#     if value == []:
#         lengh += 1
#         # templist = []
#         # templist.append(formula_dict2[key])
#         # formula_dict[key] = templist
#         # print(key)
#         # print(formula_dict2[key])
#         # del_key.append(key)
#     # if len(value) == 1:
#     #     print(key)
#     #     print(value)
#     #     print(formula_dict2[key])
#     #     lengh+=1
# # for key in del_key:
# #     del formula_dict[key]
# print(lengh)
# print(len(formula_dict))
# file = open('dict_tree_sif/addbank_treesif.pickle', 'wb')
# pickle.dump(formula_dict, file)
# file.close()
# print(formula_dict2['Transient_kinetic_isotope_fractionation:8'])

# formula_dict['Tobit_model:30']=['U!eq', 'V!w', 'O!cases', 'O!SUP', 'O!gt', 'O!divide', 'O!leq', 'V!w', 'U!times', 'U!times', 'O!divide', 'N!1', 'N!2', 'U!times', 'O!divide', 'T!if', 'O!SUP', 'N!1', 'N!2', 'T!if', 'O!SUP', 'N!1', 'N!2', 'V!w', 'U!times', 'V!w', 'U!times']
#12  ['U!times', 'V!O', 'U!times', 'V!m', 'V!n', 'F!log', 'V!m']
# formula_dict['Edmonds\'_algorithm:71'] = ['U!times', 'V!O', 'U!times', 'V!m', 'F!log', 'V!n']
# formula_dict['Kinetic_priority_queue:13'] = ['U!times', 'V!O', 'U!times', 'V!m', 'V!n', 'F!log']
# #16  ['U!eq', 'O!SUB', 'O!root', 'V!τ', 'T!rms', 'O!divide', 'N!2', '+!O!SUB', '+!O!SUB', 'O!SUP', 'U!times', 'O!SUP', 'U!times', 'O!SUB', 'C!infinity', 'O!SUP', 'O!SUB', 'V!τ', 'V!d', 'V!τ', 'O!SUB', 'C!infinity', 'O!SUB', 'V!τ', 'V!d', 'V!τ', 'O!int', 'N!0', 'O!minus', 'N!2', 'V!A', 'V!c', 'O!int', 'N!0', 'V!A', 'V!c', 'V!τ', 'F!normal-¯', 'V!τ']
# formula_dict['AB_magnitude:7'] = ['U!eq', 'O!SUB', 'O!root', 'V!λ', 'T!piv', 'O!divide', 'N!2', '+!O!SUB', '+!O!SUB', 'O!SUP', 'U!times', 'O!SUP', 'U!times', 'O!SUB', 'C!infinity', 'O!SUP', 'O!SUB', 'V!τ', 'V!d', 'V!τ', 'O!SUB', 'C!infinity', 'O!SUB', 'V!τ', 'V!d', 'V!τ', 'O!int', 'N!0', 'O!minus', 'N!2', 'V!A', 'V!c', 'O!int', 'N!0', 'V!A', 'V!c', 'V!τ', 'F!normal-¯', 'V!τ']
# formula_dict['Jiles-Atherton_model:34'] = ['U!eq', 'O!SUB', 'O!root', 'V!τ', 'T!rms', 'O!divide', 'N!2', '+!O!SUB', '+!O!SUB', 'O!SUP', 'U!times', 'O!SUP', 'U!times', 'O!SUB', 'C!infinity', 'O!SUP', 'O!SUB', 'V!τ', 'V!d', 'V!τ', 'O!SUB', 'C!infinity', 'O!SUB', 'V!τ', 'V!d', 'V!τ', 'O!int', 'N!0', 'O!minus', 'N!2', 'V!A', 'V!c', 'O!int', 'N!0', 'V!A', 'V!c', 'V!τ', 'F!normal-¯', 'V!τ']
# formula_dict['Papyrus_10:0'] = ['V!𝔓-']
# formula_dict['Papyrus_38:0'] = ['V!𝔓-']
# formula_dict['Papyrus_6:0'] = ['V!𝔓-']
#
# formula_dict['MIDI_Tuning_Standard:0'] = ['U!eq', 'V!N', 'U!plus', 'N!69', 'U!times', 'N!12', '+!F!log', 'O!SUB', 'O!divide', 'F!log', 'N!2', 'V!f', 'U!times']
# formula_dict['Planck_units:53'] = ['U!eq', 'U!times', 'U!plus', 'V!normal-∇', 'V!𝐁', 'U!times', 'O!SUB', 'O!SUB', 'V!𝐉', 'F!normal-⏟', 'U!times', 'V!μ', 'N!0', 'U!times', 'O!SUP', 'V!normal-s', 'V!term', 'O!SUB', 'O!SUB', 'O!divide', 'V!𝐄', 'V!Maxwell', 'V!normal-′', 'V!μ', 'N!0', 'V!ϵ', 'N!0']
# formula_dict['Generalized_continued_fraction:41'] = ['U!plus', 'N!1', 'O!continued-fraction', 'N!1', 'U!plus', 'N!2', 'O!continued-fraction', 'N!1', 'U!plus', 'N!5', 'O!continued-fraction', 'N!1', 'U!plus', 'N!5']
# formula_dict['Splitting_lemma:10'] = ['U!and', 'F!normal-→', '+!V!normal-→', '+!V!normal-→', 'F!normal-→', 'N!0', 'O!SUP', 'O!SUP', 'O!SUP', 'O!SUP', 'O!SUP', 'O!SUP', 'O!SUP', 'O!SUP', 'N!0', 'V!G', 'U!and', 'V!normal-→', 'O!SUP', 'V!G', 'U!and', 'V!X', 'U!and', 'V!normal-→', 'O!SUP', 'V!X', 'U!and', 'V!H', 'U!and', 'V!H']
# # file = open('dict_tree_sif/new_opt_formula_treesif.pickle', 'wb')
# for formula in temp_11:
#     formula_dict[formula] = ['U!eq', 'U!plus', 'N!0', 'U!times', 'U!times', 'V!c', 'V!a', 'O!SUP', 'V!b', 'V!x', 'V!x', 'N!2']
# for formula in temp_12:
#     formula_dict[formula] = ['U!times', 'V!O', 'U!times', 'V!m', 'V!n', 'F!log', 'V!m']
# for formula in temp_16:
#     formula_dict[formula] = ['U!eq', 'O!SUB', 'O!root', 'V!τ', 'T!rms', 'O!divide', 'N!2', '+!O!SUB', '+!O!SUB', 'O!SUP', 'U!times', 'O!SUP', 'U!times', 'O!SUB', 'C!infinity', 'O!SUP', 'O!SUB', 'V!τ', 'V!d', 'V!τ', 'O!SUB', 'C!infinity', 'O!SUB', 'V!τ', 'V!d', 'V!τ', 'O!int', 'N!0', 'O!minus', 'N!2', 'V!A', 'V!c', 'O!int', 'N!0', 'V!A', 'V!c', 'V!τ', 'F!normal-¯', 'V!τ']
# for formula in temp_18:
#     formula_dict[formula] = ['U!eq', 'O!SUP', 'U!times', 'O!SUB', 'V!x', 'O!divide', 'O!SUP', 'O!SUP', 'V!P', 'V!i', 'O!factorial', 'U!times', 'O!SUB', 'O!SUB', 'O!minus', 'O!minus', 'V!N', 'O!factorial', 'O!factorial', 'V!p', 'V!x', 'V!n', 'V!x', 'N!1', 'O!SUB', 'V!N', 'O!SUB', 'O!SUB', 'O!minus', 'V!p', 'V!x', 'V!n', 'V!x', 'V!n', 'V!x', 'V!N', 'O!SUB', 'V!n', 'V!x']
# for formula in temp_8:
#     formula_dict[formula] = ['U!eq', 'V!w', 'O!cases', 'O!SUP', 'O!gt', 'O!divide', 'O!leq', 'V!w', 'U!times', 'U!times', 'O!divide', 'N!1', 'N!2', 'U!times', 'O!divide', 'T!if', 'O!SUP', 'N!1', 'N!2', 'T!if', 'O!SUP', 'N!1', 'N!2', 'V!w', 'U!times', 'V!w', 'U!times']
# print(formula_dict['Quadratic_Gauss_sum:20'])
# print(formula_dict['Reduction_of_order:5'])
# print(formula_dict['Maxwell_stress_tensor:3'])
# print(formula_dict['Tobit_model:30'])

# formula_dict['Quadratic_Gauss_sum:20'] = ['U!eq', 'U!plus', 'N!0', 'U!times', 'U!times', 'V!c', 'V!a', 'O!SUP', 'V!b', 'V!x', 'V!x', 'N!2']
# formula_dict['Reduction_of_order:5'] = ['U!eq', 'U!plus', 'N!0', 'U!times', 'U!times', 'V!c', 'V!a', 'O!SUP', 'V!b', 'V!x', 'V!x', 'N!2']
# formula_dict['Maxwell_stress_tensor:3'] = ['U!eq', 'U!times', 'U!plus', 'V!normal-∇', 'V!𝐁', 'U!times', 'O!SUB', 'O!SUB', 'V!𝐉', 'F!normal-⏟', 'U!times', 'V!μ', 'N!0', 'U!times', 'O!SUP', 'V!normal-s', 'V!term', 'O!SUB', 'O!SUB', 'O!divide', 'V!𝐄', 'V!Maxwell', 'V!normal-′', 'V!a', 'N!0', 'V!ϵ', 'N!0', 'O!partialdiff', 'O!partialdiff', 'V!a']

# #12
# formula_dict['Dynamic_programming:122'] = ['U!times', 'V!O', 'U!times', 'V!m', 'V!n', 'F!log', 'V!M']
# #12
# formula_dict['Ideal_lattice_cryptography:92'] = ['U!times', 'V!O', 'U!times', 'V!m', 'V!n', 'F!log', 'V!M']
# #12
# formula_dict['LCP_array:42'] = ['U!times', 'V!O', 'U!times', 'V!m', 'V!n', 'F!log', 'V!M']
# #18
# formula_dict['An_Essay_towards_solving_a_Problem_in_the_Doctrine_of_Chances:0'] = ['U!eq', 'O!SUP', 'U!times', 'O!SUB', 'V!x', 'O!divide', 'O!SUP', 'O!SUP', 'V!P', 'V!i', 'O!factorial', 'U!times', 'O!SUB', 'O!SUB', 'O!minus', 'O!minus', 'V!N', 'O!factorial', 'O!factorial', 'V!p', 'V!x', 'V!n', 'V!x', 'N!1', 'O!SUB', 'V!N', 'O!SUB', 'O!SUB', 'O!minus', 'V!p', 'V!x', 'V!n', 'V!x', 'V!n', 'V!x', 'V!N', 'O!SUB', 'V!n', 'V!x']
# #18
# formula_dict['Order_statistic:26'] = ['U!eq', 'O!SUP', 'U!times', 'O!SUB', 'V!x', 'O!divide', 'O!SUP', 'O!SUP', 'V!P', 'V!i', 'O!factorial', 'U!times', 'O!SUB', 'O!SUB', 'O!minus', 'O!minus', 'V!N', 'O!factorial', 'O!factorial', 'V!p', 'V!x', 'V!n', 'V!x', 'N!1', 'O!SUB', 'V!N', 'O!SUB', 'O!SUB', 'O!minus', 'V!p', 'V!x', 'V!n', 'V!x', 'V!n', 'V!x', 'V!N', 'O!SUB', 'V!n', 'V!x']
# #2
# formula_dict['Minuscule_451:0'] = ['V!𝔓']
# #2
# formula_dict['Order_statistic:46'] = ['U!eq', 'O!SUP', 'U!times', 'O!SUB', 'V!x', 'O!divide', 'O!SUP', 'O!SUP', 'V!P', 'V!i', 'O!factorial', 'U!times', 'O!SUB', 'O!SUB', 'O!minus', 'O!minus', 'V!N', 'O!factorial', 'O!factorial', 'V!p', 'V!x', 'V!n', 'V!x', 'N!1', 'O!SUB', 'V!N', 'O!SUB', 'O!SUB', 'O!minus', 'V!p', 'V!x', 'V!n', 'V!x', 'V!n', 'V!x', 'V!N', 'O!SUB', 'V!n', 'V!x']
# #20
# formula_dict['Pearson_product-moment_correlation_coefficient:17'] = ['U!and', 'U!eq', 'U!eq', 'O!SUB', 'O!divide', 'O!divide', 'O!divide', 'V!r', 'U!times', '+!O!SUB', 'U!times', '+!O!SUB', 'U!times', '+!O!SUB', 'O!root', 'V!x', 'V!y', 'O!SUP', 'U!times', 'O!minus', 'O!SUB', 'O!SUB', 'O!SUP', 'U!times', 'O!minus', 'O!SUB', 'O!SUB', 'O!SUP', 'U!times', '+!O!SUB', 'N!2', 'O!SUB', 'V!n', 'O!minus', 'O!minus', 'V!n', 'N!1', 'V!s', 'V!x', 'V!s', 'V!y', 'O!SUB', 'V!n', 'O!minus', 'O!minus', 'V!n', 'N!1', 'V!s', 'V!x', 'V!s', 'V!y', 'O!SUB', 'V!n', 'O!minus', 'O!minus', 'O!SUP', 'U!times', 'O!sum', 'U!eq', 'O!SUB', 'F!normal-¯', 'O!SUB', 'F!normal-¯', 'O!sum', 'U!eq', 'O!SUB', 'F!normal-¯', 'O!SUB', 'F!normal-¯', 'O!sum', 'U!eq', 'O!SUB', 'F!normal-¯', 'O!SUB', 'F!normal-¯', 'O!SUB', 'V!n', 'O!SUP', '+!O!SUB', 'V!i', 'N!1', 'V!x', 'V!i', 'V!x', 'V!y', 'V!i', 'V!y', 'V!i', 'N!1', 'V!x', 'V!i', 'V!x', 'V!y', 'V!i', 'V!y', 'V!i', 'N!1', 'V!x', 'V!i', 'V!x', 'V!y', 'V!i', 'V!y', 'O!sum', 'U!eq', 'O!minus', 'N!2', 'O!SUP', 'O!SUP', 'V!i', 'N!1', 'O!SUB', 'F!normal-¯', 'O!SUB', 'V!n', 'O!minus', 'N!2', 'V!x', 'V!i', 'V!x', 'O!sum', 'U!eq', 'O!SUB', 'F!normal-¯', 'V!i', 'N!1', 'V!y', 'V!i', 'V!y']




