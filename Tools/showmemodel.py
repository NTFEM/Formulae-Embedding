import gensim
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import numpy as np
import pandas as pd
from DataReader.wiki_data_reader import WikiDataReader

#
# data_reader = WikiDataReader('NTCIR-12_MathIR_Wikipedia_Corpus/MathTagArticles', 0, 'query_file')
#统计公式长度
with open('Dict/binary_formula.pickle','rb') as file:
    formula_dict = pickle.load(file)
len_sum = 0
for key in formula_dict:
    len_sum += len(formula_dict[key])
print(len_sum/len(formula_dict))


# 原始读取文本
# dictionary_formula_slt_tuple = data_reader.get_collection()
# 保存字典
# print('get_collection and saving:')
# file = open('Dict/init_formula.pickle','wb')
# pickle.dump(dictionary_formula_slt_tuple,file)
# file.close()

# # 字典判空
# with open('dict_tree_sif/flat_addbank_formula_treesif.pickle','rb') as file:
# with open('dict_tree_sif/new_opt_formula_treesif.pickle','rb') as file:
#     formula_dict = pickle.load(file)

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
# sif_table = pd.read_table('Retrieval_Results/res_tangent_cft',sep = " ",header=None,engine='python')

#判空
# count_match = 0
# miss_match = 0
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
#
# print("匹配到公式个数:")
# print(count_match)
# print(miss_match)


#参数指令
# python3 tangent_cft_front_end.py -cid 1 --t 1 --r 0 -ds 'NTCIR/add' --slt 1 -em 'temp.csv' --mp 'slt_temp
# you can judge a new finding a way that is very easy to find the other getyuio
# 检测解析树
# dictionary_formula_slt_tuple = data_reader.get_collection()
# print(dictionary_formula_slt_tuple)


#读取query
# data_reader = WikiDataReader('', 0, 'query_file')
# query_dict = data_reader.get_query()
# print(query_dict)

# kai
# with open('dict_tree_sif/new_opt_formula_treesif.pickle','rb') as file:
#     formula_dict = pickle.load(file)



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




# count_not = 0
# count_exit = 0
# for i in range(1202):
#     if user_table[3][i] == 3.0:
#         count_not += 1
#         if user_table[2][i] not in formula_dict.keys():
#             print(user_table[2][i])
#
#     else:
#         count_exit += 1
# print('--------------------------')
# print(count_exit)
# print(count_not)

# file = open('dict_tree_sif/new_opt_formula_treesif.pickle', 'wb')
# pickle.dump(formula_dict, file)
# file.close()

# temp = ['Algebra:0','Algebra:10','Brahmagupta:2','Help:Displaying_a_formula:367','Periodic_points_of_complex_quadratic_mappings:38','Quadratic_equation:128','Quadratic_function:8','Slope:10',
#         'Dinic\'s_algorithm:48','Goertzel_algorithm:70',
#         'Gabor_wavelet:1','Gabor_wavelet:6','Generation_time:5','Opacity_(optics):17','Poisson_limit_theorem:8','Binomial_regression:21','Non-analytic_smooth_function:0']
# # print(query_dict['Algebera:0'])
#
# temp_11 = ['Algebra:0','Algebra:10','Brahmagupta:2','Help:Displaying_a_formula:367','Periodic_points_of_complex_quadratic_mappings:38','Quadratic_equation:128','Quadratic_function:8','Slope:10']
# temp_12 = ['Dinic\'s_algorithm:48','Goertzel_algorithm:70']
# temp_16 = ['Gabor_wavelet:1','Gabor_wavelet:6','Generation_time:5','Opacity_(optics):17']
# temp_18 = ['Poisson_limit_theorem:8']
# temp_8 = ['Binomial_regression:21','Non-analytic_smooth_function:0']
#
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






#检验模型
# model = Doc2Vec.load('Models/slt_300_model.wv.vectors.npy')
# print(model)
# print('Sridhara:0')
# print(model.docvecs['Sridhara:0'])
# print('Quadratic:8')
# print(model.docvecs['Quadratic_function:8'])
# print('Monic_poly:1')
# print(model.docvecs['Monic_polynomial:1'])
# temp = ['V!a\tV!x\tn\t-', 'V!a\tN!2\tna\t-', 'V!a\t+\tnn\t-', 'V!x\tN!2\ta\tn', 'N!2\t0!\tn\tna', 'V!x\t+\tn\tn', 'V!x\tV!b\tnn\tn', '+\tV!b\tn\tnn', '+\tV!x\tnn\tnn', 'V!b\tV!x\tn\tnnn', 'V!b\t+\tnn\tnnn', 'V!x\t+\tn\tnnnn', 'V!x\tV!c\tnn\tnnnn', '+\tV!c\tn\tnnnnn', '+\t=\tnn\tnnnnn', 'V!c\t=\tn\t6n', 'V!c\tN!0\tnn\t6n', '=\tN!0\tn\t7n', 'N!0\t0!\tn\t8n']


# def cos_sim(vector_a,vector_b):
#     vector_a = np.mat(vector_a)
#     vector_b = np.mat(vector_b)
#     num = float(vector_a * vector_b.T)
#     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
#     cos = num/denom
#     sim =0.5 + 0.5 * cos
#     return sim
# print(cos_sim(model.docvecs['Sridhara:0'],model.infer_vector(temp)))
# print(cos_sim(model.docvecs['Sridhara:0'],model.docvecs['Periodic_continued_fraction:5']))
# print(cos_sim(model.docvecs['Periodic_continued_fraction:5'],model.docvecs['Monic_polynomial:1']))



# with open('Dict/pickle_formula.pickle', 'rb') as file:
#     a_dict1 =pickle.load(file)
#     print(a_dict1)
# print(a_dict1['Sridhara:0'])
# print(a_dict1['Periodic_continued_fraction:5'])
# # print(a_dict1['Algebra:10'])
# # print(a_dict1['Quadratic_function:8'])
# print(a_dict1['Monic_polynomial:1'])
# print(a_dict1['Loss_of_significance:2'])
# print(a_dict1['Algebraic_solution:1'])
# print(a_dict1['Quadratic_equation:128'])

