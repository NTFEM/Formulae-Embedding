import gensim
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import numpy as np
# from DataReader.wiki_data_reader import WikiDataReader

# data_reader = WikiDataReader('NTCIR/add', 0, 'query_file')

# with open('dict_tree_sif/dict_formula.pickle','rb') as file:
#     formula_dict = pickle.load(file)
# print(len(formula_dict))
# print(formula_dict)
#展开
def flat(l):
    for k in l:
        if not isinstance(k,(list,tuple)):
            yield k
        else:
            yield from flat(k)
# for key,value in formula_dict.items():
#     formula_dict[key] = list(flat(formula_dict[key]))
# print(len(formula_dict))
# file = open('dict_tree_sif/opt_formula_treesif.pickle','wb')
# pickle.dump(formula_dict,file)
# file.close()
# print(formula_dict)
# test_dict =  '[1,0[2,0[4],1[5],2[6]],1[3]]'
#
# test_dict = {}
#
#
#
# print(test_dict)
# print(formula_dict)


#参数指令
# python3 tangent_cft_front_end.py -cid 1 --t 1 --r 0 -ds 'NTCIR/add' --slt 1 -em 'temp.csv' --mp 'slt_temp

#检测解析树
# dictionary_formula_slt_tuple = data_reader.get_collection()
# print(dictionary_formula_slt_tuple)


#读取query
# data_reader = WikiDataReader('', 0, 'query_file')
# dict = data_reader.get_query()
# print(dict)

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

