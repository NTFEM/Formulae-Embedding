import logging
import os
import numpy
from Configuration.configuration import Configuration
from NTFEM_model import NTFEM_base
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
from Tools.counter import get_frequency
from sklearn.decomposition import PCA
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()


class NTFEM_modeltotrain:
    def __init__(self, model_file_path=None):
        #创建cftmodel对象 这是最后一层
        self.model = NTFEM_base()
        #是否需要加载模型
        if model_file_path is not None:
            print("Loading the model")
            #---------------fasttext-----------
            print('fasttext loading')
            self.model.load_model(model_file_path)
            #---------------tfidf--------------
            # print('tfidf loading')
            # self.model.tfidf_load_model(model_file_path)
            #---------------lda----------------
            # print('lda loading')
            # self.model.lda_load_model(model_file_path)

    def train_doc2vec_model(self,doc2vecdocuments):
        print('module training doc2vec')
        self.model.train_doc2vec(doc2vecdocuments)
        return self.model

    def train_model(self, configuration, lst_lst_encoded_tuples):
        """
        这里传入了encode之后的公式 以及配置文件
        :param configuration:
        :param lst_lst_encoded_tuples:
        :return:
        """
        print("Setting Configuration")
        #传入配置文件，encode分解之后的公式，进行训练
        self.model.train(configuration, lst_lst_encoded_tuples)
        # print('train tfidf')
        # self.model.train_Tfid(configuration, lst_lst_encoded_tuples)

        return self.model

    def save_model(self, model_file_path):
        print('show model file path')
        print(model_file_path)
        self.model.save_model(model_file_path)

    #doc2vec collection
    def doc2vec_index_collection(self,dictionary_formula_lst_encoded_tuples):
        numpy_lst = []
        index_formula_id = {}
        idx = 0
        #获取doc2vec向量
        self.d2vmodel = Doc2Vec.load('Models/doc2vec')
        for formula in dictionary_formula_lst_encoded_tuples:
            # print('formula:')
            # print(formula[1][0])
            numpy_lst.append(self.d2vmodel.docvecs[formula[1][0]].reshape(1,300))
            index_formula_id[idx] = formula[1][0]
            idx+=1
        temp = numpy.concatenate(numpy_lst, axis=0)
        # tensor_values = Variable(torch.tensor(temp).double()).cuda()
        tensor_values = Variable(torch.tensor(temp).double())

        return tensor_values, index_formula_id

    #tfidf_collection
    def Tfidf_index_collection(self,dictionary_formula_lst_encoded_tuples):
        numpy_lst = []
        index_formula_id = {}
        idx = 0
        # print('show dic len')
        # print(dictionary_formula_lst_encoded_tuples)
        print('show temp---------------------------')
        formula_2_list = list(dictionary_formula_lst_encoded_tuples.values())
        # print('len')
        # print(formula_2_list)
        self.dictionary = corpora.Dictionary(formula_2_list)
        print('corpus转换为doc2bow')
        # corpus = [self.dictionary.doc2bow(text) for text in formula_2_list]

        #进行 formula index   直接将模型的向量提取出来，不要一个个去get

        for formula in dictionary_formula_lst_encoded_tuples:
            print('(dictionary_formula_lst_encoded_tuples[formula])')
            print(dictionary_formula_lst_encoded_tuples[formula])
            test_formula = self.dictionary.doc2bow(dictionary_formula_lst_encoded_tuples[formula])
            # print('test formula doc')
            # print(test_formula)
            numpy_lst.append(self.__get_tfidf_vector_represntation(test_formula))
            index_formula_id[idx] = formula
            idx+=1
        #计算tfidf向量
        # for formula in corpus:
        #     print(formula)
        #     numpy_lst.append(self.__get_tfidf_vector_represntation(corpus[formula]))

        temp = numpy.concatenate(numpy_lst, axis=0)
        # tensor_values = Variable(torch.tensor(temp).double()).cuda()
        tensor_values = Variable(torch.tensor(temp).double())
        return tensor_values, index_formula_id

    #fasttext collection
    def index_collection(self, dictionary_formula_lst_encoded_tuples):
        numpy_lst = []
        index_formula_id = {}
        idx = 0

        for formula in dictionary_formula_lst_encoded_tuples:
            numpy_lst.append(self.__get_vector_representation(dictionary_formula_lst_encoded_tuples[formula]))
            index_formula_id[idx] = formula
            idx+=1

        temp = numpy.concatenate(numpy_lst, axis=0)
        # tensor_values = Variable(torch.tensor(temp).double()).cuda()
        tensor_values = Variable(torch.tensor(temp).double())

        return tensor_values, index_formula_id

    #SIF query
    def get_SIF_query_vector(self,dictionary_formula,lst_encoded_tuples,tensor_values,index_formula_id):
        temp = '0'
        for key,value in dictionary_formula.items():
            if value == lst_encoded_tuples:
                # print('get query')
                temp = key
                break

        for key,value in index_formula_id.items():
            if value == temp:
                # print('tensor key')
                # print(tensor_values[key])
                return tensor_values[key]


    #分类获取
    def get_query_vector(self, lst_encoded_tuples,model_type,dictionary_formula_lst_encoded_tuples):
        # #sif
        # if model_type == 3:
        #     print('sif')
        #     return

        #doc2vec
        if model_type == 2:
            # print(self.d2vmodel.infer_vector(lst_encoded_tuples).reshape(1,300))
            print('query down:')
            print(lst_encoded_tuples)
            return self.d2vmodel.infer_vector(lst_encoded_tuples,steps=20, alpha=0.025).reshape(1,300)

        #fasttext type
        if model_type == 0:
            return self.__get_vector_representation(lst_encoded_tuples)

        #tfidf_type
        if model_type == 1:
            # print('list(lst_encoded_tuples)')
            # print(list(lst_encoded_tuples))
            formula_2_list = list(dictionary_formula_lst_encoded_tuples.values())
            self.dictionary = corpora.Dictionary(formula_2_list)
            tuplus_list = []
            tuplus_list.append(lst_encoded_tuples)
            self.dictionary.add_documents(tuplus_list)

            test_formula = self.dictionary.doc2bow(lst_encoded_tuples)
            # print('get lst encoded tuples')
            # print(lst_encoded_tuples)
            print('get_query test_formula')
            print(test_formula)
            return self.__get_tfidf_vector_represntation(test_formula)

    @staticmethod
    def formula_retrieval(collection_tensor, formula_index, query_vector):
        # query_vec = torch.from_numpy(query_vector)
        query_vec = query_vector.reshape(1,300)
        # print(query_vec.shape)
        # print(query_vec)
        # query_vec = query_vector
        dist = F.cosine_similarity(collection_tensor, query_vec)
        index_sorted = torch.sort(dist, descending=True)[1]
        top_1000 = index_sorted[:1000]
        top_1000 = top_1000.data.cpu().numpy()
        cos_values = torch.sort(dist, descending=True)[0][:1000].data.cpu().numpy()
        result = {}
        count = 1
        for x in top_1000:
            doc_id = formula_index[x]
            score = cos_values[count - 1]
            result[doc_id] = score
            count += 1
        return result

    #get tfidf formula vector
    def __get_tfidf_vector_represntation(self,lst_encoded_tuples):
        # temp_vector = None
        # first = True
        # counter = 0
        temp_vector = self.model.get_tfidf_vector_representation(lst_encoded_tuples)
        return temp_vector

    #get fasttext formula vector
    def __get_vector_representation(self, lst_encoded_tuples):
        """
         This method take the converted-tuple formula file path (the file on which a list the converted tuples for
         formula is saved, then it get vector representation of each of the tuple. The formula vector is the average of its
         tuples vectors.
        :param lst_encoded_tuples: averaging vector representation for these tuples
        :return: vector representation for the formulaf
        """
        temp_vector = None
        first = True
        counter = 0
        for encoded_tuple in lst_encoded_tuples:
            # if the tuple vector cannot be extracted due to unseen n-gram, then we pass over that tuple.
            try:
                if first:
                    temp_vector = self.model.get_vector_representation(encoded_tuple)
                    first = False
                else:
                    temp_vector = temp_vector + self.model.get_vector_representation(encoded_tuple)
                counter = counter + 1
            except Exception as e:
                logging.exception(e)
        return (temp_vector / counter).reshape(1, 300)

    #获取SIF 向量
    def SIF_index_collection(self, dictionary_formula_lst_encoded_tuples):
        numpy_lst = []
        index_formula_id = {}
        idx = 0
        formula_SIF = []
        #获取词频统计
        self.freq = get_frequency(dictionary_formula_lst_encoded_tuples)
        #对每条formula进行SIF计算
        #--------------------------------------------
        for formula in dictionary_formula_lst_encoded_tuples:
            numpy_lst.append(self.__get_SIF_vector_representation(dictionary_formula_lst_encoded_tuples[formula]))
            index_formula_id[idx] = formula
            idx += 1
        print(np.array(numpy_lst).shape)
        #计算主成分
        pca = PCA()
        pca.fit(np.array(numpy_lst).reshape(len(dictionary_formula_lst_encoded_tuples),300))
        print('show pca')
        print(np.array(numpy_lst).reshape(len(dictionary_formula_lst_encoded_tuples),300).shape)
        #返回最大主成分
        u = pca.components_[0]
        #投射矩阵
        u = np.multiply(u,np.transpose(u))
        if len(u) < 300:
            for i in range(300 - len(u)):
                u = np.append(u, 0)
        #计算最终vec
        formula_vector = []
        for vs in numpy_lst:
            sub = np.multiply(u,vs)
            formula_vector.append(np.subtract(vs,sub))

        temp = numpy.concatenate(formula_vector, axis=0)
        # temp = numpy.concatenate(numpy_lst, axis=0)
        # tensor_values = Variable(torch.tensor(temp).double()).cuda()
        tensor_values = Variable(torch.tensor(temp).double())
        return tensor_values, index_formula_id

    # SIF formula vector
    def __get_SIF_vector_representation(self, lst_encoded_tuples):
        formula_SIF = []
        a_par: float = 1e-1
        vs = np.zeros(300)
        for encoded_tuple in lst_encoded_tuples:
            a_value = a_par / (a_par + self.freq[encoded_tuple])  # SIF
            vs = np.add(vs, np.multiply(a_value, self.model.get_vector_representation(encoded_tuple)))  # vs += sif * wordvector
        vs = np.divide(vs,len(lst_encoded_tuples))

        # print('show vs size')
        # print(vs.shape)
        return vs.reshape(1,300)
        # #---------------------------------
        # temp_vector = None
        # first = True
        # counter = 0
        # for encoded_tuple in lst_encoded_tuples:
        #     # if the tuple vector cannot be extracted due to unseen n-gram, then we pass over that tuple.
        #     try:
        #         if first:
        #             temp_vector = self.model.get_vector_representation(encoded_tuple)
        #             first = False
        #         else:
        #             temp_vector = temp_vector + self.model.get_vector_representation(encoded_tuple)
        #         counter = counter + 1
        #     except Exception as e:
        #         logging.exception(e)
        # return (temp_vector / counter).reshape(1, 300)
    def new_SIF_index_collection(self, dictionary_formula_lst_encoded_tuples,weight_matrix):
        numpy_lst = []
        index_formula_id = {}
        idx = 0
        formula_SIF = []
        #获取词频统计
        self.freq = get_frequency(dictionary_formula_lst_encoded_tuples)
        #对每条formula进行SIF计算

        #设置向量字典
        vector_dict = {}

        #--------------------------------------------
        for formula in dictionary_formula_lst_encoded_tuples:
            numpy_lst.append(self.__new_get_SIF_vector_representation(dictionary_formula_lst_encoded_tuples[formula],weight_matrix[formula]))
            #保存公式向量
            # vector_dict[formula] = self.__new_get_SIF_vector_representation(dictionary_formula_lst_encoded_tuples[formula],weight_matrix[formula])
            index_formula_id[idx] = formula
            idx += 1
        #保存向量字典
        print("-----saving vectordict-----")
        # file = open('vector_file/vector_dict.pickle','wb')
        # pickle.dump(vector_dict,file)
        # file.close()

        print(np.array(numpy_lst).shape)
        #计算主成分
        pca = PCA()
        pca.fit(np.array(numpy_lst).reshape(len(dictionary_formula_lst_encoded_tuples),300))
        print('show pca')
        print(np.array(numpy_lst).reshape(len(dictionary_formula_lst_encoded_tuples),300).shape)
        #返回最大主成分
        u = pca.components_[0]
        #投射矩阵
        u = np.multiply(u,np.transpose(u))
        if len(u) < 300:
            for i in range(300 - len(u)):
                u = np.append(u, 0)
        #计算最终vec
        formula_vector = []
        for vs in numpy_lst:
            sub = np.multiply(u,vs)
            formula_vector.append(np.subtract(vs,sub))

        temp = numpy.concatenate(formula_vector, axis=0)
        # temp = numpy.concatenate(numpy_lst, axis=0)
        # tensor_values = Variable(torch.tensor(temp).double()).cuda()
        tensor_values = Variable(torch.tensor(temp).double())
        return tensor_values, index_formula_id

    # SIF formula vector
    def __new_get_SIF_vector_representation(self, lst_encoded_tuples,weight_matrix):
        formula_SIF = []
        a_par: float = 1e-1
        vs = np.zeros(300)
        flag = 0
        for encoded_tuple in lst_encoded_tuples:
            a_value = weight_matrix[flag] * a_par / (a_par + self.freq[encoded_tuple])  # SIF
            vs = np.add(vs, np.multiply(a_value, self.model.get_vector_representation(encoded_tuple)))  # vs += sif * wordvector
            flag += 1
        vs = np.divide(vs,len(lst_encoded_tuples))

        # print('show vs size')
        # print(vs.shape)
        return vs.reshape(1,300)

