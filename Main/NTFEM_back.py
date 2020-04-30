from Configuration.configuration import Configuration
from DataReader.mse_data_reader import MSEDataReader
from DataReader.wiki_data_reader import WikiDataReader
from Embedding_Preprocessing.encoder_tuple_level import TupleEncoder, TupleTokenizationMode
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from NTFEM_module import TangentCFTModule
import pickle
import time
class NTFEMBackEnd:
    def __init__(self, config_file, path_data_set, is_wiki=True, read_slt=True, queries_directory_path=None):
        if is_wiki:
            self.data_reader = WikiDataReader(path_data_set, read_slt, queries_directory_path)
        else:
            self.data_reader = MSEDataReader(path_data_set, read_slt)
        self.config = Configuration(config_file)
        self.tuple_encoder = TupleEncoder()
        self.encoder_map = {}
        self.node_id = 60000
        self.module = None

    #制作doc2vec输入
    def __encode_train_doc2vec_tuples(self):
        wrong_flag = 0
        dictionary_lst_encoded_tuples = {}
        print("reading train data...")
        dictionary_formula_slt_tuple = self.data_reader.get_collection()
        file = open('Dict/pickle_formula_opt.pickle','rb')
        pickle.dump(dictionary_formula_slt_tuple,file)
        file.close()
        # with open('Dict/pickle_formula.pickle', 'rb') as file:
        #     dictionary_formula_slt_tuple = pickle.load(file)
        #未经过处理的doc2vec
        documents = [TaggedDocument(doc,[key]) for key,doc in dictionary_formula_slt_tuple.items()]
        # print(documents)
        return documents


    def __encode_train_tuples(self, embedding_type, ignore_full_relative_path, tokenize_all, tokenize_number):
        """
        :param embedding_type: 嵌入类型，1、符号类型 2、符号值 3、分词后的类型加符号 4、未分词的类型加符号
        :param ignore_full_relative_path: 是否忽略全相关路径
        :param tokenize_all: 是否分词
        :param tokenize_number: 是否分词数字
        :return: 按slt或者opt解析分词后的字典
        """
        sum_flag = 0
        sum_true = 0
        dictionary_lst_encoded_tuples = {}
        print("reading train data...")
        #getcollection 利用tangents的处理方式将NTCIR的公式提取出来 建立tuple字典
        #tuple的格式是 ‘文章名+公式编号：公式’

        with open('Dict/binary_formula.pickle', 'rb') as file:
            dictionary_formula_slt_tuple =pickle.load(file)

        #原始读取文本
        # dictionary_formula_slt_tuple = self.data_reader.get_collection()
        # #
        # #保存字典
        # print('get_collection and saving:')
        # file = open('Dict/binary_formula.pickle','wb')
        # pickle.dump(dictionary_formula_slt_tuple,file)
        # file.close()

        # print("encoding train data...")
        # return 0
        for formula in dictionary_formula_slt_tuple:
            try:
                # dictionary_lst_encoded_tuples[formula] = self.__encode_lst_tuples(dictionary_formula_slt_tuple[formula],
                #                                                                   embedding_type, ignore_full_relative_path,
                #                                                                   tokenize_all,
                #                                                                   tokenize_number)
                dictionary_lst_encoded_tuples[formula] = dictionary_formula_slt_tuple[formula]
            except:
                sum_flag+=1
        print('wrong_flag:')
        print(sum_flag)
        print('get new lst encoded_tuples')

        # print(dictionary_lst_encoded_tuples)
        return dictionary_lst_encoded_tuples

    # encode-tuples temp-update-list
    def __encode_lst_tuples(self, list_of_tuples, embedding_type, ignore_full_relative_path, tokenize_all,
                            tokenize_number):

        encoded_tuples, temp_update_list, node_id = self.tuple_encoder.encode_tuples(self.encoder_map, self.node_id,
                                                                                     list_of_tuples, embedding_type,
                                                                                     ignore_full_relative_path,
                                                                                     tokenize_all, tokenize_number)
        # print('^--------------^')
        # print(encoded_tuples)
        # print(temp_update_list)
        # print(node_id)
        # print('^--------------^')

        self.node_id = node_id
        self.encoder_map.update(temp_update_list)
        return encoded_tuples

    def __save_encoder_map(self, map_file_path):
        file = open(map_file_path, "w")
        for item in self.encoder_map:
            file.write(str(item) + "," + str(self.encoder_map[item]) + "\n")
        file.close()

    def __load_encoder_map(self, map_file_path):
        file = open(map_file_path)
        line = file.readline().strip("\n")
        while line:
            self.encoder_map[(line.split(",")[0])] = int(line.split(",")[1])
            line = file.readline().strip("\n")
        self.node_id = max(list(self.encoder_map.values())) + 1
        file.close()

    #训练doc2vec
    def train_doc_model(self, map_file_path, model_file_path=None,
                    embedding_type=TupleTokenizationMode.Both_Separated, ignore_full_relative_path=True,
                    tokenize_all=False,
                    tokenize_number=True):
        self.module = TangentCFTModule()
        #得到doc2vec训练文本
        dictionary_formula_tuples_collection = self.__encode_train_doc2vec_tuples()

        # print("training the doc2vec text model...")
        # self.module.train_doc2vec_model(dictionary_formula_tuples_collection)

        print('skip training')
        return dictionary_formula_tuples_collection


    #训练模型
    def train_model(self, map_file_path, model_file_path=None,
                    embedding_type=TupleTokenizationMode.Both_Separated, ignore_full_relative_path=True,
                    tokenize_all=False,
                    tokenize_number=True):

        #创建module对象，负责把配置文件传入训练模型
        self.module = TangentCFTModule()

        #获取formula字典
        # 1 建立字典，{文章名+公式号：公式 list}
        # 2 修订字典，{文章名+公式号：分解成slt或者opt之后的list}
        dictionary_formula_tuples_collection = self.__encode_train_tuples(embedding_type, ignore_full_relative_path,
                                                                          tokenize_all, tokenize_number)

        print("training the fast text model...")
        #传入有学习参数的控制文件 和 字典的值
        time1 = time.time()
        #2 fasttext模型学习得到module.model
        self.module.train_model(self.config, list(dictionary_formula_tuples_collection.values()))
        time2 = time.time()
        print('training time use:',time2-time1,'s')
        #保存encoder map
        #encoder map 是对公式进行lst分解的副产物，每分解一条公式，就对其中的symbol进行编号这样就得到了所有symble的ID号
        self.__save_encoder_map(map_file_path)

        #保存model file
        if model_file_path is not None:
            print("saving the fast text model...")
            self.module.save_model(model_file_path)

        #保存model
        return dictionary_formula_tuples_collection

        #加载模型
    def load_model(self, map_file_path, model_file_path,
                   embedding_type=TupleTokenizationMode.Both_Separated, ignore_full_relative_path=True,
                   tokenize_all=False,
                   tokenize_number=True
                   ):
        #加载模型
        self.module = TangentCFTModule(model_file_path)

        #加载encoder map文件
        # self.__load_encoder_map(map_file_path)

        #获取formula tuple字典：同上
        dictionary_formula_tuples_collection = self.__encode_train_tuples(embedding_type, ignore_full_relative_path,
                                                                          tokenize_all, tokenize_number)

        #保存模型
        # self.__save_encoder_map(map_file_path)
        return dictionary_formula_tuples_collection
    def doc2vec_retrieval(self,dictionary_formula_tuples_collection, embedding_type, ignore_full_relative_path, tokenize_all,
                  tokenize_number):
        #得到query
        dictionary_query_tuples = self.data_reader.get_query()
        #获取检索结果
        retrieval_result = {}
        tensor_values, index_formula_id = self.module.doc2vec_index_collection(dictionary_formula_tuples_collection)
        for query in dictionary_query_tuples:
            query_vec = self.module.get_query_vector(dictionary_query_tuples[query],2,dictionary_formula_tuples_collection)
            retrieval_result[query] = self.module.formula_retrieval(tensor_values,index_formula_id,query_vec)
        return retrieval_result


    def retrieval(self, dictionary_formula_tuples_collection, embedding_type, ignore_full_relative_path, tokenize_all,
                  tokenize_number):
        print(len(dictionary_formula_tuples_collection))
        # print(dictionary_formula_tuples_collection)
        #创建向量字典和公式编号索引
        #----------------导入权重矩阵-------------------
        with open('dict_tree_sif/weight_matrix.pickle', 'rb') as file:
            weight_matrix = pickle.load(file)

        #--------------fasttext retrival---------------
        # tensor_values, index_formula_id = self.module.index_collection(dictionary_formula_tuples_collection)
        # --------------sif retrival-------------------
        # tensor_values, index_formula_id = self.module.SIF_index_collection(dictionary_formula_tuples_collection)
        time_st = time.time()
        tensor_values, index_formula_id = self.module.new_SIF_index_collection(dictionary_formula_tuples_collection,weight_matrix)
        time_ed = time.time()
        print("index time:",time_ed-time_st,"s")
        # --------------tfidf retrival-----------------
        # tensor_values, index_formula_id = self.module.Tfidf_index_collection(dictionary_formula_tuples_collection)
        # --------------lda retrival-------------------

        print(len(index_formula_id))
        #从queryfile中获取查询集
        dictionary_query_tuples = self.data_reader.get_query()
        print("getting query from collection")
        # print(dictionary_query_tuples)

        #获取检索结果
        retrieval_result = {}

        for query in dictionary_query_tuples:
            # print('show query')
            # print(query)
            # encoded_tuple_query = self.__encode_lst_tuples(dictionary_query_tuples[query], embedding_type,
            #                                                ignore_full_relative_path, tokenize_all, tokenize_number)
            encoded_tuple_query = dictionary_query_tuples[query]
            #根据模型种类来对query进行encoding 0:fasttext 1:other
            # print('dictionary_query_tuples[query]')
            # print(dictionary_query_tuples[query])

            # print('encoded_tuple_query')
            # print(encoded_tuple_query)

            #基本的查询向量
            # print('show query vec')
            # print(encoded_tuple_query)
            # query_vec = self.module.get_query_vector(encoded_tuple_query,0,dictionary_formula_tuples_collection)

            #SIF query_vec

            query_vec = self.module.get_SIF_query_vector(dictionary_formula_tuples_collection,encoded_tuple_query,tensor_values,index_formula_id)

            # print('query_vec')
            # print(query_vec)
            # print('show query tuples2')
            # print(query_vec)
            time_st = time.time()
            retrieval_result[query] = self.module.formula_retrieval(tensor_values, index_formula_id, query_vec)
            time_ed = time.time()
            print("query_time", time_ed - time_st, "s")
        return retrieval_result

    @staticmethod
    def create_result_file(result_query_doc, result_file_path, run_id):
        file = open(result_file_path, "w")

        for query_id in result_query_doc:
            count = 1
            query = "NTCIR12-MathWiki-" + str(query_id)
            line = query + " xxx "
            for x in result_query_doc[query_id]:
                doc_id = x
                score = result_query_doc[query_id][x]
                temp = line + doc_id + " " + str(count) + " " + str(score) + " Run_" + str(run_id)
                count += 1
                file.write(temp + "\n")
        file.close()
