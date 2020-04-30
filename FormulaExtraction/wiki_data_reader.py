import os
import pickle
import unicodedata
from abc import ABC
from DataReader.abstract_data_reader import AbstractDataReader
from TangentS.math_tan.math_document import MathDocument
from TangentS.math_tan.math_extractor import MathExtractor
import traceback
from DataReader.tree_sif import flat

class WikiDataReader(AbstractDataReader, ABC):
    def __init__(self, collection_file_path, read_slt=True, queries_directory_path=None):
        self.read_slt = read_slt
        self.collection_file_path = collection_file_path
        self.queries_directory_path = queries_directory_path
        super()

    # def flat(l):
    #     for k in l:
    #         if not isinstance(k, (list, tuple)):
    #             yield k
    #         else:
    #             yield fr
    def get_collection_treesif(self, ):
        except_count = 0
        dictionary_formula_tuples = {}
        dictionary_formula_tuples_treesif= {}
        root = self.collection_file_path
        for directory in os.listdir(root):
            # print(directory)
            temp_address = root+"/"+directory+"/"
            # print(temp_address)
            if not os.path.isdir(temp_address):
                # print('no file')
                continue
            #old series
            # temp_address = temp_address +"/Articles"
            # temp_address = temp_address
            for filename in os.listdir(temp_address):
                # print('temp file_name')
                # print(file_name)
                file_path = temp_address + '/' + filename
                parts = filename.split('/')
                file_name = os.path.splitext(parts[len(parts) - 1])[0]
                try:
                    (ext, content) = MathDocument.read_doc_file(file_path)

                    #得到formulas公式集合，一个字典是一篇文章中的
                    formulas = MathExtractor.parse_from_xml(content, 1, operator=(not self.read_slt), missing_tags=None,
                                                            problem_files=None)
                    formulas_for_treeisf = MathExtractor.parse_from_xml_second(content, 1, operator=(not self.read_slt), missing_tags=None,
                                                            problem_files=None)
                    temp = str(unicodedata.normalize('NFKD', file_name).encode('ascii', 'ignore'))
                    temp = temp[2:]
                    file_name = temp[:-1]
                    # print("show formulas:")
                    # print(formulas)
                    #从formula中解析paris
                    for key in formulas:
                        tuples = formulas[key].get_pairs(window=2, eob=True)
                        dictionary_formula_tuples[file_name + ":" + str(key)] = tuples
                    #提取treesif formula原型
                    for key in formulas_for_treeisf:
                        tuples_2 = formulas_for_treeisf[key]
                        dictionary_formula_tuples_treesif[file_name + ":" + str(key)] = tuples_2
                except:
                    traceback.print_exc()
                    except_count += 1
                    print(file_name)

        #保存所需要的formula
        # file = open('dict_tree_sif/pari_formula_treesif.pickle','wb')
        # pickle.dump(dictionary_formula_tuples_treesif,file)
        # file.close()

        return dictionary_formula_tuples_treesif
    def get_collection(self, ):
        except_count = 0
        dictionary_formula_tuples = {}
        dictionary_formula_tuples_treesif= {}
        root = self.collection_file_path
        for directory in os.listdir(root):
            # print(directory)
            temp_address = root+"/"+directory+"/"
            # print(temp_address)
            if not os.path.isdir(temp_address):
                # print('no file')
                continue
            #old series
            # temp_address = temp_address +"/Articles"
            # temp_address = temp_address
            for filename in os.listdir(temp_address):
                # print('temp file_name')
                # print(file_name)
                file_path = temp_address + '/' + filename
                parts = filename.split('/')
                file_name = os.path.splitext(parts[len(parts) - 1])[0]
                try:
                    (ext, content) = MathDocument.read_doc_file(file_path)

                    #得到formulas公式集合，一个字典是一篇文章中的
                    formulas = MathExtractor.parse_from_xml(content, 1, operator=(not self.read_slt), missing_tags=None,
                                                            problem_files=None)
                    formulas_for_treeisf = MathExtractor.parse_from_xml_second(content, 1, operator=(not self.read_slt), missing_tags=None,
                                                            problem_files=None)
                    temp = str(unicodedata.normalize('NFKD', file_name).encode('ascii', 'ignore'))
                    temp = temp[2:]
                    file_name = temp[:-1]
                    # print("show formulas:")
                    # print(formulas)
                    #从formula中解析paris
                    for key in formulas:
                        tuples = formulas[key].get_pairs(window=2, eob=True)
                        dictionary_formula_tuples[file_name + ":" + str(key)] = tuples
                    #提取treesif formula原型
                    for key in formulas_for_treeisf:
                        tuples_2 = formulas_for_treeisf[key]
                        dictionary_formula_tuples_treesif[file_name + ":" + str(key)] = tuples_2
                except:
                    traceback.print_exc()
                    except_count += 1
                    print(file_name)

        #保存所需要的formula
        # file = open('dict_tree_sif/dict_formula.pickle','wb')
        # pickle.dump(dictionary_formula_tuples_treesif,file)
        # file.close()

        # print("test dic")
        # print(dictionary_formula_tuples_treesif)
        # print("test dic for formula_2")
        # print(dictionary_formula_tuples)
        return dictionary_formula_tuples

    def get_query(self,):
        except_count = 0
        dictionary_query_tuples = {}
        query_formula_tuples_treesif = {}
        for j in range(1, 21):
            temp_address = self.queries_directory_path + '/' + str(j) + '.html'
            try:
                (ext, content) = MathDocument.read_doc_file(temp_address)
                formulas = MathExtractor.parse_from_xml(content, 1, operator=(not self.read_slt), missing_tags=None,
                                                        problem_files=None)
                formulas_for_treeisf = MathExtractor.parse_from_xml_second(content, 1, operator=(not self.read_slt),
                                                                           missing_tags=None,
                                                                           problem_files=None)
                for key in formulas:
                    tuples = formulas[key].get_pairs(window=2, eob=True)
                    dictionary_query_tuples[j] = tuples
                for key in formulas_for_treeisf:
                    tuples_2 = formulas_for_treeisf[key]
                    query_formula_tuples_treesif[j] = tuples_2
                for key, value in query_formula_tuples_treesif.items():
                    query_formula_tuples_treesif[key] = list(flat(query_formula_tuples_treesif[key]))
            except:
                except_count += 1
                # print(j)
        return query_formula_tuples_treesif

    def get_noflat_query(self,):
        except_count = 0
        dictionary_query_tuples = {}
        query_formula_tuples_treesif = {}
        for j in range(1, 21):
            temp_address = self.queries_directory_path + '/' + str(j) + '.html'
            try:
                (ext, content) = MathDocument.read_doc_file(temp_address)
                formulas = MathExtractor.parse_from_xml(content, 1, operator=(not self.read_slt), missing_tags=None,
                                                        problem_files=None)
                formulas_for_treeisf = MathExtractor.parse_from_xml_second(content, 1, operator=(not self.read_slt),
                                                                           missing_tags=None,
                                                                           problem_files=None)
                for key in formulas:
                    tuples = formulas[key].get_pairs(window=2, eob=True)
                    dictionary_query_tuples[j] = tuples
                for key in formulas_for_treeisf:
                    tuples_2 = formulas_for_treeisf[key]
                    query_formula_tuples_treesif[j] = tuples_2
                # for key, value in query_formula_tuples_treesif.items():
                #     query_formula_tuples_treesif[key] = list(flat(query_formula_tuples_treesif[key]))
            except:
                except_count += 1
                # print(j)
        return query_formula_tuples_treesif