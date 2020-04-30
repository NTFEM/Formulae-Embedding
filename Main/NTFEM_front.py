import argparse

from Embedding_Preprocessing.encoder_tuple_level import TupleTokenizationMode
from tangent_cft_back_end import TangentCFTBackEnd
import time

def main():
    #终端输入参数
    parser = argparse.ArgumentParser(description='Given the configuration file for training Tangent_CFT model.'
                                                 'This function train the model and then does the retrieval task on'
                                                 'NTCIR-12 formula retrieval task.')

    # parser.add_argument('--t', type=bool, help="Value True for training a new model and False for loading a model",
    #                     default=True)
    parser.add_argument('--t', type=int, help="0:use exsit model 1:train a new model",choices=range(0,3),
                        default=1)
    # parser.add_argument('--r', type=bool, help="Value True to do the retrieval on NTCIR12 dataset",
    #                     default=True)
    parser.add_argument('--r', type=int, help="Value True to do the retrieval on NTCIR12 dataset",
                        default=0,choices=range(0,2))

    parser.add_argument('-ds', type=str, help="File path of training data. If using NTCIR12 dataset, "
                                              "it should be MathTagArticles directory. If using the MSE dataset, it"
                                              "should be csv file of formula", required=True)
    parser.add_argument('-cid', metavar='cid', type=int, help='Configuration file.', required=True)
    parser.add_argument('--wiki', type=bool, help="Determines if the dataset is wiki or not.", default=True)
    # parser.add_argument('--wiki', type=int, help="Determines if the dataset is wiki or not.", default=1,choices=range(0,2))

    # parser.add_argument('--slt', type=bool, help="Determines to use slt (True) or opt(False)", default=True)
    parser.add_argument('--slt', type=int, help="Determines to use slt (True) or opt(False)", default=1,choices=range(0,2))

    parser.add_argument('-em', type=str, help="File path for encoder map.", required=True)
    parser.add_argument('--mp', type=str, help="Model file path.", default=None)
    parser.add_argument('--qd', type=str, help="NTCIR12 query directory.", default=None)
    parser.add_argument('--rf', type=str, help="Retrieval result file path.", default="ret_res")
    parser.add_argument('--ri', type=int, help="Run Id for Retrieval.", default=1)

    # parser.add_argument('--frp', type=bool, help="Determines to ignore full relative path", default=True)
    parser.add_argument('--frp', type=int, help="Determines to ignore full relative path", default=1,choices=range(0,2))

    parser.add_argument('--ta', type=bool, help="Determines to tokenize all", default=False)
    # parser.add_argument('--ta', type=int, help="Determines to tokenize all", default=0,choices=range(0,2))

    parser.add_argument('--tn', type=bool, help="Determines to tokenize numbers", default=True)
    # parser.add_argument('--tn', type=int, help="Determines to tokenize numbers", default=1,choices=range(0,2))

    parser.add_argument('--et', help='Embedding type; 1:Value, 2:Type, 3:Type and Value separated and'
                                     ' 4: Type and Value Not Separated, 2 for formula level', choices=range(1, 5),
                        default=3, type=int)
    args = vars(parser.parse_args())

    #参数含义
    train_model = (args['t'])
    do_retrieval = (args['r'])
    dataset_file_path = args['ds']

    config_id = args['cid']
    is_wiki = args['wiki']
    read_slt = args['slt']
    encoder_file_path = args['em']
    model_file_path = args['mp']
    res_file = args['rf']
    run_id = args['ri']
    ignore_full_relative_path = args['frp']
    tokenize_all = args['ta']
    tokenize_number = args['tn']
    queries_directory_path = args['qd']
    embedding_type = TupleTokenizationMode(args['et'])
    map_file_path = "Saved_model/Embedding_Preprocessing/" + str(encoder_file_path)
    config_file_path = "Configuration/config/config_" + str(config_id)

    #config file：配置文件
    #pathdataset：数据集
    #iswiki：是否是wiki数据
    #read——slt：确认读入文件是否slt
    #query：query查询文件路径
    system = TangentCFTBackEnd(config_file=config_file_path, path_data_set=dataset_file_path, is_wiki=is_wiki,
                               read_slt=read_slt, queries_directory_path=queries_directory_path)

    # t = 2 训练doc2vec向量
    if train_model == 2:
        print('now training doc2vec')
        dictionary_formula_tuples_collection = system.train_doc_model(
            map_file_path=map_file_path,
            model_file_path=model_file_path,
            embedding_type=embedding_type, ignore_full_relative_path=ignore_full_relative_path,
            tokenize_all=tokenize_all,
            tokenize_number=tokenize_number
        )
        print('doc2vec train finished')
        if do_retrieval:
            print('now doing retrieval with doc2vec')
            retrieval_result = system.doc2vec_retrieval(dictionary_formula_tuples_collection,embedding_type,ignore_full_relative_path,tokenize_all,tokenize_number)
            system.create_result_file(retrieval_result,"Retrieval_Results/" + res_file, run_id)

    # t 为1 训练词向量模型
    if train_model == 1:
        print('now is training a new model')
        #调用backend中的数据 训练模型
        dictionary_formula_tuples_collection = system.train_model(
            map_file_path=map_file_path,
            model_file_path=model_file_path,
            embedding_type=embedding_type, ignore_full_relative_path=ignore_full_relative_path,
            tokenize_all=tokenize_all,
            tokenize_number=tokenize_number
        )
        # r 为1 进行搜索查询
        if do_retrieval:
            print('now is doing retrieval')
            #获得检索结果
            retrieval_result = system.retrieval(dictionary_formula_tuples_collection)
            system.create_result_file(retrieval_result, "Retrieval_Results/" + res_file, run_id)
    elif train_model == 0:
        print('now use exsit model')
        dictionary_formula_tuples_collection = system.load_model(
            map_file_path=map_file_path,
            model_file_path=model_file_path,
            embedding_type=embedding_type, ignore_full_relative_path=ignore_full_relative_path,
            tokenize_all=tokenize_all,
            tokenize_number=tokenize_number
        )
        print('load finished')
        if do_retrieval:
            print('now is doing retrieval')

            retrieval_result = system.retrieval(dictionary_formula_tuples_collection,embedding_type,ignore_full_relative_path,tokenize_all,tokenize_number)

            system.create_result_file(retrieval_result, "Retrieval_Results/" + res_file, run_id)
            print('retrival finished')
if __name__ == "__main__":
    main()
