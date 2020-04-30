from gensim.models import FastText
from gensim import corpora, models, similarities
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import datetime


class NTFEM_base:
    def __init__(self,):
        self.model = None

    def train_Lda(self, config, fast_text_train_data):
        print('init corpus for tfidf')
        self.dictionary = corpora.Dictionary(fast_text_train_data)
        self.corpus_init = [self.dictionary.doc2bow(text) for text in fast_text_train_data]
        print("Training the model")
        # 导入fasttext模型进行训练
        self.model = models.LdaModel(self.corpus_init)

    def train_Tfid(self, config, fast_text_train_data):
        print('init corpus for tfidf')
        self.dictionary = corpora.Dictionary(fast_text_train_data)
        self.corpus_init = [self.dictionary.doc2bow(text) for text in fast_text_train_data]

        print("Training the model")
        # 导入fasttext模型进行训练
        self.model = models.TfidfModel(self.corpus_init)
        print('show copus')
        corpus_tfidf = self.model[self.corpus_init]
        # self.model = FastText(fast_text_train_data, size=size, window=window, sg=sg, hs=hs,
        #                       workers=1, negative=negative, iter=iteration, min_n=min_n,
        #                       max_n=max_n, word_ngrams=word_ngrams)
    def train_doc2vec(self,doc2vecdocuments):
        self.model = Doc2Vec(doc2vecdocuments,min_count = 1 ,vector_size = 300, window = 5,sample = 1e-3,negative=5,workers = 4)
        self.model.save('Models/doc2vec')

    def train(self, config, fast_text_train_data):
        """
        Takes in the fastText parameters and the train data and trains the FastText model
        :param config: configuration for fastText model
        :param fast_text_train_data: train data
        :return:
        """
        size = config.vector_size                #向量大小
        window = int(config.context_window_size) #滑动窗口大小
        sg = int(config.skip_gram)               #是否使用skip_gram
        hs = int(config.hs)                      #是否使用hs
        negative = int(config.negative)          #是否使用negative
        iteration = int(config.iter)             #是否使用iteration
        min_n = int(config.min)                  #ngram min
        max_n = int(config.max)                  #ngram max
        word_ngrams = int(config.ngram)          #ngram 参数

        train_start_time = datetime.datetime.now()
        print("Training the model")
        print("get parameter size")
        print(size)
        print('get parameter window')
        print(window)
        print('get parameter hs')
        print(hs)
        print('get parameter ns')
        print(negative)
        #导入fasttext模型进行训练
        self.model = FastText(fast_text_train_data, size=size, window=window, sg=sg, hs=hs,
                              workers=1, negative=negative, iter=iteration, min_n=min_n,
                              max_n=max_n, word_ngrams=word_ngrams)

        train_end_time = datetime.datetime.now()
        "Returns the train time of the model"
        return train_end_time - train_start_time

    def save_model(self, model_file_path):
        print('new model_file_path')
        print(model_file_path)
        file_name = (model_file_path+".wv.vectors.npy")
        self.model.save(file_name)

    def load_model(self, model_file_path):
        self.model = FastText.load(model_file_path + ".wv.vectors.npy")

    def tfidf_load_model(self,model_file_path):
        self.model = models.TfidfModel.load(model_file_path + ".wv.vectors.npy")

    def lda_load_model(self,model_file_path):
        self.model = models.LdaModel.load(model_file_path+".wv.vectors.npy")

    def lsi_load_model(self,model_file_path):
        self.model = models.LsiModel.load(model_file_path+".wv.vectors.npy")

    def get_vector_representation(self, encoded_math_tuple):
        return self.model.wv[encoded_math_tuple]

    # def get_weight_vector_representation(self,encoded_math_tuple,we):



    def get_tfidf_vector_representation(self,encoded_math_tuple):
        # print('show final tuple')
        # print(encoded_math_tuple)
        #
        # print('show doc_bow')
        # doc_bow = [(0,1),(1,1)]
        # print(self.model[doc_bow])

        # print('model[0]')
        # print(self.model)
        print('self.model[encoded_math_tuple]')
        print(self.model[encoded_math_tuple])
        return self.model[encoded_math_tuple]
