import gensim
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import numpy as np
import pandas as pd
from DataReader.wiki_data_reader import WikiDataReader