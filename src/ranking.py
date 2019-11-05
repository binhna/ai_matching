import pickle
import MeCab
import re
import os
import os.path
from os import path
import pandas as pd
import sys
import logging
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])
sys.path.append(dir_path)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from src.utils import *

logging.getLogger().setLevel(logging.INFO)
DATA_PATH = os.path.join(dir_path, 'data/')

m = MeCab.Tagger(f"-Ochasen")

# Pre-calculated for candidates
vocabs_path = DATA_PATH + 'vocabs.pkl'
tfidftransformer_path = DATA_PATH + 'tfidf_transformer_candidate.pkl'
tfidf_docs_path = DATA_PATH + 'tfidf_candidate.pkl'
index2id_path = DATA_PATH + 'index2id.pkl'

def load_feature(path_df = ''):
    def dump_to_file(object, path):
        with open(path, 'wb') as fw:
            pickle.dump(object, fw)

    if not path.exists(vocabs_path) or not path.exists(tfidftransformer_path)\
            or not path.exists(tfidf_docs_path) or not path.exists(index2id_path):
        logging.info('The feature does not exist, we are creating them')
        if path_df == '':
            logging.error('You need to specify the path to candidate file')
            exit()
        df_candidate = pd.read_excel(path_df)
        df_candidate.rename(columns=trans, inplace=True)
        df_candidate = df_candidate.replace({pd.np.nan: None})
        candidate = df_candidate[skills+list(trans.values())]
        #row number to id
        index2id = candidate['id'].to_dict()
        dump_to_file(index2id, index2id_path)

        # from df to list of doc strings containing keywords
        docs = df2collection(candidate)
        #instantiate CountVectorizer()
        cv = CountVectorizer()
        #tfidf calculation
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        # this steps generates word counts for the words in your docs
        word_count_collections = cv.fit_transform(docs)

        tfidf_transformer.fit(word_count_collections)
        tfidf_docs = tfidf_transformer.transform(word_count_collections)

        dump_to_file(cv.vocabulary_, vocabs_path)
        dump_to_file(tfidf_transformer, tfidftransformer_path)
        dump_to_file(tfidf_docs, tfidf_docs_path)
        itos = {idx: word for word, idx in cv.vocabulary_.items()}
    else:
        logging.info('Loading...')
        cv = CountVectorizer(
            decode_error="replace", vocabulary=pickle.load(open(vocabs_path, "rb")))
        tfidf_transformer = pickle.load(open(tfidftransformer_path, "rb"))
        tfidf_docs = pickle.load(open(tfidf_docs_path, "rb"))
        index2id = pickle.load(open(index2id_path, "rb"))
        itos = {idx: word for word, idx in cv.vocabulary.items()}
        logging.info('Done!')
    return cv, tfidf_transformer, tfidf_docs, index2id, itos

def get_ranking(query, k=15):
    cv, tfidf_transformer, tfidf_docs, index2id, itos = load_feature(
        '/Users/binhna/Downloads/Thong tin candidates_sample_20191101_1.xlsx')
    
    query = [' '.join(preprocess(query))]
    print(f"query: {query}")
    tfidf_query = tfidf_transformer.transform(cv.transform(query))
    # count matrix for query
    word_count_query = cv.transform(query)
    # tfidf vector for query
    tfidf_query = tfidf_transformer.transform(word_count_query)

    cosine_similarities = linear_kernel(tfidf_query, tfidf_docs).flatten()
    id_score = [(index2id[i], round(score, 3))
                for i, score in enumerate(cosine_similarities)]
    id_score = sorted(id_score, key=lambda x: x[1], reverse=True)
    return id_score[:k]

if __name__ == "__main__":
    query = '治療の脱臼と縫合の経験がある候補者を見つける'
    query = '医療および腹腔鏡のスキルを持つ候補者'
    query = '歯科'
    query = '東京の女性歯科医'
    query = 'コンピュータグラフィックス候補'
    ranks = get_ranking(query)
    print(ranks)

