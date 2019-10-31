import pickle
import MeCab
import re
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])
sys.path.append(dir_path)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_PATH = os.path.join(dir_path, 'data/')

m = MeCab.Tagger(f"-Ochasen")
symbols = list("!！\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n？、。〃〄々〆〇〈〉～《》「」『』【】〒〓〔〕〖〗〘〙〚〛〜〝〞〟〠〡〢〣〤〥〦〧〨〩 〬 〭 〮 〯〰〵〶〷〸〹〺〻〼〽〾〿）（")
stop_words = ['経験', '治療', '候補', '者', '区']
feature_path = DATA_PATH + 'feature.pkl'
tfidftransformer_path = DATA_PATH + 'tfidf_transformer.pkl'
tfidf_docs_path = DATA_PATH + 'tfidf_docs.pkl'
index2id_path = DATA_PATH + 'index2id.pkl'

loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))
tfidf_transformer = pickle.load(open(tfidftransformer_path, "rb"))
tfidf_docs = pickle.load(open(tfidf_docs_path, "rb"))
index2id = pickle.load(open(index2id_path, "rb"))

itos = {idx: word for word, idx in loaded_vec.vocabulary.items()}


def word_tokenize(s, pre_pos=['名詞']):
    s = m.parse(s)
    result = [i.split('\t')[0] for i in s.split('\n')[:-2]]
    pos = [i.split('\t')[3].split('-')[0] for i in s.split('\n')[:-2]]
    if len(pos) == len(result) and type(pre_pos) == list:
        result = [result[i] for i in range(len(result)) if pos[i] in pre_pos]
    return result

def remove_punctuation(data):
    result = list(set(data) - set(symbols))
    return result

def stopword_remover(sent):
    sent = [word for word in sent if word not in stop_words]
    return sent

def preprocess(s, pre_pos=['名詞']):
    result = word_tokenize(s, pre_pos)
    result = stopword_remover(result)
    result = remove_punctuation(result)
    return result

def get_doc_by_id(index: int, itos = itos):
    tmp_arr = tfidf_docs[index].toarray()[0]
    s = ''
    for i, value in enumerate(tmp_arr):
        if value != 0:
            s+=f' {itos[i]}'
    s = s.rstrip()
    return s

def get_ranking(query, k=15):
    query = [' '.join(preprocess(query))]
    tfidf_query = tfidf_transformer.transform(loaded_vec.transform(query))

    cosine_similarities = linear_kernel(tfidf_query, tfidf_docs).flatten()
    id_score = [(index2id[i], round(score, 3))
                for i, score in enumerate(cosine_similarities)]
    id_score = sorted(id_score, key=lambda x: x[1], reverse=True)
    return id_score[:k]

if __name__ == "__main__":
    query = '治療の脱臼と縫合の経験がある候補者を見つける'
    print(get_ranking(query))
