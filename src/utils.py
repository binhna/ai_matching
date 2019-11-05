from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
import MeCab
import re
import os
import os.path
from os import path
import pandas as pd
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])
sys.path.append(dir_path)

symbols = list(
    "!！\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n？、。〃〄々〆〇〈〉～《》「」『』【】〒〓〔〕〖〗〘〙〚〛〜〝〞〟〠〡〢〣〤〥〦〧〨〩 〬 〭 〮 〯〰〵〶〷〸〹〺〻〼〽〾〿）（")
stop_words = ['経験', '治療', '候補', '者', '区']

trans = {'会員ID': 'id', 'ふりがな': 'kana_name', '年齢': 'age', '性別': 'gender',
         '職種': 'career', '学位取得': 'degree', '大学名': 'uni', '携帯電話': 'phone', 'メールアドレス': 'email',
         '住所1': 'address_1', '住所2': 'address_2', 'その他住所1': 'other_address_1', 'その他住所2': 'other_address_2',
         '専門分野': 'spec_field', '専門科目': 'spec_subject', '専門系': 'spec', '常勤求職': 'fulltime',
         '定期非常勤求職': 'parttime', '市区町村': 'ward', '都道府県': 'city'}

skills = ['外科救急', '健診', '検査経験', '骨折対応', '子宮細胞診', '手術経験', '上部内視鏡',
          '心エコー', '心電図', '人工透析', '挿管', '脱臼対応', '特殊健診', '読影(胃部)',
          '読影(胸部)', '読影（その他）', '内科救急', '乳房視触診', '婦人科検診', '腹部エコー', '縫合処置']
# columns that we need to take the information
feature = ['gender', 'career', 'degree', 'uni', 'address_1', 'address_2', 'other_address_1',
           'other_address_2', 'spec_field', 'spec_subject', 'spec', 'ward', 'city', 'fulltime', 'parttime']
m = MeCab.Tagger(f"-Ochasen")

# input: a sentence
# if pre_pos is a list, tokenize sentence and select only words have POS in pre_pos
# return a list of words
def word_tokenize(s, pre_pos=['名詞']):
    s = m.parse(s)
    result = [i.split('\t')[0] for i in s.split('\n')[:-2]]
    pos = [i.split('\t')[3].split('-')[0] for i in s.split('\n')[:-2]]
    if len(pos) == len(result) and type(pre_pos) == list:
        result = [result[i] for i in range(len(result)) if pos[i] in pre_pos]
    return result

# input list of words
# remove all symbols in list of data
# return a list of words which are not symbols
def remove_punctuation(data):
    result = list(set(data) - set(symbols))
    return result

# input: list of words
# remove all stop words in sentence
# return list of words
def stopword_remover(sent):
    sent = [word for word in sent if word not in stop_words]
    return sent

# input: a sentence
# do word_tokenize, stopwords, remove punctuation
# return a list of words
def preprocess(s, pre_pos=['名詞']):
    result = word_tokenize(s, pre_pos)
    result = stopword_remover(result)
    result = remove_punctuation(result)
    return result

# input: a dict (a row in a df)
# get all values (columns name or value of columns), tokenize them, store in a raw text
# a string of keywords seperated by space
def to_rawtext(sample: dict):
    raw = ''
    for ft in skills:
        if sample[ft] in ['はい', 'できる']:
            raw += ' '.join(preprocess(ft, ''))
            raw += ' '
    for ft in feature[:-2]:
        if sample[ft] != None:
            #print(sample[ft])
            preprocess_text = ' '.join(rename(sample[ft]))
            raw += ' '.join(preprocess(preprocess_text, ''))
            raw += ' '
    for ft in feature[-2:]:
        if sample[ft] == '○':
            raw += f'{ft} '
    return raw.rstrip()

# input: a string
# find all words that are not numbers, alphabet words, _ and -
# return a list of words satisfy the condition
def rename(test_string):
    pattern = '[^\d\s_-]+'
    result = re.findall(pattern, test_string)
    return result

# input: a dataframe (candidate)
# output: a list of string, each string is a candidate resume which is being converted to string of keywords
def df2collection(df):
    n_row, n_col = df.shape
    collections = []
    for i in range(n_row):
        dump_sample = df.iloc[i, :].to_dict()
        collections.append(to_rawtext(dump_sample))
    return collections

#========================== Maybe we dont need these, I will explain it later if I think it's useful


# def get_doc_by_id(index: int, itos=itos, tfidf_docs=None):
#     tmp_arr = tfidf_docs[index].toarray()[0]
#     s = ''
#     for i, value in enumerate(tmp_arr):
#         if value != 0:
#             s += f' {itos[i]}'
#     s = s.rstrip()
#     return s


# def get_info_by_id(id, goal_df):
#     index = goal_df[goal_df['id'] == id].index.values[0]
#     print('index: ', index)
#     print(docs[index])
#     return goal_df[goal_df['id'] == id].to_dict()
#==============================================================================
