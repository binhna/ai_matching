from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import MeCab
import re
from copy import deepcopy
from numpy import dot
from numpy.linalg import norm


m = MeCab.Tagger(f"-Ochasen")
symbols = list(
    "!！\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n？、。〃〄々〆〇〈〉～《》「」『』【】〒〓〔〕〖〗〘〙〚〛〜〝〞〟〠〡〢〣〤〥〦〧〨〩 〬 〭 〮 〯〰〵〶〷〸〹〺〻〼〽〾〿）（")
stop_words = ['経験', '治療', '候補', '者', '区']

trans = {'会員ID': 'id', 'ふりがな': 'kana_name', '年齢': 'age', '性別': 'gender',
         '職種': 'career', '学位取得': 'degree', '大学名': 'uni', '携帯電話': 'phone', 'メールアドレス': 'email',
         '住所1': 'address_1', '住所2': 'address_2', 'その他住所1': 'other_address_1', 'その他住所2': 'other_address_2',
         '専門分野': 'spec_field', '専門科目': 'spec_subject', '専門系': 'spec', '常勤求職': 'fulltime',
         '定期非常勤求職': 'parttime', '市区町村': 'city'}

skills = ['外科救急', '健診', '検査経験', '骨折対応', '子宮細胞診', '手術経験', '上部内視鏡',
          '心エコー', '心電図', '人工透析', '挿管', '脱臼対応', '特殊健診', '読影(胃部)',
          '読影(胸部)', '読影（その他）', '内科救急', '乳房視触診', '婦人科検診', '腹部エコー', '縫合処置']

df = pd.read_excel('/Users/binhna/Downloads/会員情報サンプル.xlsx')
df.rename(columns=trans, inplace=True)
df = df.replace({pd.np.nan: None})

# print("Column headings:")
# print(df.columns)

goal_df = df[skills+list(trans.values())]
# columns that we need to take the information
feature = ['gender', 'career', 'degree', 'uni', 'address_1', 'address_2', 'other_address_1',
           'other_address_2', 'spec_field', 'spec_subject', 'spec', 'city', 'fulltime', 'parttime']

# input is a string, pre_pos could be a list of POS or just an empty string
# use mecab to tokenize words in the string and return an array of words
def word_tokenize(s, pre_pos=['名詞']):
    s = m.parse(s)
    result = [i.split('\t')[0] for i in s.split('\n')[:-2]]
    pos = [i.split('\t')[3].split('-')[0] for i in s.split('\n')[:-2]]
    if len(pos) == len(result) and type(pre_pos) == list:
        result = [result[i] for i in range(len(result)) if pos[i] in pre_pos]
    return result


def cosine(a, b):
    return dot(a, b)/(norm(a)*norm(b))


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


def rename(test_string):
    pattern = '[^\d\s_-]+'
    result = re.findall(pattern, test_string)
    return result

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


def df2collection(df):
    n_row, n_col = df.shape
    collections = []
    for i in range(n_row):
        dump_sample = goal_df.iloc[i, :].to_dict()
        collections.append(to_rawtext(dump_sample))
    return collections


def get_by_id(id):
    index = goal_df[goal_df['id'] == id].index.values[0]
    print('index: ', index)
    print(docs[index])
    return goal_df[goal_df['id'] == id].to_dict()

def docs2tfidf(docs):
    # settings that you use for count vectorizer will go here
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)

    # just send in all your docs here
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
    return tfidf_matrix


def padding(docs):
    max_len = max([len(cv) for cv in docs])
    new_docs = []
    for cv in docs:
        pad = ['PAD']*(max_len-len(cv))
        pad = ' '.join(pad)
        new_docs.append(f'{cv} {pad}')
    return new_docs

def get_ranking(query, docs):
    query = ' '.join(preprocess(query))
    print('keywords from query: ', preprocess(query))
    new_docs = docs.copy()
    new_docs.append(query)
    # new_docs = padding(new_docs)
    tfidf = docs2tfidf(new_docs)
    ids = list(goal_df['id'])
    ranking = []
    for i in range(len(ids)):
        ranking.append(
            (ids[i], cosine(tfidf[i].toarray()[0], tfidf[-1].toarray()[0])))
    ranking = sorted(ranking, key=lambda x: x[1], reverse=True)
    return ranking

if __name__ == "__main__":
    query = '歯科専門医候補者'
    # tìm ứng viên có kinh nghiệm điều trị trật khớp và khâu
    query = '治療の脱臼と縫合の経験がある候補者を見つける'
    # query = '治療の脱臼と縫合の経験がある候補者を見つける parttime'
    docs = df2collection(goal_df)
    ranking = get_ranking(query, docs)
    print(ranking[:10])
