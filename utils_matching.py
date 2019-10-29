import MeCab
m = MeCab.Tagger(f"-Ochasen")

stop_words = ['経験', '治療', '候補', '者']
#https://www.singhealth.com.sg/patient-care/specialties-services
with open('skills.txt', 'r') as f:
    skills = f.readlines()
    skills = [l.rstrip() for l in skills]

def stopword_remover(sent):
    sent = [word for word in sent if word not in stop_words]
    return sent


def word_tokenize(s, pre_pos=['名詞']):
    s = m.parse(s)
    result = [i.split('\t')[0] for i in s.split('\n')[:-2]]
    pos = [i.split('\t')[3].split('-')[0] for i in s.split('\n')[:-2]]
    if len(pos) == len(result) and type(pre_pos) == list:
        result = [result[i] for i in range(len(result)) if pos[i] in pre_pos]
    return result

# jaccard_similarity(sent, skill)
def jaccard_similarity(predicted, gold):
    # predicted = predicted.split(' ')
    # gold = word_tokenize(gold, '')
    intersection = set(predicted).intersection(set(gold))
    union = set(predicted).union(set(gold))
    return len(intersection)/len(union)

def preprocess(query):
    sent = word_tokenize(query)
    sent = stopword_remover(sent)
    print(sent)
    new_sent = ''.join(sent)
    return new_sent

# preprocessed sentence, list of skills
def matcher(sent, skills):
    # print(f"input: {predicted}")
    result = []
    for i, c in enumerate(skills):
        result.append((jaccard_similarity(sent, c), skills[i]))
    result = sorted(result, key=lambda x: x[0], reverse=True)
    print(result)
    # print(f"output: {countries[result.index(max(result))]}")
    #return ac_db[result.index(max(result))]
    score = result[0][0]
    if score != 0:
        result = [skill[1] for skill in result if skill[0] == score]
        return result
    print('Cant find candidate')
    return None

# input is a row query, output is a list of skills in database
def get_skills(query):
    sent = preprocess(query)
    print(matcher(sent, skills))

if __name__ == "__main__":
    # tìm ứng viên có kinh nghiệm điều trị trật khớp
    query = '治療の脱臼の経験がある候補者を見つける'
    # tìm ứng viên có kinh nghiệm điều trị trật khớp và khâu
    query = '治療の脱臼と縫合の経験がある候補者を見つける'
    # ứng viên đồ họa máy tính
    # query = 'コンピュータグラフィックス候補'
    # ứng viên hồi sức cấp cứu
    query = '緊急蘇生の候補者'
    get_skills(query)
