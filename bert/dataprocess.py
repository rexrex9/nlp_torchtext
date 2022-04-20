import random

def getVocabs(data):
    vacab_list=['<pad>', '<mask>', '<cls>', '<sep>']
    vacab_set = set()
    for ss in data:
        for s in ss:
            vacab_set|=set(s)
    vacab_list.extend(list(vacab_set))
    vacab_dict = {v:i for i,v in enumerate(vacab_list)}
    return vacab_dict,vacab_set

def getTokensAndSegmentsSingle(tokens_a, tokens_b):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    tokens += tokens_b + ['<sep>']
    segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

def getTokensAndSegments(data):
    tokens,segments = [],[]
    for s in data:
        token,seg = getTokensAndSegmentsSingle(*s)
        tokens.append(token)
        segments.append(seg)
    return tokens,segments

def getParas(data):
    paras = []
    for d in data:
        paras.append(d[0])
        paras.append(d[1])
    return paras

def getNspData(data):
    paras = getParas(data)
    nsp_data = []
    nsp_Y = []
    for d in data:
        sentences = [d[0]]
        if random.random() < 0.5:
            sentences.append(d[1])
            nsp_Y.append(1)
        else:
            sentences.append(random.choice(paras))
            nsp_Y.append(0)
        nsp_data.append(sentences)
    return nsp_data,nsp_Y

def mapping(tokenss,mlm_true_wordss,vocab_dict):
    n_tokenss,mlm_Y = [],[]
    for tokens in tokenss:
        n_tokenss.append([vocab_dict[token] for token in tokens])
    for words in mlm_true_wordss:
        mlm_Y.append([vocab_dict[word] for word in words])
    return n_tokenss,mlm_Y

def maskMlmData(tokens,vocab_set):
    num_pred = round(len(tokens) * 0.15)  # 预测15%个随机词
    mlm_true_words,mlm_pred_positions=[],[]
    for i in range(num_pred):
        while True: #如果要替换的位置是'<mask>', '<cls>', '<sep>',则继续选择
            change_index = random.choice(range(len(tokens)))
            if tokens[change_index] not in ['<mask>', '<cls>', '<sep>']:
                break
        mlm_pred_positions.append(change_index)
        mlm_true_words.append(tokens[change_index])
        if random.random() < 0.8: # 80%概率mask
            tokens[change_index] = '<mask>'
        else:
            # 10%用随机词替换该词, 剩余10%保持不变
            if random.random() < 0.5:
                tokens[change_index] = random.choice(list(vocab_set))
    return tokens,mlm_true_words,mlm_pred_positions

def getMlmData(tokenss,vocab_set):
    n_tokenss,mlm_true_wordss, mlm_pred_positionss = [],[],[]
    for tokens in tokenss:
        tokens, mlm_true_words, mlm_pred_positions = maskMlmData(tokens,vocab_set)
        n_tokenss.append(tokens)
        mlm_true_wordss.append(mlm_true_words)
        mlm_pred_positionss.append(mlm_pred_positions)
    return n_tokenss,mlm_true_wordss, mlm_pred_positionss

def getPreData(data):
    vocab_dict, vacab_set = getVocabs(seqs)
    nsp_data, nsp_Y = getNspData(data) #生成nsp任务的文本数据
    tokenss, segmentss = getTokensAndSegments(nsp_data) #生成bert encoder所需输入
    tokenss, mlm_true_wordss, mlm_pred_positionss = getMlmData(tokenss,vacab_set) #生成mlm任务的文本数据
    tokenss, mlm_Y = mapping(tokenss,mlm_true_wordss,vocab_dict) #映射成索引
    return tokenss,segmentss,mlm_pred_positionss,nsp_Y,mlm_Y,vocab_dict

seqs = [[['i','b','c','d','e','f'],['a','m','c','f','j','g']],
        [['d','e','f','e','a','f'],['a','b','c','d','e','d']],
        [['h','i','j','k','h','b'],['a','b','e','f','g','a']],
        [['a','b','c','d','e','f'],['a','b','c','e','m','g']],
        [['b','l','n','e','f','h'],['e','e','m','d','j','f']],
        [['b','g','d','m','f','g'],['e','e','c','d','e','f']]]

if __name__ == '__main__':
    getPreData(seqs)
