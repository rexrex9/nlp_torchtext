from gensim.models import word2vec

if __name__ == '__main__':
    s1 = [0,1,2,3,4]
    s2 = [0,2,4,5,6]
    s3 = [2,3,4,4,6]
    s4 = [1,3,5,0,3]
    seqs = [s1,s2,s3,s4]
    model = word2vec.Word2Vec(seqs, vector_size=16, min_count=1)

    print(model.wv[1])

    print(model.wv.most_similar(1, topn=3))