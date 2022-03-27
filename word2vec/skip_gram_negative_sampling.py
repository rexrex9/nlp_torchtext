from torch import nn
import torch
from word2vec.vocabs import getvocabsOnlyIndex
from word2vec.negative_sampling import negative_sample

class Skip_Gram_With_Negative_Sampling(nn.Module):

    def __init__(self,vocab_numbers,vector_size):
        super().__init__()
        self.word_embs = nn.Embedding(vocab_numbers,vector_size)
        self.bkp_word_embs = nn.Embedding(vocab_numbers,vector_size)

    def forward(self,cs,os):
        cs = self.word_embs(cs)
        os = self.bkp_word_embs(os)
        y = torch.sum(cs*os,1)
        y = torch.sigmoid(y)
        return y

def word2vec( seqs, window_size = 1 ):
    vocabs = getvocabsOnlyIndex(seqs)
    net = Skip_Gram_With_Negative_Sampling(len(vocabs),vector_size=16)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD( net.parameters(), lr=0.01)
    net.train()
    for seq in seqs:
        for i in range(0,len(seq)-(window_size*2)):
            window = seq[i:i+1+window_size*2]
            # [window*2+neg]
            cs = torch.LongTensor([window[window_size] for _ in range(window_size*4)])
            print(cs.shape)
            window.pop(window_size)
            y = [1 for _ in window]
            neg = negative_sample(window,vocabs)
            y.extend([0 for _ in neg])
            window.extend(neg)

            optimizer.zero_grad()
            # [window*2+neg]
            os = torch.LongTensor(window)
            y_pred = net(cs,os)
            # [window*2+neg]
            y  = torch.FloatTensor(y)
            loss = criterion(y_pred,y)
            loss.backward()
            optimizer.step()
            print(loss)


if __name__ == '__main__':
    s1 = [0,1,2,3,4]
    s2 = [0,2,4,5,6]
    s3 = [2,3,4,4,6]
    s4 = [1,3,5,0,3]
    seqs = [s1,s2,s3,s4]
    word2vec(seqs)