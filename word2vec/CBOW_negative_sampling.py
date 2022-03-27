from torch import nn
import torch
from word2vec.vocabs import getvocabsOnlyIndex
from word2vec.negative_sampling import negative_sample

class CBOW_With_Negative_Sampling(nn.Module):

    def __init__(self,vocab_numbers,vector_size):
        super().__init__()

        self.word_embs = nn.Embedding(vocab_numbers,vector_size)
        self.bkp_word_embs = nn.Embedding(vocab_numbers,vector_size)

    def forward(self,cs,os):
        os = self.word_embs(os)
        os = torch.mean(os,0)
        cs = self.bkp_word_embs(cs)
        y = torch.sum(cs*os,1)
        return torch.sigmoid(y)

def word2vec( seqs, window_size = 1 ):
    vocabs = getvocabsOnlyIndex(seqs)
    net = CBOW_With_Negative_Sampling(len(vocabs),vector_size=16)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD( net.parameters(), lr=0.01)
    net.train()
    for seq in seqs:
        for i in range(0,len(seq)-(window_size*2)):
            window = seq[i:i+1+window_size*2]
            cs = [window[window_size]]
            neg = negative_sample(cs,vocabs)
            cs.extend(neg)
            #[2]
            cs = torch.LongTensor(cs)
            y=[1,0]
            window.pop(window_size)

            optimizer.zero_grad()
            #[2,window_size*2]
            os = torch.concat([torch.unsqueeze(torch.LongTensor(window),0) for _ in y],0)
            y_pred = net(cs,os)
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