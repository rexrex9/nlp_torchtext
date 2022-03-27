from torch import nn
import torch
from word2vec.vocabs import getvocabsOnlyIndex

class CBOW(nn.Module):

    def __init__(self,vocabs,vector_size):
        super().__init__()
        self.vocabs = torch.LongTensor(vocabs)
        vocab_numbers = len(vocabs)
        self.word_embs = nn.Embedding(vocab_numbers,vector_size)
        self.bkp_word_embs = nn.Embedding(vocab_numbers,vector_size)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.word_embs(x)
        x = torch.mean(x,0)
        bkp = self.bkp_word_embs(self.vocabs)
        y = torch.matmul(x,bkp.T)
        y = self.softmax(y)
        return torch.unsqueeze(y,0)

def word2vec( seqs, window_size = 1 ):
    vocabs = getvocabsOnlyIndex(seqs)
    net = CBOW(vocabs,vector_size=16)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD( net.parameters(), lr=0.01)
    net.train()
    for seq in seqs:
        for i in range(0,len(seq)-(window_size*2)):
            optimizer.zero_grad()
            window = seq[i:i+1+window_size*2]
            #[1]
            y = torch.LongTensor([window[window_size]])
            window.pop(window_size)
            #[window*2]
            x =  torch.LongTensor(window)
            y_pred = net(x)
            loss = criterion(y_pred, y)
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