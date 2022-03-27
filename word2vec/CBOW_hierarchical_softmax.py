from torch import nn
import torch
from word2vec.vocabs import getvocabsOnlyIndex
from word2vec.hierarchical_softmax import Hierarchical_Softmax

class CBOW_With_Hierarchical_Softmax(nn.Module):

    def __init__(self,node_number,vector_size):
        super().__init__()
        self.node_embs = nn.Embedding(node_number,vector_size)

    def forward(self,os,nodes):
        os = self.node_embs(os)
        os = torch.mean(os, 1)
        nodes = self.node_embs(nodes)
        y = torch.sum(os*nodes,1)
        y = torch.sigmoid(y)
        return y

def word2vec( seqs, window_size = 1 ):
    vocabs = getvocabsOnlyIndex(seqs)
    HS = Hierarchical_Softmax(vocabs)

    net = CBOW_With_Hierarchical_Softmax(HS.getNodeNumber(),vector_size=16)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD( net.parameters(), lr=0.01)
    net.train()
    for seq in seqs:
        for i in range(0,len(seq)-(window_size*2)):
            window = seq[i:i+1+window_size*2]
            cs = [window[window_size]]
            paths, isLefts = HS.getPathByLeaves(cs) #得到所有节点与是否是走左边的标注
            window.pop(window_size)
            # 开始迭代
            optimizer.zero_grad()
            # [deep,window]
            os = torch.concat([torch.unsqueeze(torch.LongTensor(window), 0) for _ in paths], 0)
            # [deep]
            nodes = torch.LongTensor(paths)
            y_pred = net(os,nodes)
            # [deep]
            y = torch.FloatTensor(isLefts)
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