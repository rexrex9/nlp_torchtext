from torch import nn
import torch
from torch.nn import functional as F
from transformer.transformer import EncoderLayer

class BERTEncoder(nn.Module):

    def __init__(self, vocab_size, e_dim, h_dim, n_heads, n_layers, max_len=1024):
        '''
        :param vocab_size: 词汇数量
        :param e_dim: 词向量维度
        :param h_dim: Transformer编码层中间层的维度
        :param n_heads: Transformer多头注意力的头数
        :param n_layers: Transformer编码层的层数
        :param max_len: 序列最长长度
        '''
        super(BERTEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, e_dim)
        self.segment_embedding = nn.Embedding(2, e_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,e_dim))
        self.encoder_layers = nn.ModuleList( [EncoderLayer( e_dim, h_dim, n_heads ) for _ in range( n_layers )] )

    def forward(self, tokens, segments):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for layer in self.encoder_layers:
            X = layer(X)
        return X

class MaskLM(nn.Module):

    def __init__(self, vocab_size, h_dim, e_dim):
        super(MaskLM, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(e_dim, h_dim),
                                 nn.ReLU(),
                                 nn.LayerNorm(h_dim),
                                 nn.Linear(h_dim, vocab_size),
                                 nn.Softmax())

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):

    def __init__(self, e_dim):
        super(NextSentencePred, self).__init__()
        self.output = nn.Linear(e_dim, 2)

    def forward(self, X):
        return F.softmax(self.output(X))

class BERTModel(nn.Module):

    def __init__( self, vocab_size, e_dim, transformer_h_dim, mlm_h_dim, n_heads, n_layers, max_len = 1024 ):
        '''
        :param vocab_size:  词汇数量
        :param e_dim: 词向量维度
        :param transformer_h_dim: transformer中间隐藏层的维度
        :param mlm_h_dim: mlm网络中间隐藏层维度
        :param n_heads: Transformer多头注意力的头数
        :param n_layers: Transformer编码层的层数
        :param max_len: 序列最长长度
        '''
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, e_dim, transformer_h_dim, n_heads, n_layers, max_len=max_len)

        self.mlm = MaskLM(vocab_size, mlm_h_dim, e_dim)
        self.nsp = NextSentencePred(e_dim)

    def forward(self, tokens, segments, pred_positions=None):
        encoded_X = self.encoder(tokens, segments)

        mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        nsp_Y_hat = self.nsp(encoded_X[:, 0, :])
        return encoded_X, mlm_Y_hat, nsp_Y_hat

if __name__ == '__main__':
    net = BERTModel(vocab_size=100, e_dim=768, transformer_h_dim=768, mlm_h_dim=768, n_heads=3, n_layers=12, max_len = 1024)
    batch_size = 24
    tokens = torch.randint(0,100,(batch_size,12))
    segments = torch.cat([torch.zeros(batch_size,7,dtype=int),torch.ones(batch_size,5,dtype=int)],dim=1)
    pred_positions = torch.randint(0,12,(batch_size,3))
    encoded_X, mlm_Y_hat, nsp_Y_hat = net(tokens,segments,pred_positions)

    print(encoded_X.shape)
    print(mlm_Y_hat.shape)
    print(nsp_Y_hat.shape)


