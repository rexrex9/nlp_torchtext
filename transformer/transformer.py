
import torch
from torch import nn
from torch.autograd import Variable

#多头注意力层
class MultiHeadAttentionLayer( nn.Module ):

    def __init__( self, e_dim, h_dim, n_heads ):
        '''
        :param e_dim: 输入的向量维度
        :param h_dim: 每个单头注意力层输出的向量维度
        :param n_heads: 头数
        '''
        super().__init__()
        self.atte_layers = nn.ModuleList([ OneHeadAttention( e_dim, h_dim ) for _ in range( n_heads ) ] )
        self.l = nn.Linear( h_dim * n_heads, e_dim)

    def forward( self, seq_inputs, querys = None, mask = None ):
        outs = []
        for one in self.atte_layers:
            out = one( seq_inputs, querys, mask )
            outs.append( out )
        # [ batch, seq_lens, h_dim * n_heads ]
        outs = torch.cat( outs, dim=-1 )
        # [ batch, seq_lens, e_dim ]
        outs = self.l( outs )
        return outs

#单头注意力层
class OneHeadAttention( nn.Module ):

    def __init__( self, e_dim, h_dim ):
        '''
        :param e_dim: 输入向量维度
        :param h_dim: 输出向量维度
        '''
        super().__init__()
        self.h_dim = h_dim

        # 初始化Q,K,V的映射线性层
        self.lQ = nn.Linear( e_dim, h_dim )
        self.lK = nn.Linear( e_dim, h_dim )
        self.lV = nn.Linear( e_dim, h_dim )

    def forward( self, seq_inputs , querys = None, mask = None ):
        '''
        :param seq_inputs: #[ batch, seq_lens, e_dim ]
        :param querys: #[ batch, seq_lens, e_dim ]
        :param mask: #[ 1, seq_lens, seq_lens ]
        :return:
        '''
        # 如果有encoder的输出, 则映射该张量，否则还是就是自注意力的逻辑
        if querys is not None:
            Q = self.lQ( querys ) #[ batch, seq_lens, h_dim ]
        else:
            Q =  self.lQ( seq_inputs ) #[ batch, seq_lens, h_dim ]
        K = self.lK( seq_inputs ) #[ batch, seq_lens, h_dim ]
        V = self.lV( seq_inputs ) #[ batch, seq_lens, h_dim ]
        # [ batch, seq_lens, seq_lens ]
        QK = torch.matmul( Q,K.permute( 0, 2, 1 ) )
        # [ batch, seq_lens, seq_lens ]
        QK /= ( self.h_dim ** 0.5 )

        # 将对应Mask序列中0的位置变为-1e9,意为遮盖掉此处的值
        if mask is not None:
            QK = QK.masked_fill( mask == 0, -1e9 )
        # [ batch, seq_lens, seq_lens ]
        a = torch.softmax( QK, dim = -1 )
        # [ batch, seq_lens, h_dim ]
        outs = torch.matmul( a, V )
        return outs

#前馈神经网络
class FeedForward(nn.Module):

    def __init__( self, e_dim, ff_dim, drop_rate = 0.1 ):
        super().__init__()
        self.l1 = nn.Linear( e_dim, ff_dim )
        self.l2 = nn.Linear( ff_dim, e_dim )
        self.drop_out = nn.Dropout( drop_rate )

    def forward( self, x ):
        outs = self.l1( x )
        outs = self.l2( self.drop_out( torch.relu( outs ) ) )
        return outs

#位置编码
class PositionalEncoding( nn.Module ):

    def __init__( self, e_dim, dropout = 0.1, max_len = 512 ):
        super().__init__()
        self.dropout = nn.Dropout( p = dropout )
        pe = torch.zeros( max_len, e_dim )
        position = torch.arange( 0, max_len ).unsqueeze( 1 )

        div_term = 10000.0 ** ( torch.arange( 0, e_dim, 2 ) / e_dim )

        #偶数位计算sin, 奇数位计算cos
        pe[ :, 0::2 ] = torch.sin( position / div_term )
        pe[ :, 1::2 ] = torch.cos( position / div_term )

        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward( self, x ):
        x = x + Variable( self.pe[:, : x.size( 1 ) ], requires_grad = False )
        return self.dropout( x )

#编码层
class EncoderLayer(nn.Module):

    def __init__( self, e_dim, h_dim, n_heads, drop_rate = 0.1 ):
        '''
        :param e_dim: 输入向量的维度
        :param h_dim: 注意力层中间隐含层的维度
        :param n_heads: 多头注意力的头目数量
        :param drop_rate: drop out的比例
        '''
        super().__init__()
        # 初始化多头注意力层
        self.attention = MultiHeadAttentionLayer( e_dim, h_dim, n_heads )
        # 初始化注意力层之后的LN
        self.a_LN = nn.LayerNorm( e_dim )
        # 初始化前馈神经网络层
        self.ff_layer = FeedForward( e_dim, e_dim//2 )
        # 初始化前馈网络之后的LN
        self.ff_LN = nn.LayerNorm( e_dim )

        self.drop_out = nn.Dropout( drop_rate )

    def forward(self, seq_inputs ):
        # seq_inputs = [batch, seqs_len, e_dim]
        # 多头注意力, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.attention( seq_inputs )
        # 残差连接与LN, 输出维度[ batch, seq_lens, e_dim ]
        outs = self.a_LN( seq_inputs + self.drop_out( outs_ ) )
        # 前馈神经网络, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.ff_layer( outs )
        # 残差与LN, 输出维度[ batch, seq_lens, e_dim ]
        outs = self.ff_LN( outs + self.drop_out( outs_) )
        return outs


class TransformerEncoder(nn.Module):

    def __init__(self, e_dim, h_dim, n_heads, n_layers, drop_rate = 0.1 ):
        '''
        :param e_dim: 输入向量的维度
        :param h_dim: 注意力层中间隐含层的维度
        :param n_heads: 多头注意力的头目数量
        :param n_layers: 编码层的数量
        :param drop_rate: drop out的比例
        '''
        super().__init__()
        #初始化位置编码层
        self.position_encoding = PositionalEncoding( e_dim )
        #初始化N个“编码层”
        self.encoder_layers = nn.ModuleList( [EncoderLayer( e_dim, h_dim, n_heads, drop_rate )
                                         for _ in range( n_layers )] )
    def forward( self, seq_inputs ):
        '''
        :param seq_inputs: 已经经过Embedding层的张量，维度是[ batch, seq_lens, dim ]
        :return: 与输入张量维度一样的张量，维度是[ batch, seq_lens, dim ]
        '''
        #先进行位置编码
        seq_inputs = self.position_encoding( seq_inputs )
        #输入进N个“编码层”中开始传播
        for layer in self.encoder_layers:
            seq_inputs = layer( seq_inputs )
        return seq_inputs


#生成mask序列
def subsequent_mask( size ):
    subsequent_mask = torch.triu( torch.ones( (1, size, size) ) ) == 0
    return subsequent_mask


#解码层
class DecoderLayer(nn.Module):

    def __init__( self, e_dim, h_dim, n_heads, drop_rate = 0.1 ):
        '''
        :param e_dim: 输入向量的维度
        :param h_dim: 注意力层中间隐含层的维度
        :param n_heads: 多头注意力的头目数量
        :param drop_rate: drop out的比例
        '''
        super().__init__()

        # 初始化自注意力层
        self.self_attention = MultiHeadAttentionLayer( e_dim, h_dim, n_heads )
        # 初始化自注意力层之后的LN
        self.sa_LN = nn.LayerNorm( e_dim )
        # 初始化交互注意力层
        self.interactive_attention = MultiHeadAttentionLayer( e_dim, h_dim, n_heads )
        # 初始化交互注意力层之后的LN
        self.ia_LN = nn.LayerNorm (e_dim )
        # 初始化前馈神经网络层
        self.ff_layer = FeedForward( e_dim, e_dim//2 )
        # 初始化前馈网络之后的LN
        self.ff_LN = nn.LayerNorm( e_dim )

        self.drop_out = nn.Dropout( drop_rate )

    def forward( self, seq_inputs , querys, mask ):
        '''
        :param seq_inputs: [ batch, seqs_len, e_dim ]
        :param querys: encoder的输出
        :param mask: 遮盖位置的标注序列 [ 1, seqs_len, seqs_len ]
        '''
        # 自注意力层, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.self_attention( seq_inputs , mask=mask )
        # 残差连与LN, 输出维度[ batch, seq_lens, e_dim ]
        outs = self.sa_LN( seq_inputs + self.drop_out( outs_ ) )
        # 交互注意力层, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.interactive_attention( outs, querys )
        # 残差连与LN, 输出维度[ batch, seq_lens, e_dim
        outs = self.ia_LN( outs + self.drop_out(outs_) )
        # 前馈神经网络, 输出维度[ batch, seq_lens, e_dim ]
        outs_ = self.ff_layer( outs )
        # 残差与LN, 输出维度[ batch, seq_lens, e_dim ]
        outs = self.ff_LN( outs + self.drop_out( outs_) )
        return outs



class TransformerDecoder(nn.Module):
    def __init__(self, e_dim, h_dim, n_heads, n_layers, n_classes,drop_rate = 0.1 ):
        '''
        :param e_dim: 输入向量的维度
        :param h_dim: 注意力层中间隐含层的维度
        :param n_heads: 多头注意力的头目数量
        :param n_layers: 解码层的数量
        :param n_classes: 类别数
        :param drop_rate: drop out的比例
        '''
        super().__init__()
        # 初始化位置编码层
        self.position_encoding = PositionalEncoding( e_dim )
        # 初始化N个“解码层”
        self.decoder_layers = nn.ModuleList( [DecoderLayer( e_dim, h_dim, n_heads, drop_rate )
                                         for _ in range( n_layers )] )
        # 线性层
        self.linear = nn.Linear(e_dim,n_classes)
        # softmax激活函数
        self.softmax = nn.Softmax()


    def forward( self, seq_inputs, querys ):
        '''
        :param seq_inputs: 已经经过Embedding层的张量，维度是[ batch, seq_lens, dim ]
        :param querys: encoder的输出，维度是[ batch, seq_lens, dim ]
        :return: 与输入张量维度一样的张量，维度是[ batch, seq_lens, dim ]
        '''
        # 先进行位置编码
        seq_inputs = self.position_encoding( seq_inputs )
        # 得到mask序列
        mask = subsequent_mask( seq_inputs.shape[1] )
        # 输入进N个“解码层”中开始传播
        for layer in self.decoder_layers:
            seq_inputs = layer( seq_inputs, querys, mask )
        # 最终线性变化后Softmax归一化
        seq_outputs = self.softmax(self.linear(seq_inputs))
        return seq_outputs

class Transformer(nn.Module):

    def __init__(self,e_dim, h_dim, n_heads, n_layers, n_classes,drop_rate=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(e_dim, h_dim, n_heads, n_layers, drop_rate)
        self.decoder = TransformerDecoder(e_dim, h_dim, n_heads, n_layers, n_classes,drop_rate)

    def forward(self,input,output):
        querys = self.encoder(input)
        pred_seqs = self.decoder(output,querys)
        return pred_seqs



if __name__ == '__main__':
    input = torch.randn( 5, 3, 12 )
    net = Transformer(12, 8, 3, 6, 20)
    output = torch.randn( 5, 3, 12)
    pred_seqs = net(input,output)

    print(pred_seqs.shape)