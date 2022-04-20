from bert import model
from bert import dataprocess as dp
import torch
from torch.utils.data import DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self,tokenss, segmentss, mlm_pred_positionss, nsp_Y, mlm_Y):
        self.tokenss = torch.LongTensor(tokenss)
        self.segmentss = torch.LongTensor(segmentss)
        self.mlm_pred_positionss= torch.LongTensor(mlm_pred_positionss)
        self.nsp_Y = torch.LongTensor(nsp_Y)
        self.mlm_Y = torch.LongTensor(mlm_Y)

    def __getitem__(self, idx):
        return (self.tokenss[idx], self.segmentss[idx],
                self.mlm_pred_positionss[idx], self.nsp_Y[idx],
                self.mlm_Y[idx])

    def __len__(self):
        return len(self.tokenss)


def train(epochs = 10,batchSize=2):
    tokenss, segmentss, mlm_pred_positionss, nsp_Y, mlm_Y, vocab_dict = dp.getPreData(dp.seqs)

    dataSet = Dataset(tokenss, segmentss, mlm_pred_positionss, nsp_Y, mlm_Y)
    net = model.BERTModel(vocab_size=len(vocab_dict), e_dim=32, transformer_h_dim=32,
                          mlm_h_dim=32, n_heads=3, n_layers=12, max_len=128)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for e in range(epochs):
        for tokenss, segmentss, mlm_pred_positionss, nsp_Y, mlm_Y in DataLoader(dataSet, batch_size=batchSize, shuffle=True):
            optimizer.zero_grad()
            encoded_X, mlm_Y_hat, nsp_Y_hat = net(tokenss, segmentss, mlm_pred_positionss)
            mlm_Y_hat = mlm_Y_hat.reshape(-1,len(vocab_dict))
            mlm_Y = mlm_Y.reshape(-1)
            mlm_loss = criterion(mlm_Y_hat,mlm_Y)
            nsp_loss = criterion(nsp_Y_hat,nsp_Y)
            loss = mlm_loss + nsp_loss
            loss.backward()
            optimizer.step()
        print('epoch {}, loss = {:.4f}'.format(e,loss))

if __name__ == '__main__':
    '''
        该示例代码未考虑padding,如要考虑padding则用<pad>填充，并记录valid_lens(实际长度)方便并行计算。
    '''
    train()