# package
import numpy as np
# import torch
from torch import nn
# from torch.nn import functional as F


class My_rnn_net(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(My_rnn_net, self).__init__()

        self.input_size = input_size #入力データ(x)
        self.hidden_dim = hidden_dim #隠れ層データ(hidden)
        self.n_layers = n_layers #RNNを「上方向に」何層重ねるか？の設定 ※横方向ではない
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size) #全結合層でhiddenからの出力を1個にする

    def forward(self, x):
        #h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(device)
        #y_rnn, h = self.rnn(x, h0)
        y_rnn, h = self.rnn(x, None) #hidden部分はコメントアウトした↑2行と同じ意味になっている。
        y = self.fc(y_rnn[:, -1, :]) #最後の時刻の出力だけを使用するので「-1」としている

        return y