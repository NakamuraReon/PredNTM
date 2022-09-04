import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# cited : https://github.com/yuewang-cuhk/TAKG/blob/master/pykp/model.py
class NTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, topic_num, n_layers, device, l1_strength=0.001):
        super(NTM, self).__init__()
        self.input_dim = input_dim
        self.topic_num = topic_num
        self.fc11 = nn.Linear(self.input_dim, hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, topic_num)
        self.fc22 = nn.Linear(hidden_dim, topic_num)
        self.fcs = nn.Linear(self.input_dim, hidden_dim, bias=False)
        self.fcg1 = nn.Linear(topic_num, topic_num)
        self.fcg2 = nn.Linear(topic_num, topic_num)
        self.fcg3 = nn.Linear(topic_num, topic_num)
        self.fcg4 = nn.Linear(topic_num, topic_num)
        self.fcd1 = nn.Linear(topic_num, self.input_dim)
        self.l1_strength = torch.FloatTensor([l1_strength]).to(device)

        self.n_layers = n_layers #RNNを「上方向に」何層重ねるか？の設定 ※横方向ではない
        self.rnn = nn.RNN(topic_num, hidden_dim, self.n_layers, batch_first=True)
        self.rnn_fc = nn.Linear(hidden_dim, topic_num) #全結合層でhiddenからの出力を1個にする


    def encode(self, x):
        e1 = F.relu(self.fc11(x))
        e1 = F.relu(self.fc12(e1))
        e1 = e1.add(self.fcs(x))
        return self.fc21(e1), self.fc22(e1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate(self, h):
        g1 = torch.tanh(self.fcg1(h))
        g1 = torch.tanh(self.fcg2(g1))
        g1 = torch.tanh(self.fcg3(g1))
        g1 = torch.tanh(self.fcg4(g1))
        g1 = g1.add(h)
        return g1

    def decode(self, z):
        d1 = F.softmax(self.fcd1(z), dim=1)
        return d1

    def forward(self, x, sita):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)
        y_rnn, h = self.rnn(sita, None) #hidden部分はコメントアウトした↑2行と同じ意味になっている。
        y = self.fc(y_rnn[:, -1, :]) #最後の時刻の出力だけを使用するので「-1」としている
        return z, g, self.decode(g), mu, logvar

    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        
        print("Writing to %s" % fn)
        fw = open(fn, 'w')
        
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()