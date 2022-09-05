import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, topic_num, device, l1_strength=0.001):
        super(Model, self).__init__()
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
        self.sita = None
        self.cat_z = torch.tensor([])
        self.sita_hat = None
        self.l1_strength = torch.FloatTensor([l1_strength]).to(device)
        self.fcp1 = nn.Linear(topic_num, topic_num)
        self.fcp2 = nn.Linear(topic_num, topic_num)
    
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

    def forward(self, x, batch_idx, last_batch_idx):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        g = self.generate(z)  
        if batch_idx == 0:
            self.cat_z = z
            # print("{}-{}".format(period, batch_idx))
        elif batch_idx == last_batch_idx:
            self.cat_z = torch.cat((self.cat_z, z), 0)
            sum_z = sum(self.cat_z)
            self.sita = torch.softmax(sum_z, dim=0)
            self.sita = torch.reshape(self.sita, (1, 15))
            self.sita_hat = self.fcp1(self.sita)
            # print("{}-{}-elif".format(period, batch_idx))
        else:
            self.cat_z = torch.cat((self.cat_z, z), 0)
            # print("{}-{}-else".format(period, batch_idx))
        return z, g, self.decode(g), mu, logvar, self.sita, self.sita_hat

    def print_topic_words(self, vocab_dic, fn, n_top_words=10):
        beta_exp = self.fcd1.weight.data.cpu().numpy().T
        print("Writing to %s" % fn)
        fw = open(fn, 'w')
        for k, beta_k in enumerate(beta_exp):
            topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
            
            print('Topic {}: {}'.format(k, ' '.join(topic_words)))
            
            fw.write('{}\n'.format(' '.join(topic_words)))
        fw.close()