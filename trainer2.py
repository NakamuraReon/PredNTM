# standard
import pandas as pd
import numpy as np
import random
import os


# torch
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
# sklearn
from sklearn.utils import shuffle
import ntm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
# BCE
def ntm_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def pred_loss_function(pred_sita, sita):
    MSE = nn.MSELoss(pred_sita, sita)
    return MSE

# なぜL1正則化が使われているのか
# https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html
def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))

def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)

def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)

def compute_loss(ntm_model, dataloader, optimizer, epoch, target_sparsity=0.85):
    ntm_model.train()
    train_loss = 0
    
    for batch_idx, data_bow in enumerate(dataloader):
        data_bow = data_bow.to(device)
        
        # normalize data
        data_bow_norm = F.normalize(data_bow)
        optimizer.zero_grad()
        
        z, g, recon_batch, mu, logvar = ntm_model(data_bow_norm)
        
        loss = ntm_loss_function(recon_batch, data_bow, mu, logvar)
        loss = loss + ntm_model.l1_strength * l1_penalty(ntm_model.fcd1.weight)
        loss.backward() # https://zenn.dev/hirayuki/articles/bbc0eec8cd816c183408
        train_loss += loss.item()
        optimizer.step()
    
    sparsity = check_sparsity(ntm_model.fcd1.weight.data)
    print("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, ntm_model.l1_strength))
    print("Target sparsity = %.3f" % target_sparsity)
    update_l1(ntm_model.l1_strength, sparsity, target_sparsity)
    
    avg_loss = train_loss / len(dataloader.data)
    
    print('Train epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss))
    
    return sparsity, avg_loss

def compute_test_loss(ntm_model, dataloader, epoch):
    ntm_model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(device)
            data_bow_norm = F.normalize(data_bow)

            z, _, recon_batch, mu, logvar = ntm_model(data_bow_norm)
            test_loss += ntm_loss_function(recon_batch, data_bow, mu, logvar).item()

    avg_loss = test_loss / len(dataloader.data)
    print('Test epoch : {} Average loss: {:.4f}'.format(epoch, avg_loss))
    return z, avg_loss


def compute_perplexity(ntm_model, dataloader):
    
    ntm_model.eval()
    loss = 0
    
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(device)
            data_bow_norm = F.normalize(data_bow)
            
            z, g, recon_batch, mu, logvar = ntm_model(data_bow_norm)
            
            #loss += ntm_loss_function(recon_batch, data_bow, mu, logvar).detach()
            loss += F.binary_cross_entropy(recon_batch, data_bow, size_average=False)
            
    loss = loss / dataloader.word_count
    perplexity = np.exp(loss.cpu().numpy())
    
    return perplexity


def compute_z(ntm_model, dataloader):
    ntm_model.eval()
    computed_z = torch.tensor([])
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(device)
            data_bow_norm = F.normalize(data_bow)
            z, _, _, _, _ = ntm_model(data_bow_norm)
            computed_z = torch.cat((computed_z, z), 0)
    return computed_z


def lasy_predict(ntm_model, dataloader,vocab_dic, num_example=5, n_top_words=5):
    ntm_model.eval()
    docs, text = dataloader.bow_and_text()
    
    docs, text = docs[:num_example], text[:num_example]
    
    docs_device = docs.to(device)
    docs_norm = F.normalize(docs_device)
    z, _, _, _, _ = ntm_model(docs_norm)
    z_a = z.detach().cpu().argmax(1).numpy()
    z = torch.softmax(z, dim=1).detach().cpu().numpy()
    
    beta_exp = ntm_model.fcd1.weight.data.cpu().numpy().T
    topics = []
    for k, beta_k in enumerate(beta_exp):
        topic_words = [vocab_dic[w_id] for w_id in np.argsort(beta_k)[:-n_top_words - 1:-1]]
        topics.append(topic_words)
    
    for i, (zi, _z_a, t) in enumerate(zip(z, z_a, text)):
        
        print('\n===== # {}, Topic : {}, p : {:.4f} %'.format(i+1, _z_a,  zi[_z_a] * 100))
        print("Topic words :", ', '.join(topics[_z_a]))
        print("Input :", ' '.join(t))
        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform(m.weight)



class DataLoader(object):
    def __init__(self, data, bow_vocab, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.bow_vocab = bow_vocab
        
        self.index = 0
        self.pointer = np.array(range(len(data)))
        
        self.data = np.array(data)
        self.bow_data = np.array([bow_vocab.doc2bow(s) for s in data])
        
        # counting total word number
        word_count = []
        for bow in self.bow_data:
            wc = 0
            for (i, c) in bow:
                wc += c
            word_count.append(wc)
        
        self.word_count = sum(word_count)
        self.data_size = len(data)
        
        self.shuffle = shuffle
        self.reset()

    
    def reset(self):
        if self.shuffle:
            self.pointer = shuffle(self.pointer)
        self.index = 0 
    
    
    # transform bow data into (1 x V) size vector.
    def _pad(self, batch):
        bow_vocab = len(self.bow_vocab)
        res_src_bow = np.zeros((len(batch), bow_vocab))
        
        for idx, bow in enumerate(batch):
            bow_k = [k for k, v in bow]
            bow_v = [v for k, v in bow]
            res_src_bow[idx, bow_k] = bow_v
            
        return res_src_bow
    
    def __iter__(self):
        return self

    def __next__(self):
        
        if self.index >= self.data_size:
            self.reset()
            raise StopIteration()
            
        ids = self.pointer[self.index: self.index + self.batch_size]
        batch = self.bow_data[ids]
        padded = self._pad(batch)
        tensor = torch.tensor(padded, dtype=torch.float, device=device)
        
        self.index += self.batch_size

        return tensor
    
    # for NTM.lasy_predict()
    def bow_and_text(self):
        if self.index >= self.data_size:
            self.reset()
            
        text = self.data[self.index: self.index + self.batch_size]
        batch = self.bow_data[self.index: self.index + self.batch_size]
        padded = self._pad(batch)
        tensor = torch.tensor(padded, dtype=torch.float, device=device)
        self.reset()

        return tensor, text




class NTMEstimator:
    def __init__(self, input_dim, hidden_dim, topic_num, l1_strength=0.0000001, learning_rate = 0.001):
        # builing model and optimiser
        # set random seeds
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(123)
        torch.manual_seed(123)
        self.ntm_model = ntm.NTM(input_dim, hidden_dim, topic_num, device, l1_strength)
        self.optimizer = optim.Adam(self.ntm_model.parameters(), lr=learning_rate)
        self.ntm_model.apply(init_weights)
    
    def fit(self, train_data, valid_data, bow_vocab, batch_size, n_epoch=200):
        writer = SummaryWriter()
        logdir = "./"
        # building dataloader
        dataloader = DataLoader(data = train_data, bow_vocab = bow_vocab, batch_size = batch_size)
        dataloader_valid = DataLoader(data = valid_data, bow_vocab = bow_vocab, batch_size = batch_size, shuffle=False)
        
        # Start Training
        for epoch in range(1, n_epoch + 1):
            print("======== Epoch", epoch, " ========")
            sparsity, train_loss = compute_loss(self.ntm_model, dataloader, self.optimizer, epoch)
            z, val_loss = compute_test_loss(self.ntm_model, dataloader_valid, epoch)
            
            pp = compute_perplexity(self.ntm_model, dataloader)
            pp_val = compute_perplexity(self.ntm_model, dataloader_valid)
            print("PP(train) = %.3f, PP(valid) = %.3f" % (pp, pp_val))
            
            writer.add_scalars('scalar/loss',{'train_loss': train_loss,'valid_loss': val_loss},epoch)
            writer.add_scalars('scalar/perplexity',{'train_pp': pp,'valid_pp': pp_val},epoch)
            writer.add_scalars('scalar/sparsity',{'sparsity': sparsity},epoch)
            writer.add_scalars('scalar/l1_strength',{'l1_strength': self.ntm_model.l1_strength},epoch)

            if epoch % 50 == 0:
                self.ntm_model.print_topic_words(bow_vocab, os.path.join(logdir, 'topwords_e%d.txt' % epoch))
                lasy_predict(self.ntm_model, dataloader_valid, bow_vocab, num_example=10, n_top_words=10)
        writer.close()
        
        z_train = compute_z(self.ntm_model, dataloader)
        z_valid = compute_z(self.ntm_model, dataloader_valid)

        return self.ntm_model, z_train, z_valid
