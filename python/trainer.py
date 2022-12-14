# standard
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
import model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.autograd.set_detect_anomaly(True)

def ntm_loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def pred_loss_function(sita, sita_hat):
    mse = nn.MSELoss()
    MSE = mse(sita, sita_hat)
    return MSE


def l1_penalty(para):
    return nn.L1Loss()(para, torch.zeros_like(para))


def check_sparsity(para, sparsity_threshold=1e-3):
    num_weights = para.shape[0] * para.shape[1]
    num_zero = (para.abs() < sparsity_threshold).sum().float()
    return num_zero / float(num_weights)


def update_l1(cur_l1, cur_sparsity, sparsity_target):
    diff = sparsity_target - cur_sparsity
    cur_l1.mul_(2.0 ** diff)

def compute_loss(model, dataloader, optimizer, epoch, last_batch_idx, sita_hat, target_sparsity=0.85):
    model.train()
    train_loss = 0
    for batch_idx, data_bow in enumerate(dataloader):
        data_bow = data_bow.to(device)
        # normalize data
        data_bow_norm = F.normalize(data_bow)
        optimizer.zero_grad()
        z, g, recon_batch, mu, logvar, sita, next_sita_hat = model(data_bow_norm, batch_idx, last_batch_idx)
        # print("sita:{}".format(sita))
        print("sita{}-batch{}".format(sita, batch_idx))
        print("next_sita_hat{}-batch{}".format(next_sita_hat, batch_idx))
        # if sita_hat == None:
        #     loss = ntm_loss_function(recon_batch, data_bow, mu, logvar)
        # elif batch_idx == last_batch_idx:
        #     # loss = ntm_loss_function(recon_batch, data_bow, mu, logvar) + pred_loss_function(sita, next_sita_hat)
        #     # loss = pred_loss_function(sita, next_sita_hat)
        #     loss = ntm_loss_function(recon_batch, data_bow, mu, logvar)
        # else:
        #     loss = ntm_loss_function(recon_batch, data_bow, mu, logvar)
        loss = ntm_loss_function(recon_batch, data_bow, mu, logvar)
        loss = loss + model.l1_strength * l1_penalty(model.fcd1.weight)
        loss.backward(retain_graph=True)
        train_loss += loss.item()
        optimizer.step()

    sparsity = check_sparsity(model.fcd1.weight.data)
    print("Overall sparsity = %.3f, l1 strength = %.5f" % (sparsity, model.l1_strength))
    print("Target sparsity = %.3f" % target_sparsity)
    update_l1(model.l1_strength, sparsity, target_sparsity)
    
    avg_loss = train_loss / len(dataloader.data)
    
    print('Train epoch: {} Average loss: {:.4f}'.format(
        epoch, avg_loss))

    return sparsity, avg_loss, sita, next_sita_hat


def compute_loss2(model, dataloader, optimizer, epoch, last_batch_idx, sita_hat, target_sparsity=0.85):
    model.train()
    # a = torch.tensor([[0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    train_loss = 0
    for batch_idx, data_bow in enumerate(dataloader):
        data_bow = data_bow.to(device)
        # normalize data
        data_bow_norm = F.normalize(data_bow)
        optimizer.zero_grad()
        z, g, recon_batch, mu, logvar, sita, next_sita_hat = model(data_bow_norm, batch_idx, last_batch_idx)
        if batch_idx == last_batch_idx:
            # sita = sita.reshape(-1,1)
            # sita_hat = sita_hat.reshape(-1,1)
            print(sita)
            print(next_sita_hat)
            if not sita_hat==None:
                # loss = pred_loss_function(sita, sita_hat)
                loss = pred_loss_function(sita, next_sita_hat)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
    return next_sita_hat

def compute_test_loss(model, dataloader, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(device)
            data_bow_norm = F.normalize(data_bow)
            z, _, recon_batch, mu, logvar, _, _ = model(data_bow_norm)
            test_loss += ntm_loss_function(recon_batch, data_bow, mu, logvar).item()
    avg_loss = test_loss / len(dataloader.data)
    print('Test epoch : {} Average loss: {:.4f}'.format(epoch, avg_loss))
    return z, avg_loss


def compute_perplexity(model, dataloader):   
    model.eval()
    loss = 0   
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(device)
            data_bow_norm = F.normalize(data_bow)      
            z, g, recon_batch, mu, logvar, _, _ = model(data_bow_norm)
            #loss += ntm_loss_function(recon_batch, data_bow, mu, logvar).detach()
            loss += F.binary_cross_entropy(recon_batch, data_bow, size_average=False)
    loss = loss / dataloader.word_count
    perplexity = np.exp(loss.cpu().numpy())
    return perplexity


def compute_z(model, dataloader):
    model.eval()
    computed_z = torch.tensor([])
    with torch.no_grad():
        for i, data_bow in enumerate(dataloader):
            data_bow = data_bow.to(device)
            data_bow_norm = F.normalize(data_bow)
            z, _, _, _, _, _, _ = model(data_bow_norm)
            computed_z = torch.cat((computed_z, z), 0)
    # print("computed_z: {}".format(computed_z))
    # print("computed_zsize: {}".format(len(computed_z)))
    # print("z: {}".format(len(z[0])))
    return computed_z


def lasy_predict(model, dataloader,vocab_dic, num_example=5, n_top_words=5):
    model.eval()
    docs, text = dataloader.bow_and_text()  
    docs, text = docs[:num_example], text[:num_example]
    docs_device = docs.to(device)
    docs_norm = F.normalize(docs_device)
    z, _, _, _, _, _, _ = model(docs_norm)
    z_a = z.detach().cpu().argmax(1).numpy()
    z = torch.softmax(z, dim=1).detach().cpu().numpy()
    beta_exp = model.fcd1.weight.data.cpu().numpy().T
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
    
    def bow_and_text(self):
        if self.index >= self.data_size:
            self.reset()        
        text = self.data[self.index: self.index + self.batch_size]
        batch = self.bow_data[self.index: self.index + self.batch_size]
        padded = self._pad(batch)
        tensor = torch.tensor(padded, dtype=torch.float, device=device)
        self.reset()
        return tensor, text


class Estimator:
    def __init__(self, input_dim, hidden_dim, topic_num, l1_strength=0.0000001, learning_rate = 0.001):
        # builing model and optimiser
        # set random seeds
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(123)
        torch.manual_seed(123)
        self.model = model.Model(input_dim, hidden_dim, topic_num, device, l1_strength)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.apply(init_weights)

    
    def fit(self, train_data, valid_data, bow_vocab, batch_size, last_batch_idx, sita_hat, period, n_epoch=200):
        logdir = "./topicwords/"
        dataloader = DataLoader(data = train_data, bow_vocab = bow_vocab, batch_size = batch_size)
        dataloader_valid = DataLoader(data = valid_data, bow_vocab = bow_vocab, batch_size = batch_size, shuffle=False)
        # Start Training
        for epoch in range(1, n_epoch + 1):
            print("======== Epoch", epoch, " ========")
            sparsity, avg_loss, sita, next_sita_hat = compute_loss(self.model, dataloader, self.optimizer, epoch, last_batch_idx, sita_hat)
            z, val_loss = compute_test_loss(self.model, dataloader_valid, epoch)
            
            pp = compute_perplexity(self.model, dataloader)
            pp_val = compute_perplexity(self.model, dataloader_valid)
            print("PP(train) = %.3f, PP(valid) = %.3f" % (pp, pp_val))

            if epoch % n_epoch == 0:
                self.model.print_topic_words(bow_vocab, os.path.join(logdir, str(period)+'-topwords_e%d.txt' % epoch))
                lasy_predict(self.model, dataloader_valid, bow_vocab, num_example=10, n_top_words=10)
        
        # z_train = compute_z(self.model, dataloader)
        # z_valid = compute_z(self.model, dataloader_valid)

        # return self.model, z_train, z_valid
        return sita, next_sita_hat