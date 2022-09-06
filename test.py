import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch import optim


# 線形回帰モデルの定義                                                          
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        y = self.layer(x)
        return y

# ここからmain                                                                  
if __name__ == '__main__':

    # GPUかCPUかを自動設定                                                      
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # modelとoptimizerの定義                                                    
    model = LinearRegression().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01)

    # data生成                                                                  
    n = 1000
    x = torch.rand(n)*2-1
    a, b = 2.0, -10.0 # weight & bias                                           
    y = a*x+b

    # dataにノイズ追加                                                          
    x = x + torch.randn(n)*0.02
    y = y + a*torch.randn(n)*0.02

    # to GPU                                                                    
    x = x.to(device)
    y = y.to(device)

    bs = 10 # batch_size                                                        
    niter = 1000
    losses = []
    for iiter in range(niter):

        # batch dataの取得                                                      
        r = np.random.choice(n, bs, replace=False)
        bx = x[r].reshape(-1,1)
        by = y[r].reshape(-1,1)
        print(by)
        # forwardとloss計算                                                     
        y_ = model.forward(bx)
        mse = nn.MSELoss()
        # loss = torch.mean((y_ - by)**2)
        loss = mse(y_, by)
        # 最適化                                                                
        opt.zero_grad() # 勾配初期化                                            
        loss.backward() # 勾配計算(backward)                                    
        opt.step() # パラメータ更新                                             

        # print('%05d/%05d loss=%.5f' % (iiter, niter, loss.item()))
        losses.append(loss.item())

    # 重みの取り出し                                                            
    a_ = model.layer.weight.detach().to('cpu').numpy()
    b_ = model.layer.bias.detach().to('cpu').numpy()
    print('a=%.3f b=%.3f' % (a_[0] ,b_[0]))

    # データと最適化した関数のplot
    xnp = x.detach().to('cpu').numpy()             
    ynp = y.detach().to('cpu').numpy()
    # plt.scatter(xnp, ynp)
    # x = np.linspace(-1,1,100)
    # y = a_[0]*x + b_[0]
    # plt.plot(x, y, c='r')
    # plt.savefig('output.png')
    # plt.tight_layout();plt.show()
