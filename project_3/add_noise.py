import torch
import torch.nn as nn
import itertools
import pickle
import matplotlib.pyplot as plt
import imageio
from torch.autograd import Variable
import torch.nn.functional as F


MODEL_PATH="/mnt/hdd1/home/joycenerd/pytorch-MNIST-CelebA-GAN-DCGAN_2/CelebA_DCGAN_results/generator_param.pkl"
SAVEPATH="/mnt/hdd1/home/joycenerd/pytorch-MNIST-CelebA-GAN-DCGAN_2/CelebA_DCGAN_results/"

class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

def show_result(noise, show = False, save = False, path = SAVEPATH):
    if noise == 'normal(0,1)':
        z_=torch.empty((5*5,100)).normal_(mean=0,std=1.0).view(-1, 100, 1, 1)
    elif noise == 'normal(-10,1)':
        z_=torch.empty((5*5,100)).normal_(mean=-10,std=1.0).view(-1, 100, 1, 1)
    elif noise == 'uniform':
        z_ = torch.empty(5*5, 100)
        z_ = nn.init.uniform_(z_).view(-1, 100, 1, 1)


    z_ = Variable(z_.cuda(), volatile=True)

    G =  generator()
    # G = load_static_dict(MODEL_PATH)
    weight =  torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    G.load_state_dict(weight)
    G.to('cuda')
    # G.eval()
    
    test_images = G(z_)

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    # label = 'Epoch {0}'.format(num_epoch)
    if noise == 'normal(0,1)':
        label = 'normal: N(0,1)'
        path = SAVEPATH + 'normal_0_1.png'
    elif noise == 'normal(-10,1)':
        label = 'normal N(-10,1)'
        path = SAVEPATH + 'normal_-10_1.png'
    elif noise == 'uniform':
        label = 'uniform U(0,1)'
        path = SAVEPATH + 'uniform_0_1.png'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

print("yes!")
show_result('normal(0,1)')
show_result('normal(-10,1)')
show_result('uniform')