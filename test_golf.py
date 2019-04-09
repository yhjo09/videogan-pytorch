"""
videogan PT version

Original torch ver from https://github.com/cvondrick/videogan
TF ver from https://github.com/GV1028/videogan
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os
import scipy.misc
import numpy as np
import glob
from utils import *
import sys
from argparse import ArgumentParser
from datetime import datetime


parser = ArgumentParser()
parser.add_argument(
    "-d", help="The dimension of each video, must be of shape [3,32,64,64]",
    nargs='*', default=[3,32,64,64]
)
parser.add_argument(
    "-zd", help="The dimension of latent vector [100]",
    type=int, default=100
)
parser.add_argument(
    "-nb", help="The size of batch images [64]",
    type=int, default=64
)
parser.add_argument(
    "-c", help="The checkpoint file name",
    type=str, default="2019-04-08-21-58-34_0_30187"
)
args = parser.parse_args()



class Generator(torch.nn.Module):
    def __init__(self, zdim=args.zd):
        super(Generator, self).__init__()
        
        self.zdim = zdim
        
        # Background
        self.conv1b = nn.ConvTranspose2d(zdim, 512, [4,4], [1,1])
        self.bn1b = nn.BatchNorm2d(512)

        self.conv2b = nn.ConvTranspose2d(512, 256, [4,4], [2,2], [1,1])
        self.bn2b = nn.BatchNorm2d(256)

        self.conv3b = nn.ConvTranspose2d(256, 128, [4,4], [2,2], [1,1])
        self.bn3b = nn.BatchNorm2d(128)

        self.conv4b = nn.ConvTranspose2d(128, 64, [4,4], [2,2], [1,1])
        self.bn4b = nn.BatchNorm2d(64)

        self.conv5b = nn.ConvTranspose2d(64, 3, [4,4], [2,2], [1,1])

        # Foreground
        self.conv1 = nn.ConvTranspose3d(zdim, 512, [2,4,4], [1,1,1])
        self.bn1 = nn.BatchNorm3d(512)

        self.conv2 = nn.ConvTranspose3d(512, 256, [4,4,4], [2,2,2], [1,1,1])
        self.bn2 = nn.BatchNorm3d(256)

        self.conv3 = nn.ConvTranspose3d(256, 128, [4,4,4], [2,2,2], [1,1,1])
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.ConvTranspose3d(128, 64, [4,4,4], [2,2,2], [1,1,1])
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.ConvTranspose3d(64, 3, [4,4,4], [2,2,2], [1,1,1])

        # Mask
        self.conv5m = nn.ConvTranspose3d(64, 1, [4,4,4], [2,2,2], [1,1,1])

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif classname.lower().find('bn') != -1:
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # Background
        b = F.relu(self.bn1b(self.conv1b(z.unsqueeze(2).unsqueeze(3))))
        b = F.relu(self.bn2b(self.conv2b(b)))
        b = F.relu(self.bn3b(self.conv3b(b)))
        b = F.relu(self.bn4b(self.conv4b(b)))
        b = torch.tanh(self.conv5b(b)).unsqueeze(2)  # b, 3, 1, 64, 64

        # Foreground
        f = F.relu(self.bn1(self.conv1(z.unsqueeze(2).unsqueeze(3).unsqueeze(4))))
        f = F.relu(self.bn2(self.conv2(f)))
        f = F.relu(self.bn3(self.conv3(f)))
        f = F.relu(self.bn4(self.conv4(f)))
        m = torch.sigmoid(self.conv5m(f))   # b, 1, 32, 64, 64
        f = torch.tanh(self.conv5(f))   # b, 3, 32, 64, 64
        
        out = m*f + (1-m)*b

        return out, f, b, m


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, [4,4,4], [2,2,2], [1,1,1])

        self.conv2 = nn.Conv3d(64, 128, [4,4,4], [2,2,2], [1,1,1])
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, [4,4,4], [2,2,2], [1,1,1])
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, [4,4,4], [2,2,2], [1,1,1])
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(512, 1, [2,4,4], [1,1,1])

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif classname.lower().find('bn') != -1:
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.conv5(x)

        return x



def main():
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not os.path.exists("./genvideos"):
        os.makedirs("./genvideos")


    # Model def
    G = Generator(zdim=args.zd).cuda()

    # Load pretrained
    if args.c is not None:
        G.load_state_dict(torch.load("./checkpoints/{}_G.pth".format(args.c)).state_dict(), strict=True)
        print("Model restored")

    # G
    noise = torch.from_numpy(np.random.normal(0, 1, size=[args.nb,args.zd]).astype(np.float32)).cuda()
    gen_video, f, b, m = G(noise)

    # Save results
    for i in range(args.nb):
        process_and_write_video(gen_video[i:i+1].cpu().data.numpy(), "sample{:02d}_video".format(i))
        process_and_write_image(b.cpu().data.numpy(), "sample{:02d}_bg".format(i))
    print ("Results saved")


if __name__ == '__main__':
    main()
