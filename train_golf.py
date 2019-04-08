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
from random import shuffle
from skimage.transform import resize


parser = ArgumentParser()
parser.add_argument(
    "-e", help="Epoch to train",
    type=int, default=10
)
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
    type=int, default=20
)
parser.add_argument(
    "-l", help="The value of sparsity regularizer [0.1]",
    type=float, default=0.1
)
parser.add_argument(
    "-c", help="The checkpoint file name",
    type=str, default="2019-04-04-01-09-10_1_4714"
)
parser.add_argument(
    "-s", help="Saving checkpoint file, every [1] epochs",
    type=int, default=1
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
    D = Discriminator().cuda()

    # Optimizer
    params_G = list(filter(lambda p: p.requires_grad, G.parameters()))
    optimizer_G = optim.Adam(params_G, lr=0.0002, betas=(0.5,0.999))
    params_D = list(filter(lambda p: p.requires_grad, D.parameters()))
    optimizer_D = optim.Adam(params_D, lr=0.0002, betas=(0.5,0.999))

    # Load pretrained
    if args.c is not None:
        G.load_state_dict(torch.load("./checkpoints/{}_G.pth".format(args.c)).state_dict(), strict=True)
        D.load_state_dict(torch.load("./checkpoints/{}_D.pth".format(args.c)).state_dict(), strict=True)
        print("Model restored")

    # Train
    # data file list
    f = open("./golf/golf.txt", 'r')
    data_lists = f.readlines()
    shuffle(data_lists)

    for epoch in range(args.e):
        for counter in range(int(len(data_lists)/args.nb)):
            # Data
            real_video = []
            i = 0
            while True:
                b = data_lists[counter*args.nb + i]
                i += 1

                img = np.asarray(PIL.Image.open(("./golf/frames-stable-many/{}").format(b.strip()))) / 127.5 - 1
                frames = []
                if img.shape[0] < 128*args.d[1]:
                    continue

                for f in range(args.d[1]):
                    frames.append( resize(img[f*128:(f+1)*128], (64,64), anti_aliasing=True)  )
                real_video.append( np.stack(frames, 0) )

                if len(real_video) >= args.nb:
                    break
            real_video = Variable(torch.from_numpy(np.stack(real_video, 0).astype(np.float32).transpose((0,4,1,2,3))), requires_grad=True).cuda()

            # D
            noise = torch.from_numpy(np.random.normal(0, 1, size=[args.nb,args.zd]).astype(np.float32)).cuda()
            with torch.no_grad():
                fake_video, f, b, m = G(noise)

            logit_real = D(real_video)
            logit_fake = D(fake_video.detach())

            prob_real = torch.mean(torch.sigmoid(logit_real))
            prob_fake = torch.mean(torch.sigmoid(logit_fake))

            loss_real = F.binary_cross_entropy_with_logits(logit_real, torch.ones_like(logit_real))
            loss_fake = F.binary_cross_entropy_with_logits(logit_fake, torch.zeros_like(logit_fake))
            loss_D = torch.mean(torch.stack([loss_real, loss_fake]))

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # G
            noise = torch.from_numpy(np.random.normal(0, 1, size=[args.nb,args.zd]).astype(np.float32)).cuda()
            gen_video, f, b, m = G(noise)

            logit_gen = D(gen_video)

            loss_gen = F.binary_cross_entropy_with_logits(logit_gen, torch.ones_like(logit_gen))
            loss_G = torch.mean(torch.stack([loss_gen])) + args.l*F.l1_loss(m, torch.zeros_like(m), True, True)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Print status
            print("Epoch {:d}/{:d} | Iter {:d}/{:d} | D {:.4e} | G {:.4e} | Real {:.2f} | Fake {:.2f}".format(epoch, args.e, counter, int(len(data_lists)/args.nb), loss_D, loss_G, prob_real, prob_fake))

            process_and_write_video(gen_video[0:1].cpu().data.numpy(), "curr_video")
            process_and_write_image(b.cpu().data.numpy(), "curr_bg")

        if (epoch+1) % args.s == 0:
            process_and_write_video(gen_video[0:1].cpu().data.numpy(), "epoch{}_iter{}_video".format(epoch, counter))
            process_and_write_image(b.cpu().data.numpy(), "epoch{}_iter{}_bg".format(epoch, counter))
                
            curr_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            torch.save(G, "./checkpoints/{}_{}_{}_G.pth".format(curr_time, epoch, counter))
            torch.save(D, "./checkpoints/{}_{}_{}_D.pth".format(curr_time, epoch, counter))
            print ("Checkpoints saved")


if __name__ == '__main__':
    main()
