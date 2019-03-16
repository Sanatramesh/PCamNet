import os
import sys
import time
import datetime
import argparse
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
from collections import OrderedDict

import torch as th
from torch.autograd import Variable
from torchvision import transforms, utils
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader


class TDNet_VGG11_Model(th.nn.Module):

    def __init__(self, freeze_encoder = False):
        super(TDNet_VGG11_Model, self).__init__()
        self.vgg_encoder = th.nn.Sequential(
                            th.nn.Conv2d(3, 64, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.Conv2d(64, 64, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.MaxPool2d(2, stride = 2),

                            th.nn.Conv2d(64, 128, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.Conv2d(128, 128, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.MaxPool2d(2, stride = 2),

                            th.nn.Conv2d(128, 256, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.Conv2d(256, 256, 3, padding = 1),
                            th.nn.ReLU(),
                            th.nn.MaxPool2d(2, stride = 2)
                        )

        # state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
        # self.vgg_encoder.load_state_dict(state_dict, strict = False)

        if freeze_encoder:
            for param in self.vgg_encoder.parameters():
                param.requires_grad = False

        self.decoder = th.nn.Sequential(
                        th.nn.Conv2d(256, 256, 3, padding = 1),
                        th.nn.ReLU(),
                        th.nn.ConvTranspose2d(256, 256, 4, stride = 2, padding = 1),

                        th.nn.Conv2d(256, 128, 3, padding = 1),
                        th.nn.ReLU(),
                        th.nn.ConvTranspose2d(128, 128, 4, stride = 2, padding = 1),

                        th.nn.Conv2d(128, 64, 3, padding = 1),
                        th.nn.ReLU(),
                        th.nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),

                        th.nn.Conv2d(64, 1, 3, padding = 1),
                        th.nn.ReLU()
                    )

    def forward(self, img):
        features = self.vgg_encoder(img)
        y = self.decoder(features)
        return y

    def get_name(self):
        return 'TDNet with VGG11 encoder'


class TDNet(object):

    def __init__(self, input_dims, output_dims, learning_rate = 1e-4):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate

    def build_model(self):
        self.model = TDNet_VGG11_Model()
        print (self.model)

    def loss_func(self, y, y_):
        loss = th.mean((y - y_) ** 2)
        return loss

    def add_optimizer(self):
        self.optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.learning_rate)

    def train_batch(self, data):
        left_cam_data, right_cam_data, disp_data = data
        left_cam_data, right_cam_data, disp_data = th.autograd.Variable(th.FloatTensor(left_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(right_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(disp_data))

        y = self.model.forward( left_cam_data, right_cam_data )
        print (y.shape, disp_data.shape)
        loss = self.loss_func( y, disp_data )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def compute_loss(self, data):
        left_cam_data, right_cam_data, disp_data = data
        left_cam_data, right_cam_data, disp_data = th.autograd.Variable(th.FloatTensor(left_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(right_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(disp_data))

        y = self.model.forward( left_cam_data, right_cam_data )
        loss = self.loss_func( y, disp_data )

        return loss.data

    def forward_pass(self, data):
        left_cam_data, right_cam_data, disp_data = data
        left_cam_data, right_cam_data, disp_data = th.autograd.Variable(th.FloatTensor(left_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(right_cam_data)), \
                                                    th.autograd.Variable(th.FloatTensor(disp_data))

        y = self.model.forward( left_cam_data, right_cam_data )
        disparity_map = y.data

        return disparity_map

    def save_model(self, ckpt_file):
        # th.save( self.model, ckpt_file )
        self.model.save_state_dict(ckpt_file)

    def load_model(self, ckpt_file = None):
        if ckpt_file != None:
            self.model.load_state_dict(th.load(ckpt_file))

    def get_name(self):
        return self.model.get_name()
