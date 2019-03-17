import numpy as np
import matplotlib.pyplot as plt

import torch as th
import torch.utils.model_zoo as model_zoo

import copy_reg
import types
import multiprocessing

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


class PCamNetVGGModel(th.nn.Module):

    def __init__(self, num_classes=10, freeze_encoder = False):
        super(PCamNetVGGModel, self).__init__()
        self.vgg_encoder = th.nn.Sequential(
                            th.nn.Conv2d(3, 64, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.MaxPool2d(kernel_size=2, stride=2),

                            th.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.MaxPool2d(kernel_size=2, stride=2),

                            th.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.Conv2d(256, 256, kernel_size=3, padding=1),
                            th.nn.ReLU(inplace=True),
                            th.nn.MaxPool2d(kernel_size=2, stride=2)
                        )

        # state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
        # self.vgg_encoder.load_state_dict(state_dict, strict = False)

        if freeze_encoder:
            for param in self.vgg_encoder.parameters():
                param.requires_grad = False

        self.classifier = th.nn.Sequential(
                            # th.nn.Dropout(0.5),  # classifier:add(nn.Dropout(0.5))
                            th.nn.Linear(4096, 256),
                            # th.nn.BatchNorm1d(512),  # classifier:add(nn.BatchNormalization(512))
                            th.nn.ReLU(inplace=True),
                            th.nn.Linear(256, num_classes)
                        )

        self.loss = th.nn.CrossEntropyLoss()

        # self.decoder = th.nn.Sequential(
        #                 th.nn.Conv2d(256, 256, 3, padding = 1),
        #                 th.nn.ReLU(inplace=True),
        #                 th.nn.ConvTranspose2d(256, 256, 4, stride = 2, padding = 1),
        #
        #                 th.nn.Conv2d(256, 128, 3, padding = 1),
        #                 th.nn.ReLU(inplace=True),
        #                 th.nn.ConvTranspose2d(128, 128, 4, stride = 2, padding = 1),
        #
        #                 th.nn.Conv2d(128, 64, 3, padding = 1),
        #                 th.nn.ReLU(inplace=True),
        #                 th.nn.ConvTranspose2d(64, 64, 4, stride = 2, padding = 1),
        #
        #                 th.nn.Conv2d(64, 1, 3, padding = 1),
        #                 th.nn.ReLU(inplace=True)
        #             )

    def forward(self, img):
        features = self.vgg_encoder(img)
        features = features.view(features.size(0), -1)
        y = self.classifier(features)
        return y

    def get_name(self):
        return 'PCamNet: a VGG like model'

    def compute_loss(self, y, y_):
        celoss = self.loss(y, y_)
        return celoss


class PCamNet(object):

    def __init__(self, input_dims, output_dims, learning_rate = 1e-4):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate

    def build_model(self):
        self.model = PCamNetVGGModel()
        print (self.model.get_name(), self.model)

    def add_optimizer(self):
        self.optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr = self.learning_rate)

    def train_batch(self, data, labels):
        y = self.model.forward( data )
        loss = self.model.compute_loss( y, labels )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def compute_loss(self, data, labels):
        y = self.model.forward( data )
        loss = self.model.compute_loss( y, labels )

        return loss.data

    def forward_pass(self, data):
        y = self.model.forward( data )
        labels = y.data

        return labels

    def save_model(self, ckpt_file):
        th.save(self.model.state_dict, ckpt_file)

    def load_model(self, ckpt_file = None):
        if ckpt_file != None:
            self.model.load_state_dict(th.load(ckpt_file))

    def get_name(self):
        return self.model.get_name()
