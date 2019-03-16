
import numpy
import argparse

from torch.utils import data


from model import *
from model_train import *
# from model_test import *
from dataloader import *

def main(args):
    print('Config arguments')
    print('dataset        :', args.dataset)
    print('mode           :', args.mode)
    print('model wights   :', args.model_weights)
    print('no epochs      :', args.epochs)
    print('batch size     :', args.batch_size)
    print('save weights   :', args.output_model)

    # init dataloader
    data_loader = None

    if args.dataset == 'cifar10':
        data_set = CIFAR10Loader()
    else:
        data_set = PCamLoader()

    data_loader = data.DataLoader(data_set, batch_size=args.batch_size,
                                    shuffle=True, num_workers=4)

    # Create model and load weights
    net = PCamNet([32, 32, 3], [1], learning_rate = 1e-4)
    net.build_model()
    net.load_model(args.model_weights)

    # Setup model training/testing
    train = None
    test  = None

    if args.mode == 'train':
        net.add_optimizer()
        train = ModelTraining(net, data_loader,
                                batch_size = args.batch_size,
                                epochs = args.epochs)
        train.train_model()
        train.save_model(args.output_model)
    else:
        test = ModelTesting(net)
        test.test_model()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse model training options')
    parser.add_argument('-d', '--dataset', default='cifar10',
                    help='Dataset to use for training/testing.\n      Options: cifar10 (default), PCam)')

    parser.add_argument('-w', '--model_weights', default=None,
                    help='Specify complete path to the model weights file to load. (default: None)')

    parser.add_argument('-m', '--mode', default='train',
                    help='Specify whether to train or test. (default: train)')

    parser.add_argument('-e', '--epochs', default=100,
                    help='Specify the number of epochs to train the model. (default: 100)')

    parser.add_argument('-bs', '--batch_size', default=32,
                    help='Specify the training batch size. (default: 32)')

    parser.add_argument('-o', '--output_model', default=32,
                    help='Specify the file name of the weights file. (default: 32)')

    parser.add_argument('-lr', '--learning_rate', default=1e-4,
                    help='Specify the learning rate. (default: 1e-4)')

    # parser.add_argument('-op', '--optimizer', default="adam",
    #                 help='Specify the learning rate. (Options: adam(default), sgd)')

    args = parser.parse_args()
    print(args)

    main(args)
