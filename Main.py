
import numpy
import argparse

from torch.utils import data


from model import *
from model_train import *
# from model_test import *
from dataloader import *


def main(args):
    print(colored('Config arguments', 'green'))
    print('dataset         :', args.dataset)
    print('mode            :', args.mode)
    print('model weights   :', args.model_weights)
    print('no epochs       :', args.epochs)
    print('batch size      :', args.batch_size)
    print('optimizer       :', args.optimizer)
    print('learning rate   :', args.learning_rate)
    print('neural net arch :', args.neural_network)
    print('save weights    :', args.output_model)

    # init dataloader
    data_loader = None

    if args.dataset == 'cifar10':
        data_set = CIFAR10Loader(mode=args.mode)
        net = PCamNet([32, 32, 3], 10, learning_rate = 1e-4)
    else:
        data_set = PCamLoader(mode=args.mode)
        net = PCamNet([32, 32, 3], 2, learning_rate = 1e-4)

    data_loader = data.DataLoader(data_set, batch_size=args.batch_size,
                                    shuffle=True, num_workers=4)

    # Create model and load weights
    net.build_model(type=args.neural_network)
    net.load_model(args.model_weights)

    # Setup model training/testing
    train = None
    test  = None

    if args.mode == 'train':
        net.add_optimizer()
        train = ModelTraining(net, data_loader,
                                batch_size = args.batch_size,
                                epochs = args.epochs)

        if args.output_model != None:
            train.set_model_save(args.output_model)

        train.train_model()

    elif args.mode == 'valid':
        net.add_optimizer()
        train = ModelTraining(net, data_loader,
                                batch_size = args.batch_size,
                                epochs = args.epochs)
        if args.output_model != None:
            train.set_model_save(args.output_model)

        train.train_model()

    elif args.mode == 'test':
        test = ModelTesting(net)
        test.test_model()
    else:
        raise Exception('Unknown: mode type - '+args.mode)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse model training options')
    parser.add_argument('-d', '--dataset', default='cifar10',
                    help='Dataset to use for training/testing.\n      Options: cifar10 (default), pcam)')

    parser.add_argument('-w', '--model_weights', default=None,
                    help='Specify complete path to the model weights file to load. (default: None)')

    parser.add_argument('-m', '--mode', default='train',
                    help='Specify whether to train, valid, or test. (default: train)')

    parser.add_argument('-e', '--epochs', default=100, type=int,
                    help='Specify the number of epochs to train the model. (default: 100)')

    parser.add_argument('-bs', '--batch_size', default=32, type=int,
                    help='Specify the training batch size. (default: 32)')

    parser.add_argument('-o', '--output_model', default=None,
                    help='Specify the file name of the weights file. (default: None)')

    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float,
                    help='Specify the learning rate. (default: 1e-4)')

    parser.add_argument('-op', '--optimizer', default="adam",
                    help='Specify the optimizer to use for training.' + \
                        '\n      Options: adam (default), sgd')

    parser.add_argument('-nn', '--neural_network', default='pcam',
                    help='Neural network architecture to use for training/testing.'+ \
                        '\n      Options: pcam (default), siamese_pcam')

    args = parser.parse_args()
    print(args)

    main(args)
