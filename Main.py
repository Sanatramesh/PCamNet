
import numpy
import argparse


# from model import *
# from model_train import *
# from model_test import *
# from dataloader import *

def main(args):
    # global LEFT_CAM_DIR, RIGHT_CAM_DIR, DISPARITY_DIR
    #
    # net = TDNet([480, 640, 3], [480, 640, 1], learning_rate = 1e-4)
    # net.build_model()
    # net.add_optimizer()
    #
    # # data_files = get_data_files_SINTEL(LEFT_CAM_DIR, RIGHT_CAM_DIR, DISPARITY_DIR)
    # data_files = get_data_files_NTSD(CAM_DIR, DISPARITY_DIR)
    # data_shuffled = data_files[:]
    # shuffle(data_shuffled)
    #
    # train_data_files = data_shuffled[len(data_shuffled)//5:]
    # test_data_files  = data_shuffled[:len(data_shuffled)//5]
    #
    # print ('Test and Train split:', len(train_data_files), len(test_data_files))
    #
    # train = ModelTraining(net, train_data_files, batch_size = 1, epochs = 10)
    # train.train_model()

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
        data_loader = CIFAR10Loader()
    else:
        data_loader = PCamLoader()

    # Create model and load weights
    net = TDNet([480, 640, 3], [480, 640, 1], learning_rate = 1e-4)
    net.build_model()
    net.load_model(args.model_weights)

    # Setup model training/testing
    train = None
    test  = None

    if args.mode == 'train':
        train = ModelTraining(net, train_data_files,
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
                    help='Dataset to use for training/testing.\n      Options: "cifar10"(default), "PCam")')

    parser.add_argument('-w', '--model_weights', default=None,
                    help='Specify complete path to the model weights file to load. (default: "")')

    parser.add_argument('-m', '--mode', default='train',
                    help='Specify whether to train or test. (default: "train")')

    parser.add_argument('-e', '--epochs', default=100,
                    help='Specify the number of epochs to train the model. (default: 100)')

    parser.add_argument('-b', '--batch_size', default=32,
                    help='Specify the training batch size. (default: 32)')

    parser.add_argument('-o', '--output_model', default=32,
                    help='Specify the file name of the weights file. (default: 32)')

    args = parser.parse_args()
    print(args)

    main(args)
