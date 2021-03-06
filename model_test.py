import os
import pickle
import torch as th
from torch.utils import data
import time
from copy import deepcopy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataloader import *


class ModelTesting:

    def __init__(self, model, data_loader, test_file = 'model/PCamNet'):
        self.model = model
        self.data_loader = data_loader # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.test_file = test_file

        self.model.model.eval()

    def test_model(self, args):

        if args.classification == 'knn':
            self._knn_classifier(args)
        else:
            self._probability_classifier()

        return 0

    def _probability_classifier(self):
        print( 'Testing model:', self.model.get_name() )
        test_loss  = 0.0
        test_count = 0
        true_labels = []
        pred_labels = []

        softmax = th.nn.Softmax( dim=1 )
        for batch_data, batch_labels in self.data_loader:
            labels = self.model.forward_pass( batch_data )
            test_loss += self.model.compute_loss( batch_data, batch_labels )

            true_labels.append(batch_labels.numpy())
            pred_labels.append(th.argmax(softmax(labels), dim=1).numpy())
            test_count += 1

        pickle.dump({'test_loss': test_loss.numpy(),
                     'test_count': test_count,
                     'true_labels': true_labels,
                     'pred_labels': pred_labels
                     }, open(self.test_file + '_test_stats.pkl','wb'))

        print('Test loss:', test_loss.numpy()/test_count)
        return 0


    def _compute_embeddings(self, args):

        # Check if we have already computed the embeddings
        if os.path.isfile(self.test_file + '_train_feats.npy') and \
            os.path.isfile(self.test_file + '_test_feats.npy'):
            print('Embeddings already computed at model/*.npy')
            return 0


        if args.dataset == 'cifar10':
            data_set = CIFAR10Loader(mode='train')
        else:
            data_set = PCamLoader(mode='train')



        test_X = []
        test_y = []

        t3 = time.time()
        for batch_data, batch_labels in self.data_loader:
            test_X.append( deepcopy(self.model.compute_features( batch_data ).numpy()) )
            test_y.append( deepcopy(batch_labels).numpy() )

        t4 = time.time()
        test_X = np.concatenate(test_X)
        test_y = np.concatenate(test_y)

        np.save(self.test_file + '_test_feats.npy', test_X)
        np.save(self.test_file + '_test_labels.npy', test_y)

        print('Computed test features for knn classifier:', test_X.shape, test_y.shape)

        train_dataloader = data.DataLoader(data_set, batch_size=args.batch_size,
                                        shuffle=False)

        train_X = []
        train_y = []

        for batch_data, batch_labels in train_dataloader:
            train_X.append( self.model.compute_features( batch_data ).numpy() )
            train_y.append( batch_labels.numpy() )

        train_X = np.concatenate(train_X)
        train_y = np.concatenate(train_y)

        np.save(self.test_file +  '_train_feats.npy', train_X)
        np.save(self.test_file + '_train_labels.npy', train_y)

        print('Computed train features:', train_X.shape, train_y.shape, )
        train_dataloader = None

        return 0


    def _knn_classifier(self, args):

        self._compute_embeddings(args)

        train_X = np.load(self.test_file + '_train_feats.npy')
        train_y = np.load(self.test_file + '_train_labels.npy')

        t1 = time.time()
        knn = KNeighborsClassifier(n_neighbors=args.num_neighbors)
        knn.fit(train_X, train_y)
        t2 = time.time()

        print('Fitted KNN model', 'Fit model time:', (t2-t1))

        test_X = np.load(self.test_file + '_test_feats.npy')
        test_y = np.load(self.test_file + '_test_labels.npy')

        knn_predict = []
        knn_score = 0.0
        knn_count = 0
        bs = args.batch_size
        print('Dims:', test_X.shape, test_y.shape)
        t3 = time.time()
        for idx in range(0, test_y.shape[0], bs):
            print('predicting:', knn_count,idx,bs)
            knn_predict.append([0.0, knn.predict(test_X[idx:idx+bs])])
            acc = accuracy_score(knn_predict[-1][1], test_y[idx:idx+bs])
            knn_predict[-1][0] = acc
            # knn_predict.append( knn.predict( test_X[-1] ) )
            # knn_score += knn.score( test_X[-1], test_y[-1] )
            knn_count += 1

        t4 = time.time()

        print('Score:', acc/knn_count, 'Pred time:', (t4-t3))
        pickle.dump(knn_predict, open(self.test_file +'_knn_'+ str(args.num_neighbors) + '_test_stats.pkl','wb'))
        return 0

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
