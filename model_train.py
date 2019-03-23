import time
import pickle
import numpy as np
from copy import deepcopy

class ModelTraining:

    def __init__(self, model, data_loader, batch_size = 10, epochs = 20, model_ckpt_file = 'model/PCamNet'):
        self.model = model
        self.data_loader = data_loader # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_batch = len( self.data_loader )
        self.model_ckpt_file = model_ckpt_file
        self.train_stats = []

    def train_model(self):
        print ( 'Training Model: %s ... ' % self.model.get_name() )

        # train model for one epoch - call fn model.train_batch(data, label) for each batch
        for epoch in range( self.num_epochs ):
            training_loss  = 0.0
            training_count = 0

            validation_loss    = 0.0
            validation_count   = 0
            validation_predict = []
            true_labels = []

            true_labels = []
            t1 = time.time()

            for batch_data, batch_labels in self.data_loader:

                if training_count <= 0.8 * self.num_batch:
                    training_loss  += self.model.train_batch( batch_data, batch_labels )
                    training_count += 1
                else:
                    validation_loss   += self.model.compute_loss( batch_data, batch_labels )
                    # validation_predict.append(self.model.forward_pass( batch_data ).numpy())
                    validation_count  += 1
                    # true_labels.append(deepcopy(batch_labels).numpy())

            t2 = time.time()
            self.train_stats.append([epoch, training_loss.numpy(), training_count,
                                    validation_loss.numpy(), validation_count,
                                    validation_predict, true_labels, t2 - t1])

            print ()

            print ( 'epoch: %4d    train loss: %20.6f     val loss: %20.6f' %
                                    ( epoch, training_loss / training_count,
                                             validation_loss / validation_count ) )

            print('epoch time:', np.round(t2 - t1, 2), 's')
            print('time for completion:', np.round((t2 - t1) * (self.num_epochs - epoch - 1) / 60, 2), 'm')
            print ('')

            self.model.save_model( self.model_ckpt_file + '.pth')

            pickle.dump(self.train_stats, open(self.model_ckpt_file+'_stats.pkl','wb'))

        print ( 'Training Model: %s ... Complete' % self.model.get_name() )
        print ( 'Saving stats into model/stats.pkl')

        return 0

    def get_model(self):
        return self.model

    def set_model_save(self, filename):
        self.model_ckpt_file = filename
