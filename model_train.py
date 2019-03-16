
import numpy as np


class ModelTraining:

    def __init__(self, model, data_loader, batch_size = 10, epochs = 20, model_ckpt_file = 'model/PCamNet.pt'):
        self.model = model
        self.data_loader = data_loader # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.num_data = len( self.data_loader )
        self.model_ckpt_file = model_ckpt_file

    def train_model(self):
        data_points = np.zeros( self.num_data )
        print ( 'Training Model: %s ... ' % self.model.get_name() )

        # train model for one epoch - call fn model.train_batch(data) for each batch
        for epoch in range( self.num_epochs ):
            training_loss = 0.0

            for batch_data, batch_labels in self.data_loader:
                print ( batch_data.shape, batch_labels.shape )
                training_loss += self.model.train_batch( batch_data, batch_labels )

            training_loss /= ( self.no_data // self.batch_size )
            print ()

            # validate the model and print test, validation accuracy
            validation_loss = self.model.compute_loss( validation_data )
            valid_output = self.model.forward_pass( validation_data )

            print ( 'epoch: %4d    train loss: %20.4f     val loss: %20.4f' %
                                    ( i, training_loss, validation_loss ) )

            print ('Mean:', np.mean(valid_output))
            print ('Max:', np.max(valid_output))
            print ('Min:', np.min(valid_output))
            print ('Unique:', np.unique(valid_output))
            print ('')
            self.model.save_model( self.model_checkpoint_dir )

        print ( 'Training Model: %s ... Complete' % self.model.get_name() )

    def get_model(self):
        return self.model
