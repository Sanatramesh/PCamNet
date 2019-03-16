


class ModelTraining:

    def __init__(self, model, data_files, batch_size = 10, epochs = 20, model_checkpoint_dir = 'model/TD_net.pt'):
        self.model = model
        self.train_data_files = data_files # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.batch_size = batch_size
        self.no_epochs = epochs
        self.no_data = len( self.train_data_files )
        self.model_checkpoint_dir = model_checkpoint_dir

    def train_model(self):
        data_points = np.zeros(len(self.train_data_files))
        print ( 'Training Model: %s ... ' % self.model.get_name() )
        # shuffle and read a batch from the train dataset
        # train model for one epoch - call fn model.train_batch(data) for each batch
        for i in range( self.no_epochs ):
            data_shuffled = self.train_data_files[:]
            shuffle(data_shuffled)
            training_loss = 0.0

            for batch in range( 0, self.no_data, self.batch_size ):
                print (batch,)
                train_data = read_data( data_shuffled[ batch:batch + self.batch_size ] )
                training_loss += self.model.train_batch( train_data )

            training_loss /= ( self.no_data // self.batch_size )
            print ()

            # validate the model and print test, validation accuracy
            batch_idx = next_batch( self.no_data, self.batch_size )
            validation_data = read_data( [ self.train_data_files[ idx ] for idx in batch_idx ] )
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
