



class ModelTesting:

    def __init__(self, model, data_files, save_dir = 'test_out'):
        self.model = model
        self.test_data_files = data_files # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.dir_out = save_dir

    def test_model(self):
        directory = self.dir_out + '/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')
        os.makedirs(directory)

        print ('Testing model:', self.model.get_name() )

        for i, data_file in enumerate(self.test_data_files):
            data = read_data( [ data_file  ] )

            disparity_map = self.model.forward_pass( data )
            test_loss = self.model.compute_loss( data )

            # Save Disparity map
            print ('data',data[2][0])
            print ('disp', disparity_map[0])
            disparity_map = disparity_map[0]
            h, w = disparity_map.shape[0], disparity_map.shape[1]
            disparity_map = disparity_map.reshape((h, w))
            plt.imshow(data[2][0].reshape((h, w)))
            plt.show()
            plt.imshow(disparity_map)
            plt.show()
            fname = self.test_data_files[i][0].split('.')[0] + '_disp.png'
            fname = fname.replace('/', '-')
            fname = directory + '/' + fname
            imsave(fname, disparity_map)

            print ( 'Test: %4d image: %s loss: %4.4f' %( i, self.test_data_files[i][0] , test_loss ) )


    def set_test_data_files(self, files):
        self.test_data_files = files
