import pickle
import torch as th


class ModelTesting:

    def __init__(self, model, data_loader, test_file = 'model/PCamNet'):
        self.model = model
        self.data_loader = data_loader # List of tuple: (left_cam, right_cam, disp_map) filenames
        self.test_file = test_file

    def test_model(self):
        print( 'Testing model:', self.model.get_name() )
        test_loss  = 0.0
        test_count = 0
        true_labels = []
        pred_labels = []

        softmax = th.nn.Softmax( dim=1 )
        for batch_data, batch_labels in self.data_loader:
            labels = self.model.forward_pass( batch_data )
            test_loss += self.model.compute_loss( batch_data, batch_labels )

            true_labels.append(batch_labels)
            pred_labels.append(th.argmax(softmax(labels), dim=1).numpy())
            test_count += 1

        pickle.dump({'test_loss': test_loss.numpy(),
                     'test_count': test_count,
                     'true_labels': true_labels,
                     'pred_labels': pred_labels
                     }, open(self.test_file + '_test_stats.pkl','wb'))

        return 0

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
