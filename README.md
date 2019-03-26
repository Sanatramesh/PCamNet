# PCamNet
Experiments on transfer learning, and triplet loss on PatchCamelyon(PCam) dataset

## Docker Setup

1. Build the docker image using the below command from the directory containing the `Docker` file:

```docker build -t <image name> .```

  This should setup the docker image with pytorch(nvidia gpu support, cuda9.0-cudnn7). The working directory of the docker image is 'workspace'.

2.Run the script inside a container by using the command:

```docker run -it --runtime=nvidia --ipc=host --user="$(id -u):$(id -g)" --volume=$PWD:/workspace <image name> python3 Main.py```

## Running the Code

All the experiments can be run buy passing different command line arguments to the `Main.py` script. 
```python3 Main.py <options>
```

The different options available are:

```optional arguments:                                                   
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset to use for training/testing. Options: cifar10
                        (default), pcam)
  -w MODEL_WEIGHTS, --model_weights MODEL_WEIGHTS
                        Specify complete path to the model weights file to
                        load. (default: None)
  -m MODE, --mode MODE  Specify whether to train, valid, or test. (default:
                        train)

   -e EPOCHS, --epochs EPOCHS                                           
                        Specify the number of epochs to train the model.
                        (default: 100)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Specify the training batch size. (default: 32)
  -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                        Specify the file name of the weights file. (default:
                        None)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Specify the learning rate. (default: 1e-4)
  -op OPTIMIZER, --optimizer OPTIMIZER
                        Specify the optimizer to use for training. Options:
                        adam (default), sgd
  -nn NEURAL_NETWORK, --neural_network NEURAL_NETWORK
                        Neural network architecture to use for
                        training/testing. Options: pcam (default),
                        siamese_pcam
  -c CLASSIFICATION, --classification CLASSIFICATION
                        Classification method to apply while testing. Options:
                        (default), knn
  -ng NUM_NEIGHBORS, --num_neighbors NUM_NEIGHBORS
                        Number of neighbors to use for KNN classifier.

```

## Reference

[1] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer. JAMA: The Journal of the American Medical Association, 318(22), 2199–2210. doi:jama.2017.14585

[2] Schroff, F., Kalenichenko, D., Philbin, J.: Facenet: A unified embedding for face recognition and clustering. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 815–823 (2015)
