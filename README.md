# PCamNet
Experiments on transfer learning, and triplet loss on PatchCamelyon(PCam) dataset

## Docker Setup

1. Build the docker image using the below command from the directory containing the `Docker` file:

```docker build -t <image name> .```

  This should setup the docker image with pytorch(nvidia gpu support, cuda9.0-cudnn7). The working directory of the docker image is 'workspace'.

2.Run the script inside a container by using the command:

```docker run -it --runtime=nvidia --ipc=host --user="$(id -u):$(id -g)" --volume=$PWD:/workspace <image name> python3 Main.py
```

## Running the Code

All the experiments can be run buy passing different command line arguments to the `Main.py` script. 
