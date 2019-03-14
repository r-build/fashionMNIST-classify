# fashionMNIST image classification
Classify images from the fashion-MNIST dataset using:
 - densley-connected neural network model
 - convolutional neural network model

## data
The data is a subset (20,000 images) from the complete [fashion-MNIST dataset here](https://github.com/zalandoresearch/fashion-mnist)

## Conda env
conda create -n fmnist-nn python=3 jupyter keras tensorflow-mkl  matplotlib scikit-learn numpy imageio pandas seaborn

N.B. use tensorflow-mkl in the abscence of a GPU and presence of a suitable Intel CPU
