# Font_Recognition

## Introduction to Project

In this project, you'll train a convolutional neural network to classify and recognize different categories of fonts. We'll be using the dataset of 100 categories of fonts to train our model.
The project is broken down into multiple steps:
- Generating the dataset of fonts from package
- Loading and preprocessing the image dataset
- Visualization of samples from the dataset
- Train the Convolutional Neural Network on your dataset
- Use the trained model to predict new fonts

The whole project is implemented in tensorflow.

### Dataset Description

Adobe VFR dataset was the first large-scale, fine-grained benchmark of font text images, for the task of font recognition and retrieval. But its size was very huge and 
![](https://raw.githubusercontent.com/rois-codh/kmnist/master/images/kmnist_examples.png)

## Files Description

- **main_file.ipynb** It contains the full code and is used to build the model using the jupyter notebook. It can be used independently to see how the model works.
- **kmnist_classmap.csv** It is used in ipynb file to map from class IDs to unicode characters for Kuzushiji-MNIST.
- **kmnist-train-images.npz** It contains the training dataset of 60,000 images (28x28 grayscale) provided in a NumPy format.
- **kmnist-test-images.npz** It contains the test dataset of 10,000 images (28x28 grayscale) provided in a NumPy format.
- **kmnist-train-labels.npz** It contains the training labels of 60,000 images provided in a NumPy format for training dataset.
- **kmnist-test-labels.npz** It contains the test labels of 10,000 images provided in a NumPy format for test dataset.

**NOTE:** kmnist-[train/test]-[images/labels].npz: These files contain the Kuzushiji-MNIST as compressed numpy arrays, and can be read with: arr = np.load(filename)['arr_0']. We recommend using these files to load the dataset.

## Installation
The Code is written in Jupyter Notebook.

Additional Packages that are required are: Numpy, Pandas, MatplotLib, Pytorch, and PIL. You can donwload them using pip

`pip install numpy pandas matplotlib pil`

In order to intall Pytorch head over to the [Pytorch](https://pytorch.org/get-started/locally/) website and follow the instructions given.

## GPU/CPU

As this project uses deep CNNs, for training of network you need to use a GPU. However after training you can always use normal CPU for the prediction phase.

## License
[MIT License](https://github.com/imshubhamkapoor/Kuzushiji_MNIST_Japanese_Character_Classification/blob/master/LICENSE)

## Author
Shubham Kapoor
