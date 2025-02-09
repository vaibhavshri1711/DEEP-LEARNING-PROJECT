# TASK-2: DEEP-LEARNING-PROJECT

COMPANY: CODTECH IT SOLUTIONS

NAME: GHODKE VAIBHAVSHRI VISHWANATH

INTERN ID: CT12IBE

DOMAIN: DATA SCIENCE

DURATION: 8 WEEEKS

MENTOR: NEELA SANTOSH

This code is a complete implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. Below is a step-by-step description of the code:

Step 1: Import Necessary Libraries
1.TensorFlow: A popular deep learning library used for building and training neural networks.

2.Keras: A high-level API for TensorFlow that simplifies the process of building neural networks.

3.Matplotlib: A plotting library used for visualizing data.
NumPy: A library for numerical operations in Python.

Step 2: Load and Preprocess the Dataset
The MNIST dataset is loaded, which consists of 70,000 images of handwritten digits (0-9).
The images are normalized to the range [0, 1] by dividing by 255.0, which helps in faster convergence during training.
A channel dimension is added to the images to make them compatible with convolutional layers. The shape of the images changes from (28, 28) to (28, 28, 1).
The labels are converted to one-hot encoding, transforming the integer labels into binary class matrices.

Step 3: Display a Few Images from the Dataset
A grid of 12 images from the training set is displayed using Matplotlib. Each image is shown with its corresponding label.

Step 4: Build the Deep Learning Model
A Sequential model is defined, which consists of:
Convolutional Layers: Two convolutional layers with ReLU activation functions to extract features from the images.

1.MaxPooling Layers: Two max pooling layers to down-sample the feature maps, reducing their dimensionality.

2.Flatten Layer: Flattens the 2D feature maps into a 1D vector.

3.Dense Layers: A fully connected layer with 128 neurons and ReLU activation, followed by an output layer with 10 neurons (one for each digit) and softmax activation to produce probabilities for each class.
The model is compiled with the Adam optimizer, categorical cross-entropy loss function, and accuracy as a metric.

Step 5: Train the Model
The model is trained on the training dataset for 10 epochs with a batch size of 32. Validation data is provided to monitor the model's performance on unseen data during training.
After training, the model is evaluated on the test dataset, and the test accuracy is printed.

Step 6: Plot Training and Validation Accuracy and Loss
Two plots are generated:
The first plot shows the training and validation accuracy over the epochs, allowing visualization of how well the model is learning.
The second plot shows the training and validation loss over the epochs, which helps in understanding if the model is overfitting or underfitting.

Step 7: Predict on Test Set
The model makes predictions on the test dataset.
A grid of 12 images from the test set is displayed, showing both the true labels and the predicted labels for each image.

OUTPUT:
![Image](https://github.com/user-attachments/assets/9df45f3d-27f4-4b50-a645-664c456c2b25)
