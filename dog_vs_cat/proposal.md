# Domain Background
Image recognition or image classification is a problem in vision science as well as in computer science. The task is to assign a image to one or more classes. This may be done by manually or algorithmically and the images to be classfied may have different ojbect in it, different size and different color. Images may classified according to edges,corners, blobs and ridge(1).

This project will focus on algorithmically method, excatly machine learning algorithms which are widely used in image recognition and image classification. There are many classification algortihms such as Decision Tree, KNN, Naives Bayes and Neural Network, each of these models have their own advantages and disadvantages. The algortihm i will use is convolutional neural network(CNN or ConvNet). It is a deep, feed-forward artifical neural network that has successfully been applied to analyzing visual imagery(2).

There are many public images online for classification, here, i will apply classification algorithms on a Kaggle project dogs vs cats(3), which contains 25000 images in train set and 12500 images in test set. The image will be classified according to their contents.


# Problem Statement
The classification of dogs vs cats is a supervised classification problem, there are 2 categories, each of the image is only belong to one category, the goal is to use transfer learning from highly matured CNNs to build our model to assign each image to the correct category.

Becasue training CNN from scratch is time consuming. I will download highly matured CNN models without top layer and use these downloaded models to extract bottleneck features based on train set and test set. Based on these features i will build my own CNN models and train my models. Finally i will validate the performance on my own models.

In this project i will use TensorFlow(4) as backend and Keras(5) as high level API to build CNN. To save time i will choose VGG16, VGG9, ResNet, InceptionV3, Xception as base CNN models because these models are all embedded in Keras Application API(6).


# Datasets and Inputs
The datasets can be downloaded from Kaggle directly(7). The train set contains 2 categories images, 12500 images for dog and 12500 images for cat. But test set don't provide any category, i only can use it when i want to generate a submission file based on our prediction via test set. Which means i only can know test result after submission. Hence i use another datasets, Oxford Pet datasets(8) as testing datasets which can run test locally. The Oxford Pet datasets contains 7390 images in total, 2400 images for cats, 4990 images for dogs.

As i will use Keras as high level API it is a good practice to use Keras image preprocceing APIs like ImageDataGenerator(9) to generate batches of tensor images. First, i need to split train data into 2 sub-folders, test data into 1 sub-folder and Oxford Pet datasets also into 2 sub-folders. Then feed these images to ImageDataGenerator by folder path and image size to get tensor images as our input tensor to CNNs.


# Solution Statement
As we build our model based on bottleneck features extrac from VGG, ResNet, Xception and Inception. We can simply build our own model just contain 2-3 layers to do classification. Then we use input tensor generate by Keras to train our model. There are 2 solutions, first is only use one model's bottleneck features to train our model. Second is combine all the models' bottleneck features to train our model and even we integrate with other learning algorithm. According to Kaggle winner interview(10) the second solution will be better than the first one. I will try both solutions in this project.


# Benchmark Model
The benchmark model i will use is VGG16. I will compare our models to VGG16.


# Evaluation Metrics
As it is a binary classification problem with thoudsands of examples, i will use two metrics here: accuracy and running time.
- accuracy: the proportion of correct label we made on our traning dataset. Ideally it should be 100%.
![Accuracy](images\accuracy.PNG)
- time: the time that the algorithm takes to do classification, a good algorithm should predict fast as end user can't wait for a long time
Above all a well performance model should have high accuracy and a reasonable running time.


# Project Design





(1) Computer vision: https://en.wikipedia.org/wiki/Feature_detection_(computer_vision)

(2) Convolutinal neural network: https://en.wikipedia.org/wiki/Convolutional_neural_network

(3) Kaggle Dogs vs Cats: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

(4) TensorFlow: https://tensorflow.google.cn/

(5) Keras: https://keras.io/

(6) Keras Applications: https://keras.io/applications/

(7) Dogs vs Cats Datasets: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

(8) Oxford Pet Datasets: http://www.robots.ox.ac.uk/%7Evgg/data/pets/

(9) Keras ImageDataGenerator: https://keras.io/preprocessing/image/

(10) Kaggle Winner Interview: http://blog.kaggle.com/2017/04/03/dogs-vs-cats-redux-playground-competition-winners-interview-bojan-tunguz/
