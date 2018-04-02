# Domain Background
Image recognition or image classification is a problem in vision science as well as in computer science. The task is to assign a image to one or more classes. This may be done by manually or algorithmically and the images to be classfied may have different ojbect in it, different size and different color. Images may classified according to edges,corners, blobs and ridge(1).

This project will focus on algorithmically method, excatly machine learning algorithms which are widely used in image recognition and image classification. There are many classification algortihms such as Decision Tree, KNN, Naives Bayes and Neural Network, each of these models have their own advantages and disadvantages. The algortihm i will use is convolutional neural network(CNN or ConvNet). It is a deep, feed-forward artifical neural network that has successfully been applied to analyzing visual imagery(2).

There are many public images online for classification, here, i will apply classification algorithms on a Kaggle project dogs vs cats(3), which contains 25000 images in train set and 12500 images in test set. The image will be classified according to their contents.


# Problem Statement
The classification of dogs vs cats is a supervised classification problem, there are 2 categories, each of the image is only belong to one category, the goal is to use transfer learning from highly matured CNNs to build our model to assign each image to the correct category.

I will download highly matured CNN models without top layer and use these downloaded models to extract bottleneck features based on train set and test set. And save these bottleneck features as numpy array on local disk for saving time. Based on these features i will build my own CNN models and train my models. Finally i will validate the performance on my own models.

In this project i will use TensorFlow(4) as backend and Keras(5) as high level API to build CNN. To save time i will choose VGG16, VGG9, ResNet, InceptionV3, Xception as base CNN models because these models are all embed in Keras Application API(6).


# Datasets and Inputs
The datasets can be downloaded from Kaggle directly(7). The train set contains 2 categories images, 12500 images for dog and 12500 images for cat. But test set don't provide any category, i only can use it when i want to generate a submission file based on our prediction via test set. Which means i only can know test result after submission. Hence i use another datasets, Oxford Pet datasets(8) as testing datasets which can run test locally. The Oxford Pet datasets contains 7390 images in total, 2400 images for cats, 4990 images for dogs.

As i will use Keras as high level API it is a good practice to use Keras image preprocceing APIs like ImageDataGenerator(9) to generate batches of tensor images. First, i need to split train data into 2 sub-folders, test data into 1 sub-folder and Oxford Pet datasets also into 2 sub-folders. Then feed these images to ImageDataGenerator by folder path to get tensor images as our input tensor to CNNs.


# Solution Statement




# Benchmark Model
The benchmark model 


# Evaluation Metrics
As it is a binary classification problem with thoudsands of examples, i will use two metrics here: accuracy and running time.
- accuracy: the proportion of correct label we made on our traning dataset. Ideally it should be 100%.
$$\textrm{accuracy} = \frac{1}{n} \sum_{i=1}^n \left[ y_i==\hat{y}_i]$$
- time:


# Project Design





[(1)](https://en.wikipedia.org/wiki/Feature_detection_(computer_vision))
[(2)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
[(3)](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
[(4)](https://tensorflow.google.cn/)
[(5)](https://keras.io/)
[(6)](https://keras.io/applications/)
[(7)](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
[(8)](http://www.robots.ox.ac.uk/%7Evgg/data/pets/)
[(9)](https://keras.io/preprocessing/image/)
