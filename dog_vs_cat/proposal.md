# Domain Background
  The capstone project i choose is a computer vision project dogs vs cats start at Kaggle in 2013. By the end 1314 teams joined this project and first place's score is 0.03302. For this project we need to reach top 10% at leaderborad which means need to score less than 0.06127.
  From 2012 convoluational neural network become the state-of-the-art in computer vision. CNN like VGG, ResNet and Inception are pretty good at classfiy and recognize the images, some are even better than human. Take ResNet for example the best ResNet's error on ImageNet datasets is 3.57% while human's error is 5.1%. So the CNN can be a very good solution to this project. The reason why i choose this project is simply just this is the first project and i plan to finish every capstone project one by one.

[Kaggle project dog_vs_cat](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
[ResNet](https://arxiv.org/abs/1512.03385)


# Problem Statement
This project need to classfiy the image and output whether this is a dog or a cat based on input image. Hence this is a binary classfication problem in supervised learning and also belong to computer vision domain. It can be sloved by using transfer learning from highly mature CNN models like ResNet, Xception, Inception base on our own datasets.

# Datasets and Inputs
The datasets can be download from Kaggle's project dogs vs cats directly. There are 3 files need to be downloaded, train.zip contains 25000 images as our trainning data, test.zip contains 12500 images as our testing data, sample_submission.csv is a sample teach us how to submit our test result, we can generate our test reslut based on this file.

[dog_vs_cat data](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
[Oxford pet data](http://www.robots.ox.ac.uk/%7Evgg/data/pets/)


# 解决办法
本项目的解决方案是基于成熟的CNN模型，ResNet，Xception，Inception等进行迁移学习。同时利用data argumentation，Fine-tuning方法来优化模型，提高模型的准确率。

Benchmark Model
A benchmark model is provided that relates to the domain, problem statement, and intended solution. Ideally, the student's benchmark model provides context for existing methods or known information in the domain and problem given, which can then be objectively compared to the student's solution. The benchmark model is clearly defined and measurable.
# 基准模型
基准模型我将选用VGG19作为基准模型来和我的模型做比较。


Evaluation Metrics
Student proposes at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model presented. The evaluation metric(s) proposed are appropriate given the context of the data, the problem statement, and the intended solution.

Project Design
Student summarizes a theoretical workflow for approaching a solution given the problem. Discussion is made as to what strategies may be employed, what analysis of the data might be required, or which algorithms will be considered. The workflow and discussion provided align with the qualities of the project. Small visualizations, pseudocode, or diagrams are encouraged but not required.

Presentation
Proposal follows a well-organized structure and would be readily understood by its intended audience. Each section is written in a clear, concise and specific manner. Few grammatical and spelling mistakes are present. All resources used and referenced are properly cited.
