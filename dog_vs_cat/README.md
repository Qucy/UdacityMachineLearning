### Capstone project DogVsCat

为什么选择了这个项目？刚从P5转战过来，这个项目与P5（狗狗分类）非常类似，但是比P5要求又高出很多作为练手的项目还是不错的。当然其他的Capstone项目我打算在毕业之后继续选做2个，一个是自然语言处理的文档归类项目以及Rossmann销售预测项目。项目本身取自Kaggle是一个已经close的项目，项目的需求也比较明确和简单，输入“彩色图片”，输出“是猫还是狗”，一个监督学习的二分类的问题。


### 环境的搭建

项目训练集一共有25000张图片解压后594m，测试集一共有12500张图片解压后有296m。考虑到本地机器性能还算不错，选择在本地搭建环境而非AWS。机器配置大概如下
```
CPU : Inter i7-6700K
Memory : 16G
GPU : Nvidia GeForce GTX 1070 8G
```
根据[Tensoflow官方](https://tensorflow.google.cn/)给出的文档在本地搭建GPU的运行环境。首先下载并安装CUDA TookKit 9.0，然后下载对应版本的cuDNN 7.0，解压后根据官方提示将解压后的cuDNN的文件拷贝到指定目录中；[NVIDIA环境的搭建的具体步骤](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/ "Markdown")。
```
安装完成后可以通过命令：nvcc -V 来检查下是否安装成功，安装成功的话会输出以下命令
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:32_Central_Daylight_Time_2017
Cuda compilation tools, release 9.0, V9.0.176
```
考虑到会使用jupyter notebook，所以选择用Anaconda，再Nivida的环境安装完成后，继续安装Anaconda的环境，脚本在Tensorflow官方文档都已经详细给出
```
1. 创建python的环境 
   1.1) conda create -n tensorflow pip python=3.5
2. 其次安装 Tensorflow的GPU版本
   2.1）activate tensorflow
   2.2）pip install --ignore-installed --upgrade tensorflow-gpu
3. 最后按照官方提示跑一个hello world的应用来校验安装是否成功
   >>> import tensorflow as tf
   >>> hello = tf.constant('Hello, TensorFlow!')
   >>> sess = tf.Session()
   >>> print(sess.run(hello))
```
![TF Hello World](\images\tf_hello_world.PNG)
