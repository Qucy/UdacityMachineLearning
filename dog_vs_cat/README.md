# 环境的搭建

项目训练集一共有25000张图片解压后594m，测试集一共有12500张图片解压后有296m。考虑到本地机器性能还算不错，选择在本地搭建环境而非AWS。机器配置大概如下
```
CPU : Inter i7-6700K
Memory : 16G
GPU : Nvidia GeForce GTX 1070 8G
OS : Win 10 64bit
```

根据[Tensorflow官方](https://tensorflow.google.cn/)给出的文档在本地搭建GPU的运行环境。首先下载并安装CUDA TookKit 9.0，然后下载对应版本的cuDNN 7.0。根据[NVIDIA官方说明](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/ "Markdown")将解压后的cuDNN的文件拷贝到指定目录中。注意，这里CUDA和cuDNN的版本一定要和官方要求的一致！！

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

当然考虑到我们需要读取图片，构建CNN，操作矩阵等等。因此还需要安装一些额外的工具包。注意Keras最好使用pip的命令来安装，这里我踩了一个坑。用conda install keras后，自动给我安装了一个cpu版本的tensorflow从而覆盖掉了GPU版本的tensorflow。

```
pip install --ignore-installed --upgrade jupyter notebook
pip install --ignore-installed --upgrade keras
pip install --ignore-installed --upgrade pillow
conda install opencv
conda install pandas
conda install matplotlib
```
至此环境准备完毕


# 模型运行时间
因为是基于深度特征来训练模型，所以模型的训练时间并不长。导出深度特征花的时间是最多，大概如下。当然模型的训练时间，已经运行时间我也有记录，在report和代码中都有记录。

 |模型名称|总耗时（单位：秒）|深度特征文件大小（单位：MB）|
 | ------ | -----:|-----:|
 | VGG16    | 205.36 | 73.3 |
 | VGG19    | 232.02 | 73.3 |
 | ResNet50 | 210.49 | 293 |
 | InceptionV3 | 274.99 | 293|
 | Xception    | 416.50 | 293|


