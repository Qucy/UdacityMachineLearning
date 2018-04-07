# 猫狗分类项目报告

## 定义
### 项目概述
图像识别或图像分类是属于视觉科学和电脑科学领域的一个问题。这类问题的目的是需要将图片分类至一个或多个类别当中。这个问题可以通过人工手动或借助算法来完成，这些图片可能包含了不同的物体，图片的大小可能不一致，颜色上也千差万别。图片仍可以根据它们所包含的边缘（edges），角（corners），区域（blobs）和脊（ridge）来做分类[1]。

本项目将会采用广泛使用的机器学习算来进行图片识别以及图片分类。机器学习包含许多的分类算法比如，决策树，KNN，朴素贝叶斯和神经网络等等，每个模型都有它们各自的优缺点。这里我们将采用卷积神经网络（简称CNN或ConvNet）来做图片分类。卷积神经网络是一个兼具深度和高识别率的神经网络，已经成功应用于图像识别领域。

本项目将基于Kaggle的一个竞赛项目Dogs vs Cats[2]来进行图像识别。这个项目包含了25000张的训练集图片以及12500张的测试集图片。这些图片将被划分到猫或狗的一个分类中。

### 问题陈述
猫狗识别的项目属于监督学习中的分类问题，本项目中一共有2个类别，每一张图片只属于其中一个类别，项目的目标是利用CNN将图片划分到它们正确的类别中。因为从零开始训练一个卷积网络需要花费大量的时间和资源，这里会使用已经身经百战的CNN模型，并利用这些模型导出数据集的深度特征。然后基于这些深度特征来训练自定义的模型，并在最后验证自定的模型。

### 评价指标
TODO

## 分析
### 数据的探索
项目的数据集能直接从Kaggle[3]上下载。Kaggle一共提供了3个文件，train.zip是训练集，test.zip是测试集以及一个csv格式的submission文件。训练集包含了2个类别的图片，猫和狗的图片各12500张。测试集包含了12500图片。训练集下数据的通过文件名提供了标签，文件名的格式是{类别}.{序号}.jpg，比如猫的图片名称为cat.1.jpg，狗的图片名称为dog.1.jpg。因此我们可以根据文件名，将其进行分类。而测试集的数据中并没有提供标签，文件名的格式是{序号}.jpg，比如1.jpg或2.jpg。同时数据集中的图片大小都是不一致的，需要在后续的步骤中处理这个问题。下面展示的是的训练集和测试集的部分数据。

训练集中猫的图片
![train set cat](images/train_set_cat.PNG)

训练集中狗的图片
![train set cat](images/train_set_dog.PNG)

测试集中的图片
![train set cat](images/test_set.PNG)

### 算法和技术
首先项目中会使用VGG[4]，ResNet[5]，InceptionV3[6]，Xception[7]等模型，它们简单的介绍如下：
- VGG: VGG网络架构比较简单，遵循基本卷积网络的原型布局，一系列卷积层、最大池化层和激活层，最后还有一些全连接的分类层。而VGG16和VGG19中的16和19分别代表了权重的层数，如下图的D列及E列。VGGNet主要有2个缺点，一个训练起来很慢，非常耗时。另外一个是模型weights很多，导致模型很大，训练时也会占用更多的磁盘和带宽。VGG16大约533MB，VGG19大约574MB。
- ResNet：也叫残差网络，诞生于一个美丽而简单的观察：为什么非常深度的网络在增加更多层时会表现得更差？作者将这些问题归结成了一个单一的假设：直接映射是难以学习的。而且他们提出了一种修正方法：不再学习从 x 到 H(x) 的基本映射关系，而是学习这两者之间的差异，也就是「残差（residual）」,假设残差为 F(x)=H(x)-x，那么现在我们的网络不会直接学习 H(x) 了，而是学习 F(x)+x。ResNet的模型虽然很深但因为使用了很多平均池化层仍然比VGG小许多，大概只有100MB左右。
- Inception：第一版叫做GoogleNet，V3，V4是不同的版本名称。这个网络的特点在同一层同时采用3x3，1x1，5x5的Filter及一个3x3的max pooling来提取多个特征，再将这些不同特征组装。如果ResNet是为了更深，那么 Inception 家族就是为了更宽。这种模型架构的信息密度更大了，这就带来了一个突出的问题：计算成本大大增加。所以作者使用 1×1 卷积来执行降维，一个 1×1 卷积一次仅查看一个值，但在多个通道上，它可以提取空间信息并将其压缩到更低的维度。Inception的模型比VGGNet和ResNet都要小，大概只有96MB。 
- Xception：Xception的意思是extreme inception，而且正如其名字表达的那样，它将 Inception 的原理推向了极致。它的假设是：「跨通道的相关性和空间相关性是完全可分离的，最好不要联合映射它们。」Xception 不再只是将输入数据分割成几个压缩的数据块，而是为每个输出通道单独映射空间相关性，然后再执行 1×1 的深度方面的卷积来获取跨通道的相关性。Xception的模型最小大概只有89MB。

有了这些基础模型，我会基于这些基础模型来进行迁移学习，同时我会使用谷歌的TensorFlow[8]作为神经网络的平台，并使用Keras[9]的API来构建并训练CNN。项目中一共采用了2个方案，方案1是基于单个模型的迁移学习，即只采用一个模型导出的深度特征来进行训练；方案2是基于当前所有模型的迁移学习，即合并所有模型的深度特征来进行训练。

### 探索性可视化
TODO 可视化深度特征

### 基准模型
这里项目的要求是达到Kaggle的排行榜的前10%，所以本项目的要求是在测试集上的LogLoss表现要低于0.06127。

## 方法
### 数据预处理
项目中一共对数据做了2次预处理，第一次因为flow_from_directory API的要求，对训练集中的2类图片进行了一个分类。根据图片的文件名称将2猫和狗的图片分别放在cat和dog的子文件夹中。同时把测试集的数据全部移动到一个子文件夹中。第二次是在导出深度特征导时，利用Lambda函数对输入的图片全部加了一个model.preproccess的预处理操作。比如VGG16，用的是vgg16.preprocess_input方法，而Xception用的是xception.preprocess_input方法（实际上调用的都是同一个方法，xception和inception指定了后台只能是TensorFlow）这样做的目的是将我们的0-255的RGB值缩小至-1到1的范围，好处是使得模型可以更快的收敛，缩短训练时间。

### 执行过程
首先需使用Kearas的Application API[10]来获取这些模型，比如通过keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')来获取VGG16的模型，include_top要设置成False，代表获取的模型不包含最后的全连接层。weights要设置层‘imagenet’代表获取的权重是pre-train过的。

其次使用Keras的图片预处理API ImageDataGenerator[11]中的方法flow_from_directory加载数据集。通过参数target size来调整图片的大小，因为不同的CNN需要的输入的图片大小会有所区别，比如VGG和ResNet要求的图片大小为（224,224）而Xception和Inception需要的图片大小为（299,299）。然后调用model的方法predict_generator来进行预测。得到所有训练集基于当前基础模型的一个深度特征。为了方便后期调试，这里我使用了HDF5 for python的API[12]把所有模型的深度特征保存在本地。

之后项目自定义一个模型来基于深度特征进行迁移学习。自定义模型一共只包含了2层，第一层为BatcNormalization层，目的是为了防止过拟合。第二层为Dense层，激活函数为sigmod，目的是为了做最后的分类。构建完模型后，还需要调用model的compile方法来编译模型。编译模型时，我传递了3个参数，第一个是optimizer代表的是优化器，项目中使用的是Adam[13],这是一款常见的优化器，特点是计算效率较高，对内存要求比较少，默认参数的超参数大多数情况下基本够用了。第二个是loss代表的是损失函数，因为是二分类我使用的是binary_crossentropy。第三个是metrics代表的是衡量指标，这里可以传递的是数组，可以有1个或多个指标，我使用的是['accuracy']。

模型compile后开始训练模型，这里不需手动将训练集划分为训练数据和验证数据，Keras的Model API[14]提供一个训练模型的方法fit。而fit中提供了一个参数validation_split可以自动帮助我们来划分训练数据和验证数据。比如validation_split = 0.2时，代表80%的数据用于训练，而剩下20%数据用于验证。最后当模型训练完成后，使用测试集的深度特征作为输入，利用model的predict方法来进行预测。将最终结果写入到需要提交的csv文件当中，提交至Kaggle。

项目中我手动调参并训练了以下场景，记录了以下数据：
- 基于VGG16的迁移学习是我最早调试的，刚开始Adam优化器全部采用默认参数并且把epoch设置为10。10次以内训练集和验证集的accuracy是不断上升的，loss也是不断下降的。因此我将Epoch加到了50次，虽然最大的accuracy和最小的loss都表现的比上次要好。但是通过verbose=1打印出来的数据显示，test loss和accuracy浮动很大，忽高忽低。推测learning rate稍微偏大了，于是下调了learning rate至0.0001并增加epoch次数,分别尝试了200，300次以及400次。epoch增加至400次时，出现了轻微的过拟合情况。因此，最终我选择了epoch=300，lr=0.0001的方案。

    |epoch|optimizer|lr|decay|train max acc|val max acc|train min loss|val min loss|time(sec)|
    | ---:| -----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
    | 10  | Adam | 0.001  | 0.0 | 0.9832 | 0.9694 | 0.0474 | 0.0790 | 6.25    |
    | 50  | Adam | 0.001  | 0.0 | 0.9902 | 0.9774 | 0.0284 | 0.0606 | 32.41   |
    | 200 | Adam | 0.0001 | 0.0 | 0.9897 | 0.9744 | 0.0310 | 0.0685 | 175.38  |
    | 300 | Adam | 0.0001 | 0.0 | 0.9908 | 0.9740 | 0.0287 | 0.0675 | 180.71  |
    | 400 | Adam | 0.0001 | 0.0 | 0.9912 | 0.9744 | 0.0269 | 0.0676 | 257.19  |

- 基于VGG19的迁移学习模型，因为VGG19和VGG16非常相似，所以我直接采用了VGG16的方案，设置epoch=300以及lr=0.0001。但是从训练日志从看出，大概从200次开始验证集的acc开始略微下降，loss开始略微上升，出现了一点过拟合的现象。推测可能是因为VGG19比VGG16多3层权重，表达的内容可以更加复杂，所以基于VGG19的迁移学习的模型并不需要那么多的训练次数，否则会出现过拟合的情况。因此，我将epoch设置为200，重新训练。虽然最大acc和最小loss表现的没上次好，但至少没有出现过拟合的情况，这样的模型泛化能力会更好。因此，我最终选择了epoch=200, lr=0.0001的方案。

    |epoch|optimizer|lr|decay|train max acc|val max acc|train min loss|val min loss|time(sec)|
    | --- | -----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
    | 200 | Adam | 0.0001 | 0.0 | 0.9902 | 0.9752 | 0.0301 | 0.0619 | 188.52 |
    | 300 | Adam | 0.0001 | 0.0 | 0.9911 | 0.9774 | 0.0272 | 0.0602 | 188.52 |

- 基于ResNet50的迁移学习模型，ResNet顾名思义一共有50层，比VGG的层数更多。所以这里我直接尝试了50，100次的epoch以及lr=0.0001。当epoch=100的时候，大概在40-50次的时候开始出现过拟合的情况。因此，我最终选择了epoch=50, lr=0.0001的方案。

    |epoch|optimizer|lr|train max acc|val max acc|train min loss|val min loss|time(sec)|
    | --- | -----:|-----:|-----:|-----:|-----:|-----:|-----:|
    | 50  | Adam | 0.0001 | 0.9940 | 0.9826 | 0.0188 | 0.0484 | 45.58 |
    | 100 | Adam | 0.0001 | 0.9980 | 0.9824 | 0.0096 | 0.0459 | 69.26 |
    
- 基于InceptionV3的迁移学习模型，越后期的模型越成熟，因此learning rate需要设置的更低，这里我直接尝试了lr=0.00005，epoch=15，30，50的场景。从训练日志中可以看出，val loss在15代之后基本不在下降了，不停的上下浮动。因此，我最终选择了epoch=15, lr=0.00005的方案。

    |epoch|optimizer|lr|train max acc|val max acc|train min loss|val min loss|time(sec)|
    | --- | -----:|-----:|-----:|-----:|-----:|-----:|-----:|
    | 15 | Adam | 0.00005 | 0.9934 | 0.9938 | 0.0260 | 0.0285 | 14.50 |
    | 30 | Adam | 0.00005 | 0.9946 | 0.9928 | 0.0190 | 0.0256 | 29.68 |
    | 50 | Adam | 0.00005 | 0.9961 | 0.9912 | 0.0131 | 0.0285 | 45.40 |

- 基于Xception的迁移学习模型，根据InceptionV3的经验，这里直接尝试了lr=0.00005，epoch=15，30的场景。和Inception类似，Xception也是在15代左右开始val loss基本不在下降，开始不停的上下浮动。因此，我最终选择了epoch=15, lr=0.00005的方案。

    |epoch|optimizer|lr|train max acc|val max acc|train min loss|val min loss|time(sec)|
    | --- | -----:|-----:|-----:|-----:|-----:|-----:|-----:|
    | 15 | Adam | 0.00005 | 0.9939 | 0.9938 | 0.0228 | 0.0251 | 16.02 |
    | 30 | Adam | 0.00005 | 0.9948 | 0.9924 | 0.0179 | 0.0263 | 26.92 |

- 基于所有模型的迁移学习，因为集成了所有模型的权重，所以我首先尝试了Inception和Xception的最优方案。即lr=0.00005，epoch=15的场景。结果好的出奇，val的loss第一次降到了0.02以下。因此，我最终选择了epoch=15, lr=0.00005的方案。

    |epoch|optimizer|lr|train max acc|val max acc|train min loss|val min loss|time(sec)|
    | --- | -----:|-----:|-----:|-----:|-----:|-----:|-----:|
    | 15 | Adam | 0.00005 | 0.9961 | 0.9934 | 0.0130 | 0.0196 | 16.02 |

综上，基于所有模型的迁移学习，因该会是表现最好的。在训练集上取得了最高的accuracy，0.9961，并在验证集上取得了第二高的accuracy，0.9934。并且在训练集上取得了最低的loss 0.0196，在验证集上也取得了最低的loss 0.0196。同时基于InceptionV3和Xception迁移学习的模型表现的也比较好。


### 完善
项目中一共完善了2个地方来提升了模型得表现及最终得评分。

- 超参数优化：以基于VGG16进行迁移学习为例，使用的优化器是Adam，刚开始得时候直接使用了Adam的默认超参数来训练，模型只训练10次，发现模型还有提高的空间。于是增加训练次数。但是当训练次数超过10次以后，模型在验证集的loss上都会出现明显的抖动，一会高，一会低，而不是持续的降低。推测原因是，Adma优化器默认的学习速度在这个场景下还是偏高了。于是调整了模型的超参数learning rate并且适当的增加了训练次数。最终对于基于VGG16进行迁移迁移学习的模型在epoch=300以及lr=0.0001的时候表现比较好。

- 输出完善：Kaggle官方使用的Logloss来进行评估，当预测结果错误的时候Logloss会出现无穷大的情况，这样会使得Logloss计算出比较大的值，从而降低排名。所以我们这里对预测值做了一个区间限制[0.005-0.995]。做完限制后Logloss降低的非常明显。比如基于VGG16迁移学习的模型，在没有对Logloss最大最小值做限制时，在测试集上的Logloss为0.10158，做了限制后降低到了0.08130，整整降低了0.02028；

## 结果
### 模型的评价与验证
基于测试集进行预测并将结果按照Kaggle的格式要求生成一份csv文件，将文件提交至Kaggle后，得到以下评估结果（如下图）。
- 基于所有模型的迁移学习，Logloss为0.04128
- 基于Xception迁移学习的模型，LogLoss为0.04390
- 基于InceptionV3迁移学习的模型，LogLoss为0.04631 
- 基于ResNet50迁移学习的模型，LogLoss为0.05707 
- 基于ResNet19迁移学习的模型，LogLoss为0.06498  
- 基于VGG16迁移学习的模型，LogLoss为0.07051 
![predictions](images/predictions.png)

### 合理性分析
在这个部分，你需要利用一些统计分析，把你的最终模型得到的结果与你的前面设定的基准模型进行对比。你也分析你的最终模型和结果是否确确实实解决了你在这个项目里设定的问题。你需要考虑：
- _最终结果对比你的基准模型表现得更好还是有所逊色？_
- _你是否详尽地分析和讨论了最终结果？_
- _最终结果是不是确确实实解决了问题？_


## 结论

### 结果可视化


在这一部分，你需要用可视化的方式展示项目中需要强调的重要技术特性。至于什么形式，你可以自由把握，但需要表达出一个关于这个项目重要的结论和特点，并对此作出讨论。一些需要考虑的：
- _你是否对一个与问题，数据集，输入数据，或结果相关的，重要的技术特性进行了可视化？_
- _可视化结果是否详尽的分析讨论了？_
- _绘图的坐标轴，标题，基准面是不是清晰定义了？_

- 深度特征导出，项目里基于当前的训练集和测试集一共导出了5个模型的深度特征,每个文件都包括了训练集和测试集深度特征，详情如下：

    |模型名称|总耗时（单位：秒）|深度特征文件大小（单位：MB）|
    | ------ | -----:|-----:|
    | VGG16    | 205.36 | 73.3 |
    | VGG19    | 232.02 | 73.3 |
    | ResNet50 | 210.49 | 293 |
    | InceptionV3 | 274.99 | 293|
    | Xception    | 416.50 | 293|

  ![VGG16 plot](images/vgg16_plot.PNG)
   
  ![VGG19 plot](images/vgg19_plot.PNG)
   
  ![ResNet50 plot](images/resnet50_plot.PNG)
   
  ![InceptionV3 plot](images/inceptionv3_plot.PNG)
  
  ![Xception plot](images/xception_plot.png)
  
  ![all model plot](images/all_model_features_plot.PNG)


### 对项目的思考
在这一部分，你需要从头到尾总结一下整个问题的解决方案，讨论其中你认为有趣或困难的地方。从整体来反思一下整个项目，确保自己对整个流程是明确掌握的。需要考虑：
- _你是否详尽总结了项目的整个流程？_
- _项目里有哪些比较有意思的地方？_
- _项目里有哪些比较困难的地方？_
- _最终模型和结果是否符合你对这个问题的期望？它可以在通用的场景下解决这些类型的问题吗？_


### 需要作出的改进
在这一部分，你需要讨论你可以怎么样去完善你执行流程中的某一方面。例如考虑一下你的操作的方法是否可以进一步推广，泛化，有没有需要作出变更的地方。你并不需要确实作出这些改进，不过你应能够讨论这些改进可能对结果的影响，并与现有结果进行比较。一些需要考虑的问题：
- _是否可以有算法和技术层面的进一步的完善？_
- _是否有一些你了解到，但是你还没能够实践的算法和技术？_
- _如果将你最终模型作为新的基准，你认为还能有更好的解决方案吗？_

----------
** 在提交之前， 问一下自己... **

- 你所写的项目报告结构对比于这个模板而言足够清晰了没有？
- 每一个部分（尤其**分析**和**方法**）是否清晰，简洁，明了？有没有存在歧义的术语和用语需要进一步说明的？
- 你的目标读者是不是能够明白你的分析，方法和结果？
- 报告里面是否有语法错误或拼写错误？
- 报告里提到的一些外部资料及来源是不是都正确引述或引用了？
- 代码可读性是否良好？必要的注释是否加上了？
- 代码是否可以顺利运行并重现跟报告相似的结果？


引用
[1] Wiki page, Feature detection:en.wikipedia.org/wiki/Feature_detection_(computer_vision)

[2] Kaggle Project Dogs vs Cats:www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

[3] Dogs vs Cats Datasets: www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

[4] Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv:1409.1556, 2014

[5] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

[6] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. Rethinking the Inception Architecture for Computer Vision. arXiv:1512.00567, 2015

[7] François Chollet. Xception: Deep Learning with Depthwise Separable Convolutions. arXiv:1610.02357, 2016

[9] TensorFlowofficial site:tensorflow.google.cn

[9] Kerasofficial site:keras.io

[10] Keras applications introduction:keras.io/applications/

[11] Keras ImageDataGenerator introduction:keras.io/preprocessing/image/

[12] H5py Datasets API：docs.h5py.org/en/latest/high/dataset.html

[13] Diederik Kingma, Jimmy Ba. Adam: A Method for Stochastic Optimization. arXiv:1412.6980, 2014

[14] Keras model API introduction: keras.io/models/model/

[15] Kaggle Team, Dogs vs. Cats Redux Playground Competition, Winner's Interview
