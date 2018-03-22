# 项目背景
本次我选择的项目为图片识别类型的项目dog_vs_cat。这个项目是Kaggle 2013年发起的一个竞赛项目，目前已经关闭。截止目前（2018-03-22）已经有1334竞赛者参加了这个项目。从2012年的AlexNet开始，CNN已经逐渐成为图片识别领域的佼佼者。在同样的ImageNet的数据集上人眼的识别率约为5.1%，而排名靠前的ResNet的准确率已经提升到了3.57，超过了人眼。所以dog_vs_cat这个项目是完全可以通过CNN来解决的。计划是做完所有毕业项目，因为这个项目排在第一个所以就选择它作为毕业项目。

[Kaggle project dog_vs_cat](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
[ResNet](https://arxiv.org/abs/1512.03385)


# 问题描述
本项目需要解决的问题是识别图片中的动物是狗还是猫，是一个监督学习的二分类问题。 我们可以利用成熟度很高的CNN模型，例如:ResNet,Xception，Inception等，然后再基于这些卷积网络进行迁移学习，从而构建我们自己的CNN。本次衡量的标准主要考量模型的训练时间，预测时间以及模型在测试集上的准确率并且结果是可以重现的。


# 输入数据
本次的数据来源为Kaggle的dog_vs_cat项目，可以直接从项目的data页面下载。一共需要下载2个文件train.zip和test.zip，train.zip解压后一共有25000张猫和狗的图片，文件名格式为{类型}.{序号}.jpg，比如猫的图片为cat.0.jpg，狗的图片为dog.1.jpg。测试集解压后一共有12500张猫和狗的图片，文件格式为{序号}.jpg。从train.zip解压出来的图片我们将作为训练集使用，从test.zip解压出来的数据我们将作为测试集使用并且按照Kaggle的规则生成可提交的CSV文件。

[dog_vs_cat data](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)


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
